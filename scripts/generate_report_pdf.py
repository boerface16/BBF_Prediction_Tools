"""Generate a blog-style PDF report summarizing model results, keeper evaluations, and 2025 retrospective."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import pandas as pd
import numpy as np
import unicodedata
from fpdf import FPDF
from pathlib import Path


def strip_accents(s):
    """Remove accents/diacritics from string for PDF compatibility."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )


def norm_name(s):
    """Normalize player name for fuzzy matching."""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn').lower().strip()


def parse_pipe_md(filepath):
    """Parse pipe-delimited markdown table (Steamer format)."""
    with open(filepath, encoding='utf-8') as f:
        lines = [l for l in f.read().split('\n')
                 if '|' in l and not set(l.replace('|','').replace('-','').replace(' ','')).issubset({'-',''})]
    header = [c.strip() for c in lines[0].split('|')[1:-1]]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ('Name', 'Team'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_tsv_md(filepath):
    """Parse tab-separated FanGraphs export (ZiPS/ZiPS-DC format)."""
    with open(filepath, encoding='utf-8') as f:
        lines = [l for l in f.read().split('\n') if '\t' in l]
    header = [c.strip() for c in lines[0].split('\t')]
    rows = [dict(zip(header, [c.strip() for c in l.split('\t')])) for l in lines[1:] if l.strip()]
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ('Name', 'Team'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_article_hitters(filepath):
    """Parse plain-text hitter ranked list: '1.  Name -- Team, Pos'"""
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().split('\n')
    rows = []
    for line in lines:
        m = re.match(r'^(\d+)\.\s+(.+?)\s+--', line)
        if m:
            rows.append({'article_rank': int(m.group(1)), 'Name': m.group(2).strip()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['name_key'] = df['Name'].apply(norm_name)
    return df


def parse_article_pitchers(filepath):
    """Parse plain-text pitcher ranked list: '1.  Name (SP)' or '1.  Name (RP)'"""
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().split('\n')
    rows = []
    for line in lines:
        m = re.match(r'^(\d+)\.\s+(.+?)\s*\((SP|RP)\)', line)
        if m:
            rows.append({'article_rank': int(m.group(1)), 'Name': m.group(2).strip()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['name_key'] = df['Name'].apply(norm_name)
    return df


def build_fg_dashboard(bat_rank, pit_rank, root):
    """Build FanGraphs comparison dashboard DataFrames (bat_dash, pit_dash)."""
    fg_dir = root / "data" / "fg_predictions"

    stm_bat = parse_pipe_md(fg_dir / "steamer" / "top_50_steamer_batters.md")
    stm_pit = parse_pipe_md(fg_dir / "steamer" / "top_50_steamer_pitchers.md")
    zps_bat = parse_tsv_md(fg_dir / "zips" / "top_50_zips_batters.md")
    zps_pit = parse_tsv_md(fg_dir / "zips" / "top_50_zips_pitchers.md")
    zdc_bat = parse_tsv_md(fg_dir / "zips_dc" / "top_50_zips_dc_batters.md")
    zdc_pit = parse_tsv_md(fg_dir / "zips_dc" / "top_50_zips_dc_pitchers.md")
    art_bat = parse_article_hitters(fg_dir / "articles" / "top_50_hitters_FG.md")
    art_pit = parse_article_pitchers(fg_dir / "articles" / "top_50_pitchers_FG.md")

    for df in [stm_bat, stm_pit, zps_bat, zps_pit, zdc_bat, zdc_pit]:
        df['name_key'] = df['Name'].apply(norm_name)

    # Extract per-system ranks
    stm_bat['stm_rank'] = stm_bat['#'] if '#' in stm_bat.columns else stm_bat['Rank']
    stm_pit['stm_rank'] = stm_pit['Rank'] if 'Rank' in stm_pit.columns else stm_pit['#']
    zps_bat['zps_rank'] = zps_bat['#']
    zps_pit['zps_rank'] = zps_pit['#']
    zdc_bat['zdc_rank'] = zdc_bat['#']
    zdc_pit['zdc_rank'] = zdc_pit['#']

    # Consensus batter FG table
    BAT_STAT_COLS = ['HR', 'R', 'RBI', 'SB', 'AVG', 'wRC+', 'ADP']
    _BAT_RENAME = {c: c.lower().replace('/','').replace('+','plus') for c in BAT_STAT_COLS}

    bat_fg = stm_bat[['name_key'] + BAT_STAT_COLS].rename(
        columns={c: f'stm_{_BAT_RENAME[c]}' for c in BAT_STAT_COLS})
    for prefix, df in [('zps', zps_bat), ('zdc', zdc_bat)]:
        renamed = df[['name_key'] + BAT_STAT_COLS].rename(
            columns={c: f'{prefix}_{_BAT_RENAME[c]}' for c in BAT_STAT_COLS})
        bat_fg = bat_fg.merge(renamed, on='name_key', how='outer')

    for c in [v for v in _BAT_RENAME.values()]:
        bat_fg[f'fg_{c}'] = bat_fg[[f'stm_{c}', f'zps_{c}', f'zdc_{c}']].mean(axis=1)

    bat_fg = bat_fg.merge(art_bat[['name_key', 'article_rank']], on='name_key', how='left')
    bat_fg = bat_fg.merge(stm_bat[['name_key', 'stm_rank']], on='name_key', how='left')
    bat_fg = bat_fg.merge(zps_bat[['name_key', 'zps_rank']], on='name_key', how='left')
    bat_fg = bat_fg.merge(zdc_bat[['name_key', 'zdc_rank']], on='name_key', how='left')

    # Consensus pitcher FG table
    PIT_STAT_COLS = ['IP', 'SO', 'K/9', 'ERA', 'WHIP', 'ADP']
    _PIT_RENAME = {c: c.lower().replace('/','').replace('+','plus') for c in PIT_STAT_COLS}

    pit_fg = stm_pit[['name_key'] + PIT_STAT_COLS].rename(
        columns={c: f'stm_{_PIT_RENAME[c]}' for c in PIT_STAT_COLS})
    for prefix, df in [('zps', zps_pit), ('zdc', zdc_pit)]:
        renamed = df[['name_key'] + PIT_STAT_COLS].rename(
            columns={c: f'{prefix}_{_PIT_RENAME[c]}' for c in PIT_STAT_COLS})
        pit_fg = pit_fg.merge(renamed, on='name_key', how='outer')

    for c in [v for v in _PIT_RENAME.values()]:
        pit_fg[f'fg_{c}'] = pit_fg[[f'stm_{c}', f'zps_{c}', f'zdc_{c}']].mean(axis=1)

    pit_fg = pit_fg.merge(art_pit[['name_key', 'article_rank']], on='name_key', how='left')
    pit_fg = pit_fg.merge(stm_pit[['name_key', 'stm_rank']], on='name_key', how='left')
    pit_fg = pit_fg.merge(zps_pit[['name_key', 'zps_rank']], on='name_key', how='left')
    pit_fg = pit_fg.merge(zdc_pit[['name_key', 'zdc_rank']], on='name_key', how='left')

    # Merge with our rankings
    bat_rank['name_key'] = (bat_rank['first'] + ' ' + bat_rank['last']).apply(norm_name)
    pit_rank['name_key'] = (pit_rank['first'] + ' ' + pit_rank['last']).apply(norm_name)

    bat_dash = bat_rank[['overall_rank', 'last', 'first', 'primary_pos',
                          'projected_pts_2026', 'pts_2025', 'PAR', 'tier', 'risk_flags', 'name_key']]\
        .merge(bat_fg[['name_key', 'fg_hr', 'fg_r', 'fg_rbi', 'fg_sb', 'fg_avg', 'fg_wrcplus', 'fg_adp',
                        'article_rank', 'stm_rank', 'zps_rank', 'zdc_rank']],
               on='name_key', how='left')

    pit_dash = pit_rank[['overall_rank', 'last', 'first',
                          'projected_pts_2026', 'pts_2025', 'PAR', 'tier', 'risk_flags', 'name_key']]\
        .merge(pit_fg[['name_key', 'fg_ip', 'fg_so', 'fg_k9', 'fg_era', 'fg_whip', 'fg_adp',
                        'article_rank', 'stm_rank', 'zps_rank', 'zdc_rank']],
               on='name_key', how='left')

    # Compute rank diff and signal
    bat_dash['adp_rank'] = bat_dash['fg_adp'].round(0)
    bat_dash['rank_diff'] = bat_dash['adp_rank'] - bat_dash['overall_rank']
    bat_dash['signal'] = bat_dash['rank_diff'].apply(
        lambda x: 'VALUE' if x > 20 else ('FADE' if x < -20 else '') if pd.notna(x) else '')

    pit_dash['adp_rank'] = pit_dash['fg_adp'].round(0)
    pit_dash['rank_diff'] = pit_dash['adp_rank'] - pit_dash['overall_rank']
    pit_dash['signal'] = pit_dash['rank_diff'].apply(
        lambda x: 'VALUE' if x > 20 else ('FADE' if x < -20 else '') if pd.notna(x) else '')

    return bat_dash, pit_dash


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
OUT_PDF = ROOT / "docs" / "fantasy_baseball_2026_report.pdf"


class ReportPDF(FPDF):
    """Custom PDF with headers, footers, and helper methods."""

    def header(self):
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Fantasy Baseball H2H Points Model  - 2026 Season Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def callout_box(self, text, color=(230, 242, 255)):
        self.set_fill_color(*color)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(40, 40, 40)
        x, y = self.get_x(), self.get_y()
        self.multi_cell(0, 6, f"  {text}", fill=True)
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None, font_size=8):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Check if table fits on current page, otherwise add page
        estimated_h = 7 + 6 * len(rows) + 3
        if self.get_y() + estimated_h > 270:
            self.add_page()
        # Header
        self.set_font("Helvetica", "B", font_size)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", font_size)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "C" if i > 0 else "L"
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def safe_image(self, path, w=180):
        """Add image if it exists, with page break check."""
        if Path(path).exists():
            img_h = w * 0.6  # estimate
            if self.get_y() + img_h > 270:
                self.add_page()
            self.image(str(path), x=15, w=w)
            self.ln(5)
        else:
            self.body_text(f"[Image not found: {Path(path).name}]")


def _fmt(val, fmt_str=".0f"):
    """Format a numeric value, returning '-' for NaN."""
    if pd.isna(val):
        return "-"
    return f"{val:{fmt_str}}"


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Load data
    bat_rank = pd.read_csv(ROOT / "outputs" / "draft_rankings_batters.csv")
    pit_rank = pd.read_csv(ROOT / "outputs" / "draft_rankings_pitchers.csv")
    keeper = pd.read_csv(ROOT / "outputs" / "keeper_rankings.csv")

    # Build FG comparison data
    bat_dash, pit_dash = build_fg_dashboard(bat_rank, pit_rank, ROOT)

    # ===================== TITLE PAGE =====================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, "Fantasy Baseball 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, "H2H Points Model Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "XGBoost + LightGBM Ensemble with Multi-Seed Optuna Tuning", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # Summary box
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6, (
        "This report summarizes the results of a machine learning pipeline that predicts ESPN H2H fantasy "
        "baseball points for the 2026 season. The model uses historical Retrosheet game logs, Statcast metrics, "
        "and FanGraphs advanced stats from 2017-2024 to project player performance."
    ), align="L")
    pdf.ln(5)

    # TL;DR callout
    n_bat_value = len(bat_dash[bat_dash['signal'] == 'VALUE'])
    n_pit_value = len(pit_dash[pit_dash['signal'] == 'VALUE'])
    n_bat_fade = len(bat_dash[bat_dash['signal'] == 'FADE'])
    n_pit_fade = len(pit_dash[pit_dash['signal'] == 'FADE'])
    auto_keepers = keeper[keeper['recommendation'] == 'AUTO-KEEP']
    auto_keep_names = ", ".join(strip_accents(f"{r['first']} {r['last']}") for _, r in auto_keepers.iterrows())

    pdf.callout_box(
        f"TL;DR\n"
        f"- Model projects {len(bat_rank)} batters and {len(pit_rank)} pitchers for 2026 ESPN H2H points\n"
        f"- 2025 retrospective: batter Spearman 0.543, pitcher 0.491 (70% within +/-100 pts)\n"
        f"- {n_bat_value} batter + {n_pit_value} pitcher VALUE picks vs FanGraphs ADP "
        f"({n_bat_fade} + {n_pit_fade} potential fades)\n"
        f"- Auto-keep recommendations: {auto_keep_names}",
        color=(220, 240, 220)
    )

    # ===================== SECTION 1: MODEL OVERVIEW =====================
    pdf.add_page()
    pdf.section_title("1. Model Overview")

    pdf.body_text(
        "The model is an ensemble of XGBoost and LightGBM regressors, blended via Ridge regression. "
        "Separate models are trained for batters, starting pitchers (SP), and relief pitchers (RP). "
        "Key design choices:"
    )

    pdf.bold_text("Multi-Seed Optuna Tuning")
    pdf.body_text(
        "Each model runs Optuna hyperparameter search across 5 random seeds (42, 123, 456, 789, 1010) "
        "with 100 trials per seed per model type. The seed producing the median Spearman correlation is "
        "selected  - this avoids both unlucky and lucky draws, giving stable, reproducible results."
    )

    pdf.bold_text("Training Data Filters")
    pdf.body_text(
        "Training uses 2017-2023 seasons (2015-2016 excluded  - missing Statcast arsenal features). "
        "Pitchers must have 50+ IP (150 outs) to qualify, filtering volatile mop-up arms. "
        "COVID 2020 rows are downweighted 50%. Features are shifted: year N stats predict year N+1 points."
    )

    pdf.bold_text("Time-Series Cross-Validation")
    pdf.body_text(
        "3-fold expanding-window TSCV: train through 2020 / validate 2021, train through 2021 / validate 2022, "
        "train through 2022 / validate 2023. This prevents future data leakage and mimics real forecasting."
    )

    # ===================== SECTION 2: RETROSPECTIVE =====================
    pdf.add_page()
    pdf.section_title("2. 2025 Season Retrospective")

    pdf.body_text(
        "The true test of any projection system: how well did it predict the season that already happened? "
        "The model used 2024 features to predict 2025 ESPN fantasy points, then compared to actual results."
    )

    # Metrics table
    pdf.subsection_title("Overall Metrics")
    headers = ["Metric", "Batters", "Pitchers"]
    rows = [
        ["Players Evaluated", "265", "217"],
        ["MAE (avg error)", "76.5 pts", "77.0 pts"],
        ["RMSE", "97.9", "101.5"],
        ["Spearman Correlation", "0.543", "0.491"],
        ["R-squared", "0.322", "0.318"],
        ["Within +/-50 pts", "43.4%", "41.9%"],
        ["Within +/-100 pts", "69.8%", "71.9%"],
        ["Top-25 Precision", "48.0%", "48.0%"],
        ["Top-50 Precision", "54.0%", "58.0%"],
        ["Bias", "+7.7 pts", "-3.2 pts"],
    ]
    pdf.add_table(headers, rows, col_widths=[60, 65, 65])

    pdf.callout_box(
        "KEY INSIGHT: The pitcher model improved dramatically after data cleaning  - Spearman jumped from "
        "0.353 to 0.491, and MAE dropped from 93 to 77. Raising the IP threshold to 50 IP removed noisy "
        "low-usage pitchers from both training and evaluation, improving signal quality."
    )

    # Scatter plots
    pdf.subsection_title("Predicted vs. Actual  - Batters")
    pdf.safe_image(FIG_DIR / "model_evaluation" / "batter_2025_scatter_labeled.png", w=170)

    pdf.add_page()
    pdf.subsection_title("Predicted vs. Actual  - Pitchers")
    pdf.safe_image(FIG_DIR / "model_evaluation" / "pitcher_2025_scatter_labeled.png", w=170)

    # Biggest misses
    pdf.subsection_title("Notable Misses (and Why)")
    pdf.body_text(
        "Large prediction errors often stem from injuries, role changes, or breakout seasons that no "
        "model can foresee. Some of the biggest misses in 2025:"
    )

    miss_headers = ["Player", "Predicted", "Actual", "Error", "Likely Cause"]
    miss_rows = [
        ["Geraldo Perdomo", "127", "512", "+385", "Breakout year  - no prior signal"],
        ["Bryan Woo", "173", "524", "+351", "Breakout  - emerged as ace"],
        ["Carlos Rodon", "228", "522", "+295", "Bounce-back from injury"],
        ["Blake Snell", "459", "164", "-295", "Injury / ineffectiveness"],
        ["Mike King", "413", "171", "-242", "Role / performance collapse"],
        ["JJ Bleday (bat)", "467", "157", "-310", "Regression from 2024 outlier"],
    ]
    pdf.add_table(miss_headers, miss_rows, col_widths=[35, 25, 25, 20, 85], font_size=8)

    pdf.callout_box(
        "IMPORTANT: A MAE of ~77 points means the model's average error is about the difference between "
        "a borderline starter and a solid regular (e.g., 280 vs 357 pts). For context, the gap between "
        "the #1 and #50 ranked batter is ~200 points. Most large errors involve injuries or breakouts "
        "that no public projection system captures well.",
        color=(255, 243, 224)
    )

    # Tier calibration
    pdf.add_page()
    pdf.subsection_title("Tier Calibration  - Are Rankings Useful?")
    pdf.body_text(
        "Beyond raw accuracy, we want to know: do players ranked higher actually score more points? "
        "The tier calibration plots show average actual points by predicted tier."
    )
    pdf.safe_image(FIG_DIR / "model_evaluation" / "batter_2025_tier_calibration.png", w=170)
    pdf.safe_image(FIG_DIR / "model_evaluation" / "pitcher_2025_tier_calibration.png", w=170)

    # Position MAE
    pdf.add_page()
    pdf.subsection_title("Accuracy by Position")
    pdf.body_text(
        "Some positions are inherently harder to predict. Catchers and relievers have smaller sample sizes "
        "and more volatile year-over-year performance."
    )
    pdf.safe_image(FIG_DIR / "model_evaluation" / "batter_2025_position_mae.png", w=170)
    pdf.safe_image(FIG_DIR / "model_evaluation" / "pitcher_2025_position_mae.png", w=170)

    # ===================== SECTION 3: KEEPERS =====================
    pdf.add_page()
    pdf.section_title("3. Keeper Evaluation")

    pdf.body_text(
        "The keeper evaluator combines two signals to recommend which players to keep for 2026:\n\n"
        "  - Trajectory Score (50%): Year-over-year trend in points and points-per-game\n"
        "  - ML Projection (50%): Model's predicted 2026 fantasy points\n\n"
        "Both scores are normalized 0-1 and combined. Players are classified as AUTO-KEEP (>0.80), "
        "KEEP (>0.60), BORDERLINE (>0.40), or CUT (<0.40)."
    )

    # Keeper table
    pdf.subsection_title("Keeper Rankings")
    k_headers = ["Rec", "Player", "Pos", "2025 Pts", "ML Proj", "Score", "Diverge?"]
    k_rows = []
    for _, r in keeper.iterrows():
        pts = f"{r['recent_pts']:.0f}" if pd.notna(r['recent_pts']) else "-"
        ml = f"{r['ml_projection']:.0f}" if pd.notna(r['ml_projection']) else "-"
        score = f"{r['combined_score']:.2f}" if pd.notna(r['combined_score']) else "-"
        rec = r['recommendation']
        div = "YES" if r['signal_divergence'] else ""
        name = strip_accents(f"{r['first']} {r['last']}")
        k_rows.append([rec, name, r['position'], pts, ml, score, div])

    pdf.add_table(k_headers, k_rows, col_widths=[28, 38, 12, 22, 22, 18, 18], font_size=7.5)

    pdf.callout_box(
        "SIGNAL DIVERGENCE flags players where trajectory and ML projection disagree significantly. "
        "Max Fried shows divergence (high ML projection but declining trajectory)  - worth investigating. "
        "Steven Kwan also diverges (strong trajectory, weak ML projection)  - could be a buy-low keeper.",
        color=(230, 255, 230)
    )

    # Cuts to consider
    cuts = keeper[keeper['recommendation'].isin(['CUT', 'BORDERLINE'])]
    if len(cuts) > 0:
        pdf.subsection_title("Cuts to Consider")
        pdf.body_text(
            "These players scored BORDERLINE or CUT  - their keeper spot may be better used on a draft pick:"
        )
        cut_headers = ["Rec", "Player", "Pos", "2025 Pts", "ML Proj", "Score"]
        cut_rows = []
        for _, r in cuts.iterrows():
            name = strip_accents(f"{r['first']} {r['last']}")
            cut_rows.append([
                r['recommendation'], name, r['position'],
                _fmt(r['recent_pts']), _fmt(r['ml_projection']), _fmt(r['combined_score'], '.2f')
            ])
        pdf.add_table(cut_headers, cut_rows, col_widths=[28, 42, 15, 25, 25, 20], font_size=8)

    # Keeper spotlight
    pdf.subsection_title("Keeper Spotlight: Top Recommendations")
    pdf.body_text(
        "Drew Rasmussen (P)  - AUTO-KEEP (0.98): Despite limited 2024 innings, the ML model projects "
        "354 points based on his elite rate stats. High upside if healthy.\n\n"
        "Agustin Ramirez (C)  - AUTO-KEEP (0.93): No 2025 history yet (prospect), but the model projects "
        "345 points. Catcher scarcity makes this a premium keep.\n\n"
        "Max Fried (P)  - AUTO-KEEP (0.86): 521 points in 2025, model projects 360 for 2026. The decline "
        "projection flags caution but he's still a top pitcher.\n\n"
        "Eugenio Suarez (3B)  - KEEP (0.78): Steady 372 points with a 355-point projection. Divergence flag "
        "is a yellow light  - trajectory says improvement, ML says plateau."
    )

    # Group trajectory plots
    pdf.add_page()
    pdf.subsection_title("Keeper Group Trajectories")
    pdf.body_text(
        "These overlay plots show fantasy point trajectories for keeper candidates grouped by position. "
        "Upward trends suggest improving players worth keeping; declining trends signal potential cuts."
    )
    pdf.safe_image(FIG_DIR / "keeper_trajectories" / "group_INF.png", w=170)
    pdf.safe_image(FIG_DIR / "keeper_trajectories" / "group_OF.png", w=170)

    pdf.add_page()
    pdf.safe_image(FIG_DIR / "keeper_trajectories" / "group_pitchers.png", w=170)

    # Individual keeper plots - show a few key ones
    pdf.subsection_title("Individual Keeper Trajectories (Selected)")
    for name in ["Fried_Max", "Rasmussen_Drew", "Kwan_Steven", "Torres_Gleyber"]:
        p = FIG_DIR / "keeper_trajectories" / f"{name}.png"
        if p.exists():
            pdf.safe_image(p, w=130)

    # ===================== SECTION 4: DRAFT RANKINGS =====================
    pdf.add_page()
    pdf.section_title("4. 2026 Draft Rankings")

    pdf.body_text(
        "The model projects 2026 ESPN H2H points and calculates Points Above Replacement (PAR)  - the "
        "projected points minus a position-specific replacement level. PAR accounts for positional scarcity "
        "(catchers have lower replacement level than outfielders). ADP Rank and Rank Diff columns compare "
        "our ranking against the FanGraphs consensus ADP (average of Steamer, ZiPS, ZiPS-DC)."
    )

    # Top 30 batters with FG comparison
    pdf.subsection_title("Top 30 Batters by PAR")
    b_headers = ["#", "Player", "Pos", "Proj", "2025", "PAR", "ADP", "Diff", "Signal"]
    b_rows = []
    for _, r in bat_dash.head(30).iterrows():
        b_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            r['primary_pos'],
            _fmt(r['projected_pts_2026']),
            _fmt(r['pts_2025']),
            _fmt(r['PAR']),
            _fmt(r['adp_rank']),
            _fmt(r['rank_diff']),
            r['signal'] if r['signal'] else ""
        ])
    pdf.add_table(b_headers, b_rows,
                  col_widths=[10, 36, 12, 20, 20, 20, 18, 18, 22], font_size=7)

    # Top 30 pitchers with FG comparison
    pdf.add_page()
    pdf.subsection_title("Top 30 Pitchers by PAR")
    p_headers = ["#", "Player", "Proj", "2025", "PAR", "ADP", "Diff", "Signal"]
    p_rows = []
    for _, r in pit_dash.head(30).iterrows():
        p_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            _fmt(r['projected_pts_2026']),
            _fmt(r['pts_2025']),
            _fmt(r['PAR']),
            _fmt(r['adp_rank']),
            _fmt(r['rank_diff']),
            r['signal'] if r['signal'] else ""
        ])
    pdf.add_table(p_headers, p_rows,
                  col_widths=[10, 40, 24, 24, 24, 20, 20, 22], font_size=7)

    pdf.callout_box(
        "RISK FLAGS: 'ERA_regression' means the pitcher's ERA was well below xERA (lucky  - likely to regress). "
        "'BABIP_regression' means BABIP was well below career average. 'age_risk' flags players 33+. "
        "These flags do NOT mean avoid  - they signal where to dig deeper.",
        color=(255, 243, 224)
    )

    # ===================== SECTION 5: VS FANGRAPHS =====================
    pdf.add_page()
    pdf.section_title("5. Our Model vs. FanGraphs Projections")

    pdf.body_text(
        "How does our model compare to FanGraphs consensus projections (Steamer, ZiPS, ZiPS-DC)? "
        "The tables below show per-system rankings alongside our model rank and FG projected stats."
    )

    pdf.safe_image(FIG_DIR / "model_evaluation" / "draft_dashboard_comparison.png", w=180)

    pdf.body_text(
        "Scatter plots (top row) show our model rank vs. FanGraphs ADP. Points near the diagonal line "
        "indicate agreement. Points far from the line are where we differ from the market.\n\n"
        "Bar charts (bottom row) show the rank difference for top-30 players. Green bars = we rank them "
        "higher than ADP (value picks). Red bars = ADP ranks them higher than us (potential fades)."
    )

    # Batter FG system ranks table
    pdf.add_page()
    pdf.subsection_title("Batter Rankings: Our Model vs. FG Systems")
    fg_bat_headers = ["#", "Player", "Pos", "Proj", "Steamer", "ZiPS", "ZiPS-DC", "Article", "FG HR", "FG R", "FG RBI", "FG SB"]
    fg_bat_rows = []
    for _, r in bat_dash.head(30).iterrows():
        fg_bat_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            r['primary_pos'],
            _fmt(r['projected_pts_2026']),
            _fmt(r['stm_rank']),
            _fmt(r['zps_rank']),
            _fmt(r['zdc_rank']),
            _fmt(r['article_rank']),
            _fmt(r['fg_hr']),
            _fmt(r['fg_r']),
            _fmt(r['fg_rbi']),
            _fmt(r['fg_sb']),
        ])
    pdf.add_table(fg_bat_headers, fg_bat_rows,
                  col_widths=[8, 30, 10, 16, 16, 14, 16, 16, 14, 14, 14, 14], font_size=6.5)

    # Pitcher FG system ranks table
    pdf.add_page()
    pdf.subsection_title("Pitcher Rankings: Our Model vs. FG Systems")
    fg_pit_headers = ["#", "Player", "Proj", "Steamer", "ZiPS", "ZiPS-DC", "Article", "FG IP", "FG SO", "FG ERA", "FG WHIP"]
    fg_pit_rows = []
    for _, r in pit_dash.head(30).iterrows():
        fg_pit_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            _fmt(r['projected_pts_2026']),
            _fmt(r['stm_rank']),
            _fmt(r['zps_rank']),
            _fmt(r['zdc_rank']),
            _fmt(r['article_rank']),
            _fmt(r['fg_ip'], '.0f'),
            _fmt(r['fg_so'], '.0f'),
            _fmt(r['fg_era'], '.2f'),
            _fmt(r['fg_whip'], '.2f'),
        ])
    pdf.add_table(fg_pit_headers, fg_pit_rows,
                  col_widths=[8, 34, 18, 16, 14, 18, 16, 16, 16, 18, 18], font_size=6.5)

    # Value picks
    pdf.add_page()
    pdf.subsection_title("Value Picks  - Our Rank >> FG ADP")
    pdf.body_text(
        "Players we rank much higher than the FanGraphs consensus ADP (rank difference > +20). "
        "These are potential draft steals  - the market may be undervaluing them."
    )

    bat_value = bat_dash[bat_dash['signal'] == 'VALUE'].sort_values('overall_rank')
    if len(bat_value) > 0:
        pdf.bold_text(f"Batter Value Picks ({len(bat_value)} players)")
        val_headers = ["#", "Player", "Pos", "Proj Pts", "ADP", "Diff"]
        val_rows = []
        for _, r in bat_value.head(15).iterrows():
            val_rows.append([
                _fmt(r['overall_rank']), strip_accents(f"{r['first']} {r['last']}"),
                r['primary_pos'], _fmt(r['projected_pts_2026']),
                _fmt(r['adp_rank']), f"+{_fmt(r['rank_diff'])}"
            ])
        pdf.add_table(val_headers, val_rows,
                      col_widths=[12, 45, 15, 30, 25, 25], font_size=7.5)

    pit_value = pit_dash[pit_dash['signal'] == 'VALUE'].sort_values('overall_rank')
    if len(pit_value) > 0:
        pdf.bold_text(f"Pitcher Value Picks ({len(pit_value)} players)")
        val_rows = []
        for _, r in pit_value.head(15).iterrows():
            val_rows.append([
                _fmt(r['overall_rank']), strip_accents(f"{r['first']} {r['last']}"),
                _fmt(r['projected_pts_2026']),
                _fmt(r['adp_rank']), f"+{_fmt(r['rank_diff'])}"
            ])
        pdf.add_table(["#", "Player", "Proj Pts", "ADP", "Diff"], val_rows,
                      col_widths=[12, 55, 30, 25, 25], font_size=7.5)

    # Potential fades
    pdf.add_page()
    pdf.subsection_title("Potential Fades  - FG ADP >> Our Rank")
    pdf.body_text(
        "Players the market values much higher than our model (rank difference < -20). "
        "Consider whether the market knows something we don't, or if these are overdrafted."
    )

    bat_fade = bat_dash[bat_dash['signal'] == 'FADE'].sort_values('overall_rank')
    if len(bat_fade) > 0:
        pdf.bold_text(f"Batter Fades ({len(bat_fade)} players)")
        fade_headers = ["#", "Player", "Pos", "Proj Pts", "ADP", "Diff"]
        fade_rows = []
        for _, r in bat_fade.head(15).iterrows():
            fade_rows.append([
                _fmt(r['overall_rank']), strip_accents(f"{r['first']} {r['last']}"),
                r['primary_pos'], _fmt(r['projected_pts_2026']),
                _fmt(r['adp_rank']), _fmt(r['rank_diff'])
            ])
        pdf.add_table(fade_headers, fade_rows,
                      col_widths=[12, 45, 15, 30, 25, 25], font_size=7.5)

    pit_fade = pit_dash[pit_dash['signal'] == 'FADE'].sort_values('overall_rank')
    if len(pit_fade) > 0:
        pdf.bold_text(f"Pitcher Fades ({len(pit_fade)} players)")
        fade_rows = []
        for _, r in pit_fade.head(15).iterrows():
            fade_rows.append([
                _fmt(r['overall_rank']), strip_accents(f"{r['first']} {r['last']}"),
                _fmt(r['projected_pts_2026']),
                _fmt(r['adp_rank']), _fmt(r['rank_diff'])
            ])
        pdf.add_table(["#", "Player", "Proj Pts", "ADP", "Diff"], fade_rows,
                      col_widths=[12, 55, 30, 25, 25], font_size=7.5)

    pdf.subsection_title("How to Use This Comparison")
    pdf.body_text(
        "VALUE PICKS (our rank >> ADP): These players are projected to produce more than the market "
        "expects. Draft them before their ADP  - the market may be sleeping on them.\n\n"
        "POTENTIAL FADES (ADP >> our rank): The market loves these players more than our model does. "
        "This doesn't mean avoid them, but if choosing between two similar players, lean toward the one "
        "our model likes.\n\n"
        "IMPORTANT CAVEAT: Our model doesn't capture everything. It knows nothing about spring training "
        "performance, coaching changes, lineup position changes, or park moves. Use disagreements as "
        "starting points for research, not as final verdicts."
    )

    # ===================== SECTION 6: WHAT DRIVES PREDICTIONS =====================
    pdf.add_page()
    pdf.section_title("6. What Drives the Predictions?")

    pdf.body_text(
        "SHAP (SHapley Additive exPlanations) analysis reveals which features matter most. "
        "The Pareto charts show cumulative feature importance  - the top ~18-19 features explain 80% "
        "of the model's predictions."
    )

    pdf.subsection_title("Batter Feature Importance")
    pdf.safe_image(FIG_DIR / "feature_importance" / "batter_shap_summary.png", w=165)

    pdf.add_page()
    pdf.subsection_title("Pitcher SP Feature Importance")
    pdf.safe_image(FIG_DIR / "feature_importance" / "pitcher_sp_shap_summary.png", w=165)

    pdf.subsection_title("Pitcher RP Feature Importance")
    pdf.safe_image(FIG_DIR / "feature_importance" / "pitcher_rp_shap_summary.png", w=165)

    # ===================== SECTION 7: KNOWN LIMITATIONS =====================
    pdf.add_page()
    pdf.section_title("7. Known Limitations & Possible Errors")

    pdf.bold_text("1. Optuna Seed Variance (High)")
    pdf.body_text(
        "Despite multi-seed tuning, pitcher SP XGBoost Spearman ranged 0.727-0.992 across 5 seeds  - "
        "a 0.265 range. This means different random seeds produce very different models. Root cause: "
        "only 893 SP training rows. The RP model is more stable (0.036 range). We mitigate by taking "
        "the median seed, but the underlying variance is a data limitation."
    )

    pdf.bold_text("2. Batter Model Regression")
    pdf.body_text(
        "Dropping 2015-2016 from training reduced batter Spearman from 0.588 to 0.543. Those years "
        "had good batter data (unlike pitchers who lacked arsenal features). A future improvement would "
        "use separate training windows: 2015-2023 for batters, 2017-2023 for pitchers."
    )

    pdf.bold_text("3. Injury Blindness")
    pdf.body_text(
        "The model has no injury information. Pitchers like Blake Snell (projected 459, actual 164) "
        "and Tyler Glasnow are impossible to predict accurately without health data. This is the single "
        "largest source of error, especially for pitchers."
    )

    pdf.bold_text("4. Breakout Detection")
    pdf.body_text(
        "Players with no prior track record (Geraldo Perdomo, Bryan Woo) produced huge positive errors. "
        "The model anchors heavily on prior-year performance  - it cannot identify players about to make "
        "a quantum leap. This is a fundamental limitation of regression-based approaches."
    )

    pdf.bold_text("5. Holds = 0")
    pdf.body_text(
        "Retrosheet doesn't track holds. Reliever points are systematically underprojected for "
        "elite setup men who earn significant points from holds in ESPN scoring."
    )

    pdf.bold_text("6. Small Test Set")
    pdf.body_text(
        "With only 1 test year (2024 features -> 2025 actuals), results can be noisy. A single unusual "
        "season can make the model look better or worse than it really is. Multi-year backtesting would "
        "give more reliable accuracy estimates."
    )

    # ===================== DRAFT DAY CHEAT SHEET =====================
    pdf.add_page()
    pdf.section_title("Draft Day Cheat Sheet")

    pdf.body_text(
        "Quick-reference page for draft day. Top players by our model, plus the biggest disagreements "
        "with FanGraphs consensus ADP."
    )

    # Top 10 batters
    pdf.subsection_title("Top 10 Batters")
    cs_bat_headers = ["#", "Player", "Pos", "Proj Pts", "PAR"]
    cs_bat_rows = []
    for _, r in bat_dash.head(10).iterrows():
        cs_bat_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            r['primary_pos'],
            _fmt(r['projected_pts_2026']),
            _fmt(r['PAR']),
        ])
    pdf.add_table(cs_bat_headers, cs_bat_rows,
                  col_widths=[12, 55, 20, 35, 35], font_size=8)

    # Top 10 pitchers
    pdf.subsection_title("Top 10 Pitchers")
    cs_pit_headers = ["#", "Player", "Proj Pts", "PAR"]
    cs_pit_rows = []
    for _, r in pit_dash.head(10).iterrows():
        cs_pit_rows.append([
            _fmt(r['overall_rank']),
            strip_accents(f"{r['first']} {r['last']}"),
            _fmt(r['projected_pts_2026']),
            _fmt(r['PAR']),
        ])
    pdf.add_table(cs_pit_headers, cs_pit_rows,
                  col_widths=[12, 65, 40, 40], font_size=8)

    # Top 5 value picks (combined)
    pdf.subsection_title("Top 5 Value Picks (All Positions)")
    all_value = pd.concat([
        bat_value[['overall_rank', 'first', 'last', 'primary_pos', 'projected_pts_2026', 'adp_rank', 'rank_diff']].assign(type='BAT'),
        pit_value[['overall_rank', 'first', 'last', 'projected_pts_2026', 'adp_rank', 'rank_diff']].assign(type='PIT', primary_pos='P'),
    ]).sort_values('rank_diff', ascending=False).head(5)
    vp_rows = []
    for _, r in all_value.iterrows():
        vp_rows.append([
            strip_accents(f"{r['first']} {r['last']}"),
            r['primary_pos'],
            _fmt(r['overall_rank']),
            _fmt(r['adp_rank']),
            f"+{_fmt(r['rank_diff'])}"
        ])
    pdf.add_table(["Player", "Pos", "Our #", "ADP #", "Diff"], vp_rows,
                  col_widths=[55, 15, 25, 25, 25], font_size=8)

    # Top 5 fades (combined)
    pdf.subsection_title("Top 5 Fades (All Positions)")
    all_fade = pd.concat([
        bat_fade[['overall_rank', 'first', 'last', 'primary_pos', 'projected_pts_2026', 'adp_rank', 'rank_diff']].assign(type='BAT'),
        pit_fade[['overall_rank', 'first', 'last', 'projected_pts_2026', 'adp_rank', 'rank_diff']].assign(type='PIT', primary_pos='P'),
    ]).sort_values('rank_diff', ascending=True).head(5)
    fd_rows = []
    for _, r in all_fade.iterrows():
        fd_rows.append([
            strip_accents(f"{r['first']} {r['last']}"),
            r['primary_pos'],
            _fmt(r['overall_rank']),
            _fmt(r['adp_rank']),
            _fmt(r['rank_diff'])
        ])
    pdf.add_table(["Player", "Pos", "Our #", "ADP #", "Diff"], fd_rows,
                  col_widths=[55, 15, 25, 25, 25], font_size=8)

    pdf.callout_box(
        "Remember: VALUE picks are players the market is sleeping on  - draft them earlier than ADP. "
        "FADES are players the market may be overhyping  - let someone else overdraft them. "
        "Always cross-reference with injury news and spring training reports before draft day.",
        color=(255, 243, 224)
    )

    # ===================== APPENDIX =====================
    pdf.add_page()
    pdf.section_title("Appendix: Model Architecture Summary")

    arch_headers = ["Component", "Detail"]
    arch_rows = [
        ["Algorithm", "XGBoost + LightGBM ensemble (Ridge-blended)"],
        ["Training Data", "2017-2023 seasons (7 years)"],
        ["Validation", "2023 (last CV fold + blend fitting)"],
        ["Test Set", "2024 features -> 2025 actuals"],
        ["Prediction Target", "2025 features -> 2026 projected points"],
        ["Batter Features", "33 (Statcast, rate stats, deltas, RE24)"],
        ["Pitcher SP Features", "33 (arsenal, Statcast, rate stats)"],
        ["Pitcher RP Features", "35 (includes is_closer, p_gf)"],
        ["Optuna Trials", "100 per seed x 5 seeds x 2 model types"],
        ["CV Strategy", "3-fold expanding-window TSCV"],
        ["Loss Function", "Pseudo-Huber (robust to outliers)"],
        ["COVID Handling", "2020 rows downweighted 50%"],
        ["Pitcher IP Threshold", "50 IP (150 outs)"],
        ["Batter PA Threshold", "200 PA"],
    ]
    pdf.add_table(arch_headers, arch_rows, col_widths=[55, 135], font_size=8)

    pdf.subsection_title("Seed Stability Results")
    seed_headers = ["Model", "XGB Spearman Range", "LGB Spearman Range", "Selected By"]
    seed_rows = [
        ["Batter", "0.185 (0.703-0.888)", "0.122 (0.862-0.984)", "Median Spearman"],
        ["Pitcher SP", "0.265 (0.727-0.992)", "0.159 (0.825-0.984)", "Median Spearman"],
        ["Pitcher RP", "0.036 (0.964-1.000)", "0.068 (0.932-1.000)", "Median Spearman"],
    ]
    pdf.add_table(seed_headers, seed_rows, col_widths=[30, 55, 55, 40], font_size=8)

    pdf.body_text(
        "The Spearman range across seeds tells us how much randomness affects results. The SP model's "
        "0.265 XGB range is concerning  - it means a 'lucky' seed can make the model look great (0.992) "
        "while an 'unlucky' seed makes it look mediocre (0.727). The median selection strategy ensures "
        "we get a representative result, not an extreme one."
    )

    # Save
    pdf.output(str(OUT_PDF))
    print(f"Report saved to: {OUT_PDF}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
