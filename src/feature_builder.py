"""Phase 5: Feature engineering — merge Statcast, build ML features, year N→N+1 shift."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_builder import load_config, ROOT


def load_id_map(cfg):
    """Load Retrosheet→MLBAM ID mapping."""
    idm = pd.read_csv(ROOT / cfg["paths"]["raw"]["id_map"])
    idm["key_mlbam"] = pd.to_numeric(idm["key_mlbam"], errors="coerce")
    return idm[["key_retro", "key_mlbam"]].dropna(subset=["key_mlbam"])


def load_fg_id_map(cfg):
    """Load Retrosheet→FanGraphs ID mapping."""
    idm = pd.read_csv(ROOT / cfg["paths"]["raw"]["fg_id_map"])
    idm["key_fangraphs"] = pd.to_numeric(idm["key_fangraphs"], errors="coerce")
    idm = idm[idm["key_fangraphs"] > 0].copy()
    return idm[["key_retro", "key_fangraphs"]].dropna()


def merge_fangraphs_batters(df, cfg):
    """Overwrite wRC_plus, GB_pct, LD_pct, HR_FB_pct with FanGraphs authoritative values."""
    idm = load_fg_id_map(cfg)
    fg = pd.read_csv(ROOT / cfg["paths"]["raw"]["fangraphs_batters"])
    fg = fg[["IDfg", "Season", "wRC+", "GB%", "LD%", "HR/FB"]].copy()
    fg.rename(columns={
        "wRC+": "_fg_wRC_plus", "GB%": "_fg_GB_pct",
        "LD%": "_fg_LD_pct", "HR/FB": "_fg_HR_FB_pct"
    }, inplace=True)

    # Map Retrosheet id → key_fangraphs → IDfg
    df = df.merge(idm, left_on="id", right_on="key_retro", how="left")
    df = df.merge(fg, left_on=["key_fangraphs", "year"], right_on=["IDfg", "Season"], how="left")

    for col, fg_col in [("wRC_plus", "_fg_wRC_plus"), ("GB_pct", "_fg_GB_pct"),
                        ("LD_pct", "_fg_LD_pct"), ("HR_FB_pct", "_fg_HR_FB_pct")]:
        if fg_col in df.columns:
            if col in df.columns:
                df[col] = df[fg_col].combine_first(df[col])
            else:
                df[col] = df[fg_col]

    df.drop(columns=[c for c in ["key_retro", "key_fangraphs", "IDfg", "Season",
                                  "_fg_wRC_plus", "_fg_GB_pct", "_fg_LD_pct", "_fg_HR_FB_pct"]
                     if c in df.columns], inplace=True)
    return df


def merge_fangraphs_pitchers(df, cfg):
    """Overwrite ERA_minus, xFIP_minus, zone/csw/swstr/chase with FanGraphs authoritative values."""
    idm = load_fg_id_map(cfg)
    fg = pd.read_csv(ROOT / cfg["paths"]["raw"]["fangraphs_pitchers"])
    fg_raw_cols = ["IDfg", "Season", "ERA-", "xFIP-", "SwStr%", "CSW%", "Zone%", "O-Swing%", "SIERA"]
    fg = fg[[c for c in fg_raw_cols if c in fg.columns]].copy()
    fg.rename(columns={
        "ERA-": "_fg_ERA_minus", "xFIP-": "_fg_xFIP_minus",
        "SwStr%": "_fg_swstr_pct", "CSW%": "_fg_csw_pct",
        "Zone%": "_fg_zone_pct", "O-Swing%": "_fg_chase_pct",
        "SIERA": "_fg_SIERA"
    }, inplace=True)

    df = df.merge(idm, left_on="id", right_on="key_retro", how="left")
    df = df.merge(fg, left_on=["key_fangraphs", "year"], right_on=["IDfg", "Season"], how="left")

    for col, fg_col in [("ERA_minus", "_fg_ERA_minus"), ("xFIP_minus", "_fg_xFIP_minus"),
                        ("swstr_pct", "_fg_swstr_pct"), ("csw_pct", "_fg_csw_pct"),
                        ("zone_pct", "_fg_zone_pct"), ("chase_pct", "_fg_chase_pct"),
                        ("SIERA", "_fg_SIERA")]:
        if fg_col in df.columns:
            if col in df.columns:
                df[col] = df[fg_col].combine_first(df[col])
            else:
                df[col] = df[fg_col]

    df.drop(columns=[c for c in ["key_retro", "key_fangraphs", "IDfg", "Season",
                                  "_fg_ERA_minus", "_fg_xFIP_minus", "_fg_swstr_pct",
                                  "_fg_csw_pct", "_fg_zone_pct", "_fg_chase_pct", "_fg_SIERA"]
                     if c in df.columns], inplace=True)
    return df


def merge_statcast_batters(batters, cfg):
    """Merge Statcast batter metrics via ID map."""
    idm = load_id_map(cfg)
    sc = pd.read_csv(ROOT / cfg["paths"]["raw"]["statcast_batters"])

    sc_cols = ["player_id", "year", "brl_percent", "ev95percent", "avg_hit_speed",
               "anglesweetspotpercent", "est_ba", "est_slg", "est_woba", "sprint_speed",
               "est_ba_minus_ba_diff", "est_slg_minus_slg_diff"]
    sc = sc[[c for c in sc_cols if c in sc.columns]].copy()

    # Map MLBAM IDs to Retrosheet
    sc = sc.merge(idm, left_on="player_id", right_on="key_mlbam", how="inner")
    sc.rename(columns={"key_retro": "id"}, inplace=True)
    sc.drop(columns=["player_id", "key_mlbam"], inplace=True)

    merged = batters.merge(sc, on=["id", "year"], how="left")
    return merged


def merge_statcast_pitchers(pitchers, cfg):
    """Merge Statcast pitcher metrics + weighted arsenal whiff% via ID map."""
    idm = load_id_map(cfg)

    # Pitcher expected stats
    sc = pd.read_csv(ROOT / cfg["paths"]["raw"]["statcast_pitchers"])
    sc_cols = ["player_id", "year", "brl_percent", "ev95percent", "avg_hit_speed",
               "est_ba", "est_slg", "est_woba", "era", "xera", "era_minus_xera_diff",
               "swstr_pct", "csw_pct", "zone_pct", "chase_pct",
               "woba", "anglesweetspotpercent", "ev50", "gb", "pa"]
    sc = sc[[c for c in sc_cols if c in sc.columns]].copy()
    sc.rename(columns={"brl_percent": "brl_percent_allowed",
                       "ev95percent": "ev95percent_against",
                       "avg_hit_speed": "avg_ev_against",
                       "woba": "wOBA_against",
                       "anglesweetspotpercent": "sweet_spot_pct",
                       "gb": "gb_pct",
                       "pa": "BF"}, inplace=True)
    sc = sc.merge(idm, left_on="player_id", right_on="key_mlbam", how="inner")
    sc.rename(columns={"key_retro": "id"}, inplace=True)
    sc.drop(columns=["player_id", "key_mlbam"], inplace=True)

    # Arsenal: weighted whiff% as SwStr% proxy
    ars = pd.read_csv(ROOT / cfg["paths"]["raw"]["statcast_arsenal"])
    ars = ars[["player_id", "year", "whiff_percent", "pitch_usage", "hard_hit_percent", "k_percent"]].dropna(subset=["whiff_percent"])
    # Weighted whiff% per pitcher-year
    ars["weighted_whiff"] = ars["whiff_percent"] * ars["pitch_usage"] / 100
    ars_agg = ars.groupby(["player_id", "year"]).agg(
        weighted_whiff_pct=("weighted_whiff", "sum"),
        avg_hard_hit_pct=("hard_hit_percent", "mean"),
        K_pct_savant=("k_percent", "mean")
    ).reset_index()
    ars_agg = ars_agg.merge(idm, left_on="player_id", right_on="key_mlbam", how="inner")
    ars_agg.rename(columns={"key_retro": "id"}, inplace=True)
    ars_agg.drop(columns=["player_id", "key_mlbam"], inplace=True)

    # Pitch velocities (fastball only — other pitch types have >25% nulls)
    vel = pd.read_csv(ROOT / cfg["paths"]["raw"]["statcast_pitch_arsenal"])
    vel = vel[["pitcher", "year", "ff_avg_speed"]].copy()
    vel = vel.merge(idm, left_on="pitcher", right_on="key_mlbam", how="inner")
    vel.rename(columns={"key_retro": "id"}, inplace=True)
    vel.drop(columns=["pitcher", "key_mlbam"], inplace=True)

    # Merge all into pitchers
    merged = pitchers.merge(sc, on=["id", "year"], how="left")
    merged = merged.merge(ars_agg, on=["id", "year"], how="left")
    merged = merged.merge(vel, on=["id", "year"], how="left")
    return merged


def add_career_rolling(df, group_col="id", cols=None, min_periods=2):
    """Add expanding career averages (no lookahead) for regression signals."""
    if cols is None:
        cols = ["BABIP"]
    df = df.sort_values([group_col, "year"])
    for col in cols:
        if col in df.columns:
            df[f"career_{col}"] = df.groupby(group_col)[col].transform(
                lambda x: x.expanding(min_periods=min_periods).mean().shift(1)
            )
    return df


def build_batter_features(cfg):
    """Build full batter feature set: stats + Statcast + RE24 + batted ball + career."""
    batters = pd.read_csv(ROOT / cfg["paths"]["processed"]["batters_season"])
    qual = batters[~batters["LOW_SAMPLE"]].copy()

    # Merge Statcast
    qual = merge_statcast_batters(qual, cfg)

    # Merge RE24
    re24_path = ROOT / cfg["paths"]["processed"]["batter_re24"]
    if re24_path.exists():
        re24 = pd.read_csv(re24_path)
        qual = qual.merge(re24[["id", "year", "RE24", "RE24_efficiency"]], on=["id", "year"], how="left")

    # Calculate wOBA, wRAA, wRC+ from season constants
    sc_path = ROOT / cfg["paths"]["processed"]["season_constants"]
    if sc_path.exists():
        from src.re24_engine import calculate_woba_wrc
        from src.advanced_stats import calculate_league_averages
        sc = pd.read_csv(sc_path)
        lg = calculate_league_averages(cfg)
        qual = calculate_woba_wrc(qual, sc, lg)

    # Merge batted ball rates
    bb_path = ROOT / cfg["paths"]["processed"]["batted_ball"].replace(".csv", "_batters.csv")
    if (ROOT / bb_path).exists():
        bb = pd.read_csv(ROOT / bb_path)
        qual = qual.merge(bb[["id", "year", "GB_pct", "FB_pct", "LD_pct", "HR_FB_pct"]], on=["id", "year"], how="left")

    # Overwrite wRC_plus, GB_pct, LD_pct, HR_FB_pct with FanGraphs authoritative values
    qual = merge_fangraphs_batters(qual, cfg)

    # Career rolling stats for regression signals
    qual = add_career_rolling(qual, cols=["BABIP", "HR_FB_pct"])

    # Regression gap features
    if "career_BABIP" in qual.columns:
        qual["BABIP_gap"] = qual["BABIP"] - qual["career_BABIP"]
    if "career_HR_FB_pct" in qual.columns:
        qual["HR_FB_gap"] = qual["HR_FB_pct"] - qual["career_HR_FB_pct"]
    if "est_ba" in qual.columns and "AVG" in qual.columns:
        qual["xBA_AVG_gap"] = qual["est_ba"] - qual["AVG"]
    if "est_slg" in qual.columns and "SLG" in qual.columns:
        qual["xSLG_SLG_gap"] = qual["est_slg"] - qual["SLG"]

    # Team context: runs per game (from gameinfo)
    gi = pd.read_csv(ROOT / cfg["paths"]["raw"]["gameinfo"], low_memory=False)
    gi = gi[gi["gametype"].str.lower() == "regular"]
    gi["year"] = gi["season"]
    team_rpg = gi.melt(id_vars=["gid", "year"], value_vars=["hometeam", "visteam"],
                       var_name="role", value_name="team")
    runs = gi.melt(id_vars=["gid", "year"], value_vars=["hruns", "vruns"],
                   var_name="run_type", value_name="runs_scored")
    # Align home/vis
    team_rpg["runs_scored"] = runs["runs_scored"].values
    team_rpg = team_rpg.groupby(["year", "team"]).agg(
        total_runs=("runs_scored", "sum"), games=("gid", "count")
    ).reset_index()
    team_rpg["team_rpg"] = team_rpg["total_runs"] / team_rpg["games"]
    qual = qual.merge(team_rpg[["year", "team", "team_rpg"]], on=["year", "team"], how="left")

    # Experience: count of prior seasons
    qual = qual.sort_values(["id", "year"])
    qual["experience"] = qual.groupby("id").cumcount()

    # SB rate per PA
    qual["SB_rate"] = qual["b_sb"] / qual["b_pa"].replace(0, np.nan)

    # Year-over-year deltas for momentum signals
    qual = qual.sort_values(["id", "year"])
    for col in ["K_pct", "BB_pct", "ISO", "BABIP"]:
        if col in qual.columns:
            qual[f"{col}_delta"] = qual.groupby("id")[col].diff()

    # Build feature column list from config
    feat_cfg = cfg.get("features", {})
    meta_cols = feat_cfg.get("meta", ["id", "year", "last", "first", "team", "primary_pos", "G"])
    counting_cols = feat_cfg.get("batter", {}).get("counting", [])
    model_cols = feat_cfg.get("batter", {}).get("model", [])
    feature_cols = meta_cols + counting_cols + model_cols + ["ESPN_Pts"]
    feature_cols = [c for c in feature_cols if c in qual.columns]
    qual = qual[feature_cols]

    qual.to_csv(ROOT / cfg["paths"]["processed"]["batter_features"], index=False)
    return qual


def build_pitcher_features(cfg):
    """Build full pitcher feature set."""
    pitchers = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"])
    qual = pitchers[~pitchers["LOW_SAMPLE"]].copy()

    # Merge Statcast
    qual = merge_statcast_pitchers(qual, cfg)

    # Merge RE24
    re24_path = ROOT / cfg["paths"]["processed"]["pitcher_re24"]
    if re24_path.exists():
        re24 = pd.read_csv(re24_path)
        qual = qual.merge(re24[["id", "year", "RE24"]], on=["id", "year"], how="left")

    # Merge batted ball — only GB_pct needed (xFIP now from FanGraphs)
    bb_path = ROOT / cfg["paths"]["processed"]["batted_ball"].replace(".csv", "_pitchers.csv")
    if (ROOT / bb_path).exists():
        bb = pd.read_csv(ROOT / bb_path)
        qual = qual.merge(bb[["id", "year", "GB_pct"]], on=["id", "year"], how="left")

    # Overwrite ERA_minus, xFIP_minus, zone/csw/swstr/chase with FanGraphs authoritative values
    qual = merge_fangraphs_pitchers(qual, cfg)

    # Career rolling
    qual = add_career_rolling(qual, cols=["BABIP_allowed", "ERA"])

    # Regression gaps
    if "career_BABIP_allowed" in qual.columns:
        qual["BABIP_gap"] = qual["BABIP_allowed"] - qual["career_BABIP_allowed"]
    if "FIP" in qual.columns and "ERA" in qual.columns:
        qual["ERA_FIP_gap"] = qual["ERA"] - qual["FIP"]

    # LOB% deviation from league average (~.720)
    qual["LOB_pct_dev"] = qual["LOB_pct"] - 0.720

    # Team win pct
    gi = pd.read_csv(ROOT / cfg["paths"]["raw"]["gameinfo"], low_memory=False)
    gi = gi[gi["gametype"].str.lower() == "regular"]
    gi["year"] = gi["season"]
    team_w = gi.groupby(["year", "wteam"]).size().reset_index(name="wins")
    team_g = pd.concat([
        gi.groupby(["year", "hometeam"]).size().reset_index(name="g").rename(columns={"hometeam": "team"}),
        gi.groupby(["year", "visteam"]).size().reset_index(name="g").rename(columns={"visteam": "team"})
    ]).groupby(["year", "team"])["g"].sum().reset_index()
    team_w.rename(columns={"wteam": "team"}, inplace=True)
    team_wp = team_g.merge(team_w, on=["year", "team"], how="left")
    team_wp["wins"] = team_wp["wins"].fillna(0)
    team_wp["team_win_pct"] = team_wp["wins"] / team_wp["g"]
    qual = qual.merge(team_wp[["year", "team", "team_win_pct"]], on=["year", "team"], how="left")

    # Closer flag
    qual["is_closer"] = ((qual["is_starter"] == 0) & (qual["SV"] >= 10)).astype(int)

    # Experience
    qual = qual.sort_values(["id", "year"])
    qual["experience"] = qual.groupby("id").cumcount()

    # Year-over-year deltas for momentum signals
    for col in ["K9", "ERA", "WHIP", "BB9"]:
        if col in qual.columns:
            qual[f"{col}_delta"] = qual.groupby("id")[col].diff()
    if "ff_avg_speed" in qual.columns:
        qual["ff_velo_delta"] = qual.groupby("id")["ff_avg_speed"].diff()

    # Build feature column list from config
    feat_cfg = cfg.get("features", {})
    meta_cols = feat_cfg.get("meta", ["id", "year", "last", "first", "team", "primary_pos", "G"])
    counting_cols = feat_cfg.get("pitcher", {}).get("counting", [])
    model_cols = feat_cfg.get("pitcher", {}).get("model", [])
    feature_cols = meta_cols + counting_cols + model_cols + ["ESPN_Pts"]
    feature_cols = [c for c in feature_cols if c in qual.columns]
    qual = qual[feature_cols]

    qual.to_csv(ROOT / cfg["paths"]["processed"]["pitcher_features"], index=False)
    return qual


def build_shifted_dataset(features_df, id_col="id", year_col="year", target_col="ESPN_Pts", cfg=None):
    """Create year N features → year N+1 target dataset (no future leakage)."""
    df = features_df.copy()

    # Create next-year target
    target_next = df[[id_col, year_col, target_col]].copy()
    target_next[year_col] = target_next[year_col] - 1
    target_next.rename(columns={target_col: "target_ESPN_Pts"}, inplace=True)

    # Merge: features from year N, target from year N+1
    shifted = df.merge(target_next, on=[id_col, year_col], how="inner")
    # Drop current year ESPN_Pts (would be leakage)
    shifted.drop(columns=[target_col], inplace=True)

    # Pro-rate 2020 targets to 162-game pace
    # target year = feature year + 1, so feature year 2019 → target year 2020
    covid_cfg = cfg.get("covid", {}) if cfg else {}
    if covid_cfg:
        covid_year = covid_cfg["year"]
        scale = covid_cfg["full_season_games"] / covid_cfg["actual_games"]
        covid_mask = shifted[year_col] == (covid_year - 1)  # feature year
        shifted.loc[covid_mask, "target_ESPN_Pts"] *= scale
        # Cap pro-rated targets at historical (non-COVID) max/min
        non_covid_mask = shifted[year_col] != (covid_year - 1)
        hist_max = shifted.loc[non_covid_mask, "target_ESPN_Pts"].max()
        hist_min = shifted.loc[non_covid_mask, "target_ESPN_Pts"].min()
        shifted["target_ESPN_Pts"] = shifted["target_ESPN_Pts"].clip(hist_min, hist_max)
        # Flag rows where feature year IS 2020 (noisy features)
        shifted["covid_feature_year"] = (shifted[year_col] == covid_year).astype(int)

    return shifted


def generate_scatter_plots(batter_features, pitcher_features, cfg):
    """Generate 20+ scatter/distribution plots."""
    fig_dir = ROOT / cfg["paths"]["figures"]["scatter_trends"]

    qual_b = batter_features[~batter_features["ESPN_Pts"].isna()]
    qual_p = pitcher_features[~pitcher_features["ESPN_Pts"].isna()]

    batter_plots = [
        ("OPS", "ESPN_Pts", "OPS vs ESPN Points"),
        ("wOBA", "ESPN_Pts", "wOBA vs ESPN Points"),
        ("ISO", "ESPN_Pts", "ISO vs ESPN Points"),
        ("BB_pct", "K_pct", "BB% vs K%"),
        ("brl_percent", "ESPN_Pts", "Barrel% vs ESPN Points"),
        ("avg_hit_speed", "ESPN_Pts", "Avg Exit Velo vs ESPN Points"),
        ("est_woba", "ESPN_Pts", "xwOBA vs ESPN Points"),
        ("sprint_speed", "SB_rate", "Sprint Speed vs SB Rate"),
        ("BABIP", "AVG", "BABIP vs AVG"),
        ("HR_PA", "ISO", "HR/PA vs ISO"),
    ]

    pitcher_plots = [
        ("K_BB_pct", "ESPN_Pts", "K-BB% vs ESPN Points"),
        ("ERA", "ESPN_Pts", "ERA vs ESPN Points"),
        ("FIP", "ESPN_Pts", "FIP vs ESPN Points"),
        ("xera", "ESPN_Pts", "xERA vs ESPN Points"),
        ("weighted_whiff_pct", "K_pct", "Weighted Whiff% vs K%"),
        ("WHIP", "ESPN_Pts", "WHIP vs ESPN Points"),
        ("GB_pct", "ERA", "GB% vs ERA"),
        ("BABIP_allowed", "ERA", "BABIP vs ERA"),
        ("LOB_pct", "ERA", "LOB% vs ERA"),
        ("BB_pct", "ESPN_Pts", "BB% vs ESPN Points (Pitchers)"),
    ]

    plot_count = 0
    for x, y, title in batter_plots:
        if x not in qual_b.columns or y not in qual_b.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        data = qual_b[[x, y]].dropna()
        ax.scatter(data[x], data[y], alpha=0.3, s=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        fig.savefig(fig_dir / f"bat_{x}_vs_{y}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        plot_count += 1

    for x, y, title in pitcher_plots:
        if x not in qual_p.columns or y not in qual_p.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        data = qual_p[[x, y]].dropna()
        ax.scatter(data[x], data[y], alpha=0.3, s=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        fig.savefig(fig_dir / f"pit_{x}_vs_{y}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        plot_count += 1

    # Distribution plots
    for col, label, data in [
        ("ESPN_Pts", "Batter ESPN Points", qual_b),
        ("ESPN_Pts", "Pitcher ESPN Points", qual_p),
        ("OPS", "OPS Distribution", qual_b),
        ("ERA", "ERA Distribution", qual_p[qual_p["is_starter"] == 1]),
    ]:
        if col not in data.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        data[col].dropna().hist(bins=50, ax=ax, alpha=0.7)
        ax.set_xlabel(col)
        ax.set_title(label)
        fig.savefig(fig_dir / f"dist_{label.replace(' ', '_').lower()}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        plot_count += 1

    return plot_count


def generate_regression_scatter_plots(batter_features, pitcher_features, cfg):
    """Scatter plot of every numeric metric vs ESPN_Pts with linear regression line + R²."""
    fig_dir = ROOT / cfg["paths"]["figures"]["scatter_trends"]

    # Columns to skip (meta/id/counting — not meaningful metrics to plot vs ESPN_Pts)
    meta = {"id", "year", "last", "first", "team", "primary_pos", "ESPN_Pts", "G",
            "b_pa", "b_hr", "b_r", "b_rbi", "b_sb", "b_k", "b_w",
            "p_k", "p_h", "p_er", "p_w", "p_hr", "p_hbp", "p_bfp", "p_ipouts",
            "W", "L", "SV", "HD", "p_gs", "p_gf",
            "is_starter", "is_closer", "PF"}

    def _plot_one(data, col, prefix, fig_dir):
        subset = data[[col, "ESPN_Pts"]].dropna()
        if len(subset) < 20:
            return False
        x, y = subset[col].values, subset["ESPN_Pts"].values
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        fit_line = np.poly1d(coeffs)
        corr = np.corrcoef(x, y)[0, 1]
        r2 = corr ** 2

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.3, s=10)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, fit_line(x_sorted), color="red", linewidth=2)
        ax.set_xlabel(col)
        ax.set_ylabel("ESPN_Pts")
        ax.set_title(f"{col} vs ESPN_Pts")
        ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=12, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        fig.savefig(fig_dir / f"{prefix}regr_{col}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True

    plot_count = 0
    for label, df, prefix in [("batter", batter_features, "bat_"),
                               ("pitcher", pitcher_features, "pit_")]:
        qual = df[~df["ESPN_Pts"].isna()].copy()
        numeric_cols = [c for c in qual.select_dtypes(include=[np.number]).columns
                        if c not in meta]
        for col in numeric_cols:
            if _plot_one(qual, col, prefix, fig_dir):
                plot_count += 1

    print(f"  {plot_count} regression scatter plots saved to {fig_dir}")
    return plot_count


def generate_pair_plots(batter_features, pitcher_features, cfg):
    """Pair plots of top SHAP features + ESPN_Pts for each player type."""
    fig_dir = ROOT / cfg["paths"]["figures"]["scatter_trends"]

    bat_cols = ["b_pa", "K_pct", "avg_hit_speed", "SB_rate", "est_ba",
                "RE24", "sprint_speed", "BB_K_pct", "ESPN_Pts"]
    pit_cols = ["p_gs", "K_BB_pct", "p_ipouts", "K9", "K_pct",
                "p_bfp", "is_starter", "IP", "ESPN_Pts"]

    for label, df, cols, fname in [
        ("batter", batter_features, bat_cols, "bat_pairplot.png"),
        ("pitcher", pitcher_features, pit_cols, "pit_pairplot.png"),
    ]:
        available = [c for c in cols if c in df.columns]
        if "ESPN_Pts" not in available:
            continue
        data = df[available].dropna()
        if len(data) < 20:
            continue
        g = sns.pairplot(data, diag_kind="kde",
                         plot_kws={"alpha": 0.3, "s": 5},
                         height=2.2)
        g.fig.set_size_inches(20, 20)
        g.savefig(fig_dir / fname, dpi=100, bbox_inches="tight")
        plt.close("all")
        print(f"  {label} pair plot saved: {fname}")

    return 2


def generate_feature_grid_plot(batter_features, pitcher_features, cfg):
    """Single consolidated grid image: every model feature vs ESPN_Pts, one subplot per feature."""
    fig_dir = ROOT / cfg["paths"]["figures"]["scatter_trends"]

    # Columns to exclude (meta, IDs, counting stats, target)
    meta = {"id", "year", "last", "first", "team", "primary_pos", "ESPN_Pts", "G",
            "b_pa", "b_hr", "b_r", "b_rbi", "b_sb", "b_k", "b_w",
            "p_k", "p_h", "p_er", "p_w", "p_hr", "p_hbp", "p_bfp", "p_ipouts",
            "W", "L", "SV", "HD", "p_gs", "p_gf",
            "is_starter", "is_closer", "PF"}

    for label, df, fname in [
        ("Batter", batter_features, "batter_feature_grid.png"),
        ("Pitcher", pitcher_features, "pitcher_feature_grid.png"),
    ]:
        qual = df[~df["ESPN_Pts"].isna()].copy()
        numeric_cols = [c for c in qual.select_dtypes(include=[np.number]).columns
                        if c not in meta]

        # Compute R² for each feature and sort descending
        r2_map = {}
        for col in numeric_cols:
            subset = qual[[col, "ESPN_Pts"]].dropna()
            if len(subset) < 20:
                continue
            corr = np.corrcoef(subset[col].values, subset["ESPN_Pts"].values)[0, 1]
            r2_map[col] = corr ** 2

        sorted_cols = sorted(r2_map.keys(), key=lambda c: r2_map[c], reverse=True)
        n = len(sorted_cols)
        if n == 0:
            continue

        ncols = 6
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = axes.flatten()
        fig.suptitle(f"{label} Features vs ESPN Fantasy Points (sorted by R²)",
                     fontsize=18, fontweight="bold", y=0.995)

        for i, col in enumerate(sorted_cols):
            ax = axes[i]
            subset = qual[[col, "ESPN_Pts"]].dropna()
            x, y = subset[col].values, subset["ESPN_Pts"].values

            ax.scatter(x, y, alpha=0.15, s=5, color="#4C72B0")

            # Regression line
            coeffs = np.polyfit(x, y, 1)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, np.poly1d(coeffs)(x_sorted), color="red", linewidth=1.5)

            r2 = r2_map[col]
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.annotate(f"R²={r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.2",
                                              facecolor="wheat", alpha=0.7))
            ax.tick_params(labelsize=6)
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Hide empty subplots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {label} feature grid saved: {fname} ({n} subplots, {nrows}x{ncols})")


def generate_re24_scatter(cfg):
    """Scatterplot of RE24 vs Available Runs for qualified batters, with LOWESS + labels."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig_dir = ROOT / cfg["paths"]["figures"]["scatter_trends"]
    re24 = pd.read_csv(ROOT / cfg["paths"]["processed"]["batter_re24"])
    names = pd.read_csv(ROOT / cfg["paths"]["processed"]["batter_features"],
                        usecols=["id", "year", "last"])
    re24 = re24.merge(names, on=["id", "year"], how="left")

    for year in [2024, 2025]:
        qual = re24[(re24["year"] == year) & (re24["PA"] >= 400)].copy()
        if len(qual) < 10:
            print(f"  {year}: only {len(qual)} qualified batters, skipping")
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(qual["Available_Runs"], qual["RE24"],
                   alpha=0.6, edgecolors="black", linewidth=0.5)

        # Zero line
        ax.axhline(y=0, color="red", linestyle="-", linewidth=1)

        # LOWESS smoother
        smooth = lowess(qual["RE24"], qual["Available_Runs"], frac=0.6)
        ax.plot(smooth[:, 0], smooth[:, 1], color="blue", linewidth=2)

        # Label elite hitters
        elite = qual[qual["RE24"] >= 40]
        for _, row in elite.iterrows():
            ax.annotate(row["last"],
                        xy=(row["Available_Runs"], row["RE24"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, alpha=0.8)

        ax.set_xlabel("Runs Potential (Available Runs)")
        ax.set_ylabel("RE24 (Total Run Value)")
        ax.set_title(f"RE24 vs Run Opportunity — Qualified Batters ({year})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"re24_vs_available_{year}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  RE24 scatter saved: re24_vs_available_{year}.png "
              f"({len(qual)} batters, {len(elite)} labeled)")
