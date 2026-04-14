"""Phase 3: RE24 engine — run expectancy matrix, wOBA weights, WAR components."""

import pandas as pd
import numpy as np
from src.data_builder import load_config, ROOT


PLAYS_KEEP_COLS = [
    "gid", "inning", "top_bot", "batter", "pitcher", "outs_pre", "outs_post",
    "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post",
    "runs", "pa", "single", "double", "triple", "hr", "walk", "iw", "hbp",
    "k", "sf", "sh", "ground", "fly", "line", "gdp",
    "sb2", "sb3", "sbh", "cs2", "cs3", "csh",
    "run_b", "run1", "run2", "run3",
    "date", "gametype"
]


def load_plays_filtered(cfg, chunksize=1_000_000):
    """Read plays.csv in chunks, filter to 2015-2025 regular season, keep needed columns."""
    path = ROOT / cfg["paths"]["raw"]["plays"]
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                             usecols=PLAYS_KEEP_COLS):
        chunk["year"] = chunk["date"].astype(str).str[:4].astype(int)
        chunk = chunk[(chunk["gametype"] == "regular") & chunk["year"].between(2015, 2025)]
        chunks.append(chunk)
        print(f"  Loaded chunk, kept {len(chunk):,} rows")
    plays = pd.concat(chunks, ignore_index=True)
    print(f"Total plays: {len(plays):,}")
    return plays


def encode_base_state(df):
    """Encode base states as 0-7 integer."""
    df["base_state_pre"] = (
        (df["br1_pre"].fillna("") != "").astype(int) * 1 +
        (df["br2_pre"].fillna("") != "").astype(int) * 2 +
        (df["br3_pre"].fillna("") != "").astype(int) * 4
    )
    df["base_state_post"] = (
        (df["br1_post"].fillna("") != "").astype(int) * 1 +
        (df["br2_post"].fillna("") != "").astype(int) * 2 +
        (df["br3_post"].fillna("") != "").astype(int) * 4
    )
    return df


def build_re_matrix(plays):
    """Build 24-cell run expectancy matrix per season (8 base states x 3 outs)."""
    plays = encode_base_state(plays)

    # Calculate runs scored from each state to end of half-inning
    # For each PA, we need: runs scored from this point to end of inning
    # Approach: for each half-inning, calculate cumulative runs from each PA forward

    # Create half-inning identifier
    plays["half_inning"] = plays["gid"] + "_" + plays["inning"].astype(str) + "_" + plays["top_bot"].astype(str)

    # Cumulative runs to end of each half-inning
    # Total runs in half-inning minus cumulative runs before this PA
    hi_runs = plays.groupby("half_inning")["runs"].transform("sum")
    cum_runs_before = plays.groupby("half_inning")["runs"].cumsum() - plays["runs"]
    plays["runs_to_end"] = hi_runs - cum_runs_before

    # RE = average runs_to_end for each (year, base_state_pre, outs_pre)
    re = plays.groupby(["year", "base_state_pre", "outs_pre"]).agg(
        avg_re=("runs_to_end", "mean"),
        count=("runs_to_end", "count")
    ).reset_index()
    re.rename(columns={"base_state_pre": "base_state", "outs_pre": "outs"}, inplace=True)
    return re


def calculate_re24(plays, re_matrix):
    """Calculate RE24 per plate appearance."""
    plays = encode_base_state(plays)

    # Merge RE for start state
    re_start = re_matrix.rename(columns={"base_state": "base_state_pre", "outs": "outs_pre",
                                          "avg_re": "RE_start"})
    plays = plays.merge(re_start[["year", "base_state_pre", "outs_pre", "RE_start"]],
                        on=["year", "base_state_pre", "outs_pre"], how="left")

    # Merge RE for end state (0 if inning ended = 3 outs)
    re_end = re_matrix.rename(columns={"base_state": "base_state_post", "outs": "outs_post",
                                        "avg_re": "RE_end"})
    plays = plays.merge(re_end[["year", "base_state_post", "outs_post", "RE_end"]],
                        on=["year", "base_state_post", "outs_post"], how="left")

    # If 3 outs, RE_end = 0
    plays["RE_end"] = plays["RE_end"].fillna(0)

    # RE24 = runs scored + RE_end - RE_start
    plays["RE24"] = plays["runs"] + plays["RE_end"] - plays["RE_start"]

    return plays


def aggregate_re24_players(plays):
    """Sum RE24 per batter-season and pitcher-season."""
    # Only count plate appearances
    pa_plays = plays[plays["pa"] == 1]

    batter_re24 = pa_plays.groupby(["batter", "year"]).agg(
        RE24=("RE24", "sum"),
        PA=("RE24", "count"),
        Available_Runs=("RE_start", "sum")
    ).reset_index()
    batter_re24.rename(columns={"batter": "id"}, inplace=True)
    batter_re24["RE24_efficiency"] = batter_re24["RE24"] / batter_re24["Available_Runs"].replace(0, np.nan)

    pitcher_re24 = pa_plays.groupby(["pitcher", "year"]).agg(
        RE24_against=("RE24", "sum"),
        BF=("RE24", "count")
    ).reset_index()
    pitcher_re24.rename(columns={"pitcher": "id"}, inplace=True)
    # Negate for pitchers (lower is better)
    pitcher_re24["RE24"] = -pitcher_re24["RE24_against"]

    return batter_re24, pitcher_re24


def derive_woba_weights(plays, re_matrix):
    """Derive wOBA weights from run values of each event type."""
    pa_plays = plays[plays["pa"] == 1].copy()

    # Event types and their boolean columns
    events = {
        "out": None,  # derived: PA that is not any positive event
        "BB": "walk",
        "HBP": "hbp",
        "1B": "single",
        "2B": "double",
        "3B": "triple",
        "HR": "hr"
    }

    # Identify outs (PA but no positive outcome)
    positive = pa_plays[["walk", "hbp", "single", "double", "triple", "hr"]].sum(axis=1) > 0
    pa_plays["is_out"] = (~positive).astype(int)

    # Average RE24 (run value) per event type per year
    weights_list = []
    for year in sorted(pa_plays["year"].unique()):
        yr = pa_plays[pa_plays["year"] == year]
        row = {"year": year}

        # Run value of an out
        outs = yr[yr["is_out"] == 1]
        rv_out = outs["RE24"].mean() if len(outs) > 0 else 0

        for event, col in events.items():
            if event == "out":
                row["rv_out"] = rv_out
                continue
            subset = yr[yr[col] == 1]
            if len(subset) > 0:
                row[f"rv_{event}"] = subset["RE24"].mean()
            else:
                row[f"rv_{event}"] = 0

        # wOBA weights = run_value - run_value_of_out, scaled to OBP
        # First get raw linear weights (relative to out)
        for event in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
            row[f"w{event}"] = row[f"rv_{event}"] - rv_out

        # Scale factor: wOBA_scale = league_OBP / league_wOBA_raw
        # league wOBA raw from counting events
        total_pa = len(yr)
        woba_num = sum(
            row[f"w{e}"] * yr[events[e]].sum()
            for e in ["BB", "HBP", "1B", "2B", "3B", "HR"]
        )
        woba_raw = woba_num / total_pa if total_pa > 0 else 0
        obp_num = yr[["walk", "hbp", "single", "double", "triple", "hr"]].sum().sum()
        league_obp = obp_num / total_pa if total_pa > 0 else 0

        scale = league_obp / woba_raw if woba_raw != 0 else 1
        row["wOBA_scale"] = scale

        # Apply scale to weights
        for event in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
            row[f"w{event}"] *= scale

        row["runs_per_out"] = -rv_out
        weights_list.append(row)

    return pd.DataFrame(weights_list)


def calculate_woba_wrc(batters, season_constants, league_avgs):
    """Calculate wOBA, wRAA, wRC, wRC+ for batters."""
    df = batters.copy()

    # Merge season constants
    df = df.merge(season_constants[["year", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR",
                                     "wOBA_scale", "runs_per_out"]],
                  on="year", how="left")
    if "lg_OBP" not in df.columns:
        df = df.merge(league_avgs[["year", "lg_OBP"]], on="year", how="left")
    drop_lg_obp = True  # track whether we should drop it at the end

    # wOBA = (wBB*BB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR) / (PA)
    # Excluding IBB from BB for wOBA
    non_ibb = df["b_w"] - df["b_iw"]
    b_1b = df["b_h"] - df["b_d"] - df["b_t"] - df["b_hr"]
    pa = df["b_pa"].replace(0, np.nan)

    df["wOBA"] = (
        df["wBB"] * non_ibb +
        df["wHBP"] * df["b_hbp"] +
        df["w1B"] * b_1b +
        df["w2B"] * df["b_d"] +
        df["w3B"] * df["b_t"] +
        df["wHR"] * df["b_hr"]
    ) / pa

    # wRAA = ((wOBA - lgwOBA) / wOBA_scale) * PA
    lg_woba = df["lg_OBP"]  # approximate: league wOBA ≈ league OBP after scaling
    df["wRAA"] = ((df["wOBA"] - lg_woba) / df["wOBA_scale"]) * df["b_pa"]

    # wRC = wRAA + (lgR/lgPA) * PA — use runs_per_out as proxy
    lg_rppa = df["runs_per_out"] * 3 / 9  # rough: runs per PA
    df["wRC"] = df["wRAA"] + lg_rppa * df["b_pa"]

    # wRC+ = ((wRAA/PA + lgR/PA) / lgR/PA) * 100, park adjusted
    pf = df["PF"].fillna(1.0) if "PF" in df.columns else 1.0
    runs_pa = df["runs_per_out"] * 0.33  # approximate runs per PA
    df["wRC_plus"] = ((df["wRAA"] / pa + runs_pa) / runs_pa.replace(0, np.nan)) * 100 / pf

    # Drop intermediate columns
    drop_cols = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR", "wOBA_scale", "runs_per_out"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def calculate_batted_ball_rates(plays):
    """Calculate GB%, FB%, LD%, HR/FB% per batter and pitcher per season."""
    pa_plays = plays[plays["pa"] == 1].copy()

    # BIP = ground + fly + line
    pa_plays["bip"] = pa_plays[["ground", "fly", "line"]].sum(axis=1).clip(upper=1)

    bip_plays = pa_plays[pa_plays["bip"] > 0]

    def bb_rates(group):
        total = len(group)
        gb = group["ground"].sum()
        fb = group["fly"].sum()
        ld = group["line"].sum()
        hr = group["hr"].sum()
        return pd.Series({
            "GB_pct": gb / total,
            "FB_pct": fb / total,
            "LD_pct": ld / total,
            "HR_FB_pct": hr / fb if fb > 0 else np.nan,
            "BIP": total
        })

    batter_bb = bip_plays.groupby(["batter", "year"]).apply(bb_rates, include_groups=False).reset_index()
    batter_bb.rename(columns={"batter": "id"}, inplace=True)

    pitcher_bb = bip_plays.groupby(["pitcher", "year"]).apply(bb_rates, include_groups=False).reset_index()
    pitcher_bb.rename(columns={"pitcher": "id"}, inplace=True)

    return batter_bb, pitcher_bb


def build_re24_outputs(cfg):
    """Full RE24 pipeline: load plays, build matrix, calculate RE24, wOBA, batted ball rates."""
    print("Loading plays.csv (chunked)...")
    plays = load_plays_filtered(cfg)

    print("Building RE matrix...")
    re_matrix = build_re_matrix(plays)

    print("Calculating RE24...")
    plays = calculate_re24(plays, re_matrix)

    print("Aggregating RE24 per player...")
    batter_re24, pitcher_re24 = aggregate_re24_players(plays)

    print("Deriving wOBA weights...")
    season_constants = derive_woba_weights(plays, re_matrix)

    print("Calculating batted ball rates...")
    batter_bb, pitcher_bb = calculate_batted_ball_rates(plays)

    # Save outputs
    re_matrix.to_csv(ROOT / cfg["paths"]["processed"]["re_matrix"], index=False)
    batter_re24.to_csv(ROOT / cfg["paths"]["processed"]["batter_re24"], index=False)
    pitcher_re24.to_csv(ROOT / cfg["paths"]["processed"]["pitcher_re24"], index=False)
    season_constants.to_csv(ROOT / cfg["paths"]["processed"]["season_constants"], index=False)
    batter_bb.to_csv(ROOT / cfg["paths"]["processed"]["batted_ball"].replace(".csv", "_batters.csv"), index=False)
    pitcher_bb.to_csv(ROOT / cfg["paths"]["processed"]["batted_ball"].replace(".csv", "_pitchers.csv"), index=False)

    print("RE24 pipeline complete.")
    return {
        "re_matrix": re_matrix,
        "batter_re24": batter_re24,
        "pitcher_re24": pitcher_re24,
        "season_constants": season_constants,
        "batter_bb": batter_bb,
        "pitcher_bb": pitcher_bb
    }
