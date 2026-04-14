"""Phase 1: Aggregate game-level Retrosheet data to season-level."""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def extract_year(df, date_col="date"):
    """Extract year from YYYYMMDD integer date column."""
    df["year"] = df[date_col].astype(str).str[:4].astype(int)
    return df


def aggregate_batting(cfg):
    """Aggregate game-level batting to season-level."""
    path = ROOT / cfg["paths"]["raw"]["batting"]
    df = pd.read_csv(path, low_memory=False)
    df = extract_year(df)
    df = df[df["gametype"] == "regular"]
    df = df[df["year"].between(2015, 2025)]

    # Counting stats to sum
    sum_cols = [
        "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr", "b_rbi",
        "b_sh", "b_sf", "b_hbp", "b_w", "b_iw", "b_k", "b_sb", "b_cs", "b_gdp"
    ]
    agg_dict = {c: "sum" for c in sum_cols}
    agg_dict["gid"] = "nunique"

    season = df.groupby(["id", "year"]).agg(agg_dict).reset_index()
    season.rename(columns={"gid": "G"}, inplace=True)
    return season


def aggregate_pitching(cfg):
    """Aggregate game-level pitching to season-level, deriving W/L/SV."""
    path = ROOT / cfg["paths"]["raw"]["pitching"]
    df = pd.read_csv(path, low_memory=False)
    df = extract_year(df)
    df = df[df["gametype"] == "regular"]
    df = df[df["year"].between(2015, 2025)]

    # wp/lp/save are 1.0/NaN flags for pitcher decisions
    df["W"] = df["wp"].fillna(0).astype(int)
    df["L"] = df["lp"].fillna(0).astype(int)
    df["SV"] = df["save"].fillna(0).astype(int)

    sum_cols = [
        "p_ipouts", "p_bfp", "p_h", "p_d", "p_t", "p_hr", "p_r", "p_er",
        "p_w", "p_iw", "p_k", "p_hbp", "p_gs", "p_gf", "W", "L", "SV"
    ]
    agg_dict = {c: "sum" for c in sum_cols}
    agg_dict["gid"] = "nunique"

    season = df.groupby(["id", "year"]).agg(agg_dict).reset_index()
    season.rename(columns={"gid": "G"}, inplace=True)
    # Holds not available — set to 0
    season["HD"] = 0
    return season


def get_primary_position(row):
    """Determine primary position from games-at-position columns."""
    pos_cols = {
        "C": "g_c", "1B": "g_1b", "2B": "g_2b", "3B": "g_3b",
        "SS": "g_ss", "LF": "g_lf", "CF": "g_cf", "RF": "g_rf",
        "OF": "g_of", "DH": "g_dh", "P": "g_p"
    }
    games = {pos: row.get(col, 0) for pos, col in pos_cols.items()}
    # OF is aggregate — only use if individual OF positions are 0
    if games.get("LF", 0) + games.get("CF", 0) + games.get("RF", 0) > 0:
        games.pop("OF", None)
    if max(games.values()) == 0:
        return "UTIL"
    return max(games, key=lambda p: games[p])


def merge_player_info(season_df, cfg):
    """Merge allplayers.csv for names and primary position."""
    path = ROOT / cfg["paths"]["raw"]["allplayers"]
    players = pd.read_csv(path)

    # Get primary position per player-season
    players["primary_pos"] = players.apply(get_primary_position, axis=1)
    info = players[["id", "season", "last", "first", "team", "g", "primary_pos"]].copy()
    info.rename(columns={"season": "year"}, inplace=True)

    # Keep only the team row with the most games (handles multi-team/all-star entries)
    info = info.sort_values("g", ascending=False).drop_duplicates(subset=["id", "year"], keep="first")
    info.drop(columns=["g"], inplace=True)

    merged = season_df.merge(info, on=["id", "year"], how="left")
    return merged


def apply_thresholds(df, stat_type, cfg):
    """Flag players below sample thresholds."""
    thresh = cfg["thresholds"]
    if stat_type == "batter":
        pa_col = "b_pa"
        normal_thresh = thresh["batter_pa"]
        covid_thresh = thresh["batter_pa_2020"]
    else:
        pa_col = "p_ipouts"
        normal_thresh = thresh["pitcher_ip_outs"]
        covid_thresh = thresh["pitcher_ip_outs_2020"]

    df["LOW_SAMPLE"] = False
    mask_2020 = df["year"] == 2020
    df.loc[mask_2020, "LOW_SAMPLE"] = df.loc[mask_2020, pa_col] < covid_thresh
    df.loc[~mask_2020, "LOW_SAMPLE"] = df.loc[~mask_2020, pa_col] < normal_thresh
    return df


def build_batters(cfg):
    """Full pipeline: aggregate -> merge info -> threshold -> save."""
    batters = aggregate_batting(cfg)
    batters = merge_player_info(batters, cfg)
    batters = apply_thresholds(batters, "batter", cfg)
    out_path = ROOT / cfg["paths"]["processed"]["batters_season"]
    batters.to_csv(out_path, index=False)
    return batters


def build_pitchers(cfg):
    """Full pipeline: aggregate -> merge info -> threshold -> save."""
    pitchers = aggregate_pitching(cfg)
    pitchers = merge_player_info(pitchers, cfg)
    pitchers = apply_thresholds(pitchers, "pitcher", cfg)
    out_path = ROOT / cfg["paths"]["processed"]["pitchers_season"]
    pitchers.to_csv(out_path, index=False)
    return pitchers
