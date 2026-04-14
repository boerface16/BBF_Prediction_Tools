"""Phase 4: Calculate ESPN H2H fantasy points — the ML target variable."""

import pandas as pd
from src.data_builder import load_config, ROOT


def calculate_batter_points(batters, cfg):
    """ESPN_Pts = TB + R + RBI + BB + SB - SO"""
    df = batters.copy()
    # Ensure TB exists
    if "TB" not in df.columns:
        b_1b = df["b_h"] - df["b_d"] - df["b_t"] - df["b_hr"]
        df["TB"] = b_1b + 2 * df["b_d"] + 3 * df["b_t"] + 4 * df["b_hr"]

    scoring = cfg["espn_scoring"]["batter"]
    df["ESPN_Pts"] = (
        df["TB"] * scoring["TB"] +
        df["b_r"] * scoring["R"] +
        df["b_rbi"] * scoring["RBI"] +
        df["b_w"] * scoring["BB"] +
        df["b_sb"] * scoring["SB"] +
        df["b_k"] * scoring["SO"]  # SO weight is -1
    )
    return df


def calculate_pitcher_points(pitchers, cfg):
    """ESPN_Pts = IP*3 + K + W*5 + SV*5 + HD*2 - H - ER*2 - BB - L*2"""
    df = pitchers.copy()
    # Ensure IP exists
    if "IP" not in df.columns:
        df["IP"] = df["p_ipouts"] / 3.0

    scoring = cfg["espn_scoring"]["pitcher"]
    df["ESPN_Pts"] = (
        df["IP"] * scoring["IP"] +
        df["p_k"] * scoring["K"] +
        df["W"] * scoring["W"] +
        df["SV"] * scoring["SV"] +
        df["HD"] * scoring["HD"] +
        df["p_h"] * scoring["H"] +    # H weight is -1
        df["p_er"] * scoring["ER"] +   # ER weight is -2
        df["p_w"] * scoring["BB"] +    # BB weight is -1
        df["L"] * scoring["L"]         # L weight is -2
    )
    return df


def build_espn_points(cfg):
    """Add ESPN_Pts to both batters and pitchers season files."""
    batters = pd.read_csv(ROOT / cfg["paths"]["processed"]["batters_season"])
    pitchers = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"])

    batters = calculate_batter_points(batters, cfg)
    pitchers = calculate_pitcher_points(pitchers, cfg)

    batters.to_csv(ROOT / cfg["paths"]["processed"]["batters_season"], index=False)
    pitchers.to_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"], index=False)

    return batters, pitchers
