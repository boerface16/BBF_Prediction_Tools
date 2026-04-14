"""Phase 2: Calculate advanced batting/pitching rates, park factors, league adjustments."""

import pandas as pd
import numpy as np
from src.data_builder import load_config, ROOT


def calculate_batter_rates(batters):
    """Add rate stats to season-level batting data."""
    df = batters.copy()
    b = df  # alias for readability

    # Singles and total bases
    b["b_1b"] = b["b_h"] - b["b_d"] - b["b_t"] - b["b_hr"]
    b["TB"] = b["b_1b"] + 2 * b["b_d"] + 3 * b["b_t"] + 4 * b["b_hr"]

    # Rate stats (guard against division by zero)
    pa = b["b_pa"].replace(0, np.nan)
    ab = b["b_ab"].replace(0, np.nan)

    b["AVG"] = b["b_h"] / ab
    b["OBP"] = (b["b_h"] + b["b_w"] + b["b_hbp"]) / pa
    b["SLG"] = b["TB"] / ab
    b["OPS"] = b["OBP"] + b["SLG"]
    b["ISO"] = b["SLG"] - b["AVG"]

    # BABIP = (H - HR) / (AB - K - HR + SF)
    babip_denom = (b["b_ab"] - b["b_k"] - b["b_hr"] + b["b_sf"]).replace(0, np.nan)
    b["BABIP"] = (b["b_h"] - b["b_hr"]) / babip_denom

    b["K_pct"] = b["b_k"] / pa
    b["BB_pct"] = b["b_w"] / pa
    b["HR_PA"] = b["b_hr"] / pa
    b["BB_K_pct"] = b["BB_pct"] - b["K_pct"]

    # SB rate
    sb_opps = (b["b_sb"] + b["b_cs"]).replace(0, np.nan)
    b["SB_pct"] = b["b_sb"] / sb_opps

    return b


def calculate_pitcher_rates(pitchers):
    """Add rate stats to season-level pitching data."""
    df = pitchers.copy()
    p = df

    # IP from outs
    p["IP"] = p["p_ipouts"] / 3.0
    ip = p["IP"].replace(0, np.nan)
    bf = p["p_bfp"].replace(0, np.nan)

    p["ERA"] = (p["p_er"] * 9) / ip
    p["WHIP"] = (p["p_h"] + p["p_w"]) / ip
    p["K9"] = (p["p_k"] * 9) / ip
    p["BB9"] = (p["p_w"] * 9) / ip
    p["HR9"] = (p["p_hr"] * 9) / ip
    p["K_pct"] = p["p_k"] / bf
    p["BB_pct"] = p["p_w"] / bf
    p["K_BB_pct"] = p["K_pct"] - p["BB_pct"]
    p["K_BB"] = p["p_k"] / p["p_w"].replace(0, np.nan)

    # Pitcher BABIP = (H - HR) / (BF - K - HR - BB - HBP)
    babip_denom = (p["p_bfp"] - p["p_k"] - p["p_hr"] - p["p_w"] - p["p_hbp"]).replace(0, np.nan)
    p["BABIP_allowed"] = (p["p_h"] - p["p_hr"]) / babip_denom

    # LOB% = (H + BB + HBP - R) / (H + BB + HBP - 1.4 * HR)
    lob_num = p["p_h"] + p["p_w"] + p["p_hbp"] - p["p_r"]
    lob_den = (p["p_h"] + p["p_w"] + p["p_hbp"] - 1.4 * p["p_hr"]).replace(0, np.nan)
    p["LOB_pct"] = lob_num / lob_den

    # Role flag
    p["is_starter"] = (p["p_gs"] >= p["G"] * 0.5).astype(int)

    return p


def calculate_park_factors(cfg):
    """Calculate park factors from gameinfo + teamstats."""
    gi = pd.read_csv(ROOT / cfg["paths"]["raw"]["gameinfo"], low_memory=False)
    gi = gi[gi["gametype"].str.lower() == "regular"]
    gi["year"] = gi["season"]

    # Runs per game at each park
    gi["total_runs"] = gi["vruns"] + gi["hruns"]
    park_runs = gi.groupby(["year", "site"]).agg(
        total_runs=("total_runs", "sum"),
        games=("gid", "count")
    ).reset_index()
    park_runs["rpg"] = park_runs["total_runs"] / park_runs["games"]

    # League average RPG per year
    league_rpg = gi.groupby("year").agg(
        total_runs=("total_runs", "sum"),
        games=("gid", "count")
    ).reset_index()
    league_rpg["lg_rpg"] = league_rpg["total_runs"] / league_rpg["games"]

    park_runs = park_runs.merge(league_rpg[["year", "lg_rpg"]], on="year")
    park_runs["PF_single"] = park_runs["rpg"] / park_runs["lg_rpg"]

    # 5-year rolling average PF (min 1 year)
    park_runs = park_runs.sort_values(["site", "year"])
    park_runs["PF"] = park_runs.groupby("site")["PF_single"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Map hometeam to site per year
    home_site = gi.groupby(["year", "hometeam"])["site"].first().reset_index()
    home_site.rename(columns={"hometeam": "team"}, inplace=True)

    # Merge PF to team-year
    team_pf = home_site.merge(park_runs[["year", "site", "PF"]], on=["year", "site"], how="left")
    return team_pf[["year", "team", "site", "PF"]]


def calculate_league_averages(cfg):
    """Calculate league average OBP, SLG, ERA per season from teamstats."""
    ts = pd.read_csv(ROOT / cfg["paths"]["raw"]["teamstats"], low_memory=False)
    ts["year"] = ts["date"].astype(str).str[:4].astype(int)
    ts = ts[ts["gametype"].str.lower() == "regular"]
    ts = ts[ts["year"].between(2015, 2025)]
    ts = ts[ts["stattype"] == "value"]

    # League batting: aggregate per year
    bat_cols = ["b_pa", "b_ab", "b_h", "b_d", "b_t", "b_hr", "b_w", "b_hbp", "b_sf"]
    pit_cols = ["p_ipouts", "p_er", "p_h", "p_hr", "p_w", "p_k", "p_hbp", "p_bfp"]
    # Coerce any non-numeric values
    for c in bat_cols + pit_cols:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")
    lg = ts.groupby("year")[bat_cols + pit_cols].sum().reset_index()

    pa = lg["b_pa"].replace(0, np.nan)
    ab = lg["b_ab"].replace(0, np.nan)
    ip = (lg["p_ipouts"] / 3).replace(0, np.nan)

    lg["lg_OBP"] = (lg["b_h"] + lg["b_w"] + lg["b_hbp"]) / pa
    lg["lg_SLG"] = (lg["b_h"] - lg["b_d"] - lg["b_t"] - lg["b_hr"]
                    + 2 * lg["b_d"] + 3 * lg["b_t"] + 4 * lg["b_hr"]) / ab
    lg["lg_ERA"] = (lg["p_er"] * 9) / ip

    # FIP constant: lgERA - ((13*lgHR + 3*(lgBB+lgHBP) - 2*lgK) / lgIP)
    fip_raw = (13 * lg["p_hr"] + 3 * (lg["p_w"] + lg["p_hbp"]) - 2 * lg["p_k"]) / ip
    lg["FIP_constant"] = lg["lg_ERA"] - fip_raw
    lg["lg_FIP"] = fip_raw + lg["FIP_constant"]  # = lg_ERA by construction; use for FIP_minus denominator

    return lg[["year", "lg_OBP", "lg_SLG", "lg_ERA", "lg_FIP", "FIP_constant"]]


def apply_ops_plus(batters, park_factors, league_avgs):
    """Calculate OPS+ = 100 * (OBP/lgOBP + SLG/lgSLG - 1) / PF."""
    df = batters.merge(park_factors[["year", "team", "PF"]], on=["year", "team"], how="left")
    df = df.merge(league_avgs[["year", "lg_OBP", "lg_SLG"]], on="year", how="left")

    # Fill missing PF with 1.0 (neutral)
    df["PF"] = df["PF"].fillna(1.0)

    pf = df["PF"].replace(0, np.nan)
    lg_obp = df["lg_OBP"].replace(0, np.nan)
    lg_slg = df["lg_SLG"].replace(0, np.nan)

    df["OPS_plus"] = 100 * (df["OBP"] / lg_obp + df["SLG"] / lg_slg - 1) / pf
    return df


def apply_pitching_indices(pitchers, park_factors, league_avgs):
    """Calculate FIP, ERA-, FIP- (park/league adjusted minus stats)."""
    # Drop columns from prior runs to avoid merge suffixes
    prior_cols = ["PF", "lg_ERA", "lg_FIP", "FIP_constant", "FIP", "ERA_plus", "ERA_minus", "FIP_minus"]
    df = pitchers.drop(columns=[c for c in prior_cols if c in pitchers.columns])
    df = df.merge(park_factors[["year", "team", "PF"]], on=["year", "team"], how="left")
    df = df.merge(league_avgs[["year", "lg_ERA", "lg_FIP", "FIP_constant"]], on="year", how="left")

    df["PF"] = df["PF"].fillna(1.0)
    pf = df["PF"].replace(0, np.nan)
    lg_era = df["lg_ERA"].replace(0, np.nan)
    lg_fip = df["lg_FIP"].replace(0, np.nan)
    ip = df["IP"].replace(0, np.nan)

    # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_constant
    df["FIP"] = (13 * df["p_hr"] + 3 * (df["p_w"] + df["p_hbp"]) - 2 * df["p_k"]) / ip + df["FIP_constant"]

    # Minus stats: lower = better, 100 = league average
    # Standard formula: (stat / lg_stat) / PF * 100
    df["ERA_minus"] = 100 * (df["ERA"] / lg_era) / pf
    df["FIP_minus"] = 100 * (df["FIP"] / lg_fip) / pf

    return df


def build_advanced_batters(cfg):
    """Full pipeline for batter advanced stats."""
    batters = pd.read_csv(ROOT / cfg["paths"]["processed"]["batters_season"])
    batters = calculate_batter_rates(batters)
    pf = calculate_park_factors(cfg)
    lg = calculate_league_averages(cfg)
    batters = apply_ops_plus(batters, pf, lg)
    batters.to_csv(ROOT / cfg["paths"]["processed"]["batters_season"], index=False)
    return batters


def build_advanced_pitchers(cfg):
    """Full pipeline for pitcher advanced stats."""
    pitchers = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"])
    pitchers = calculate_pitcher_rates(pitchers)
    pf = calculate_park_factors(cfg)
    lg = calculate_league_averages(cfg)
    pitchers = apply_pitching_indices(pitchers, pf, lg)
    pitchers.to_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"], index=False)
    return pitchers
