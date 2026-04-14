"""Phase 8: Keeper evaluator — trajectory scores, ML projections, recommendations."""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_builder import load_config, ROOT
from src.predictor import load_model_and_meta


def load_keeper_candidates(cfg):
    """Load keeper candidates from tab-separated file."""
    path = ROOT / cfg["paths"]["raw"]["keepers"]
    candidates = pd.read_csv(path, sep="\t")
    candidates.columns = ["position", "first", "last"]
    return candidates


def _normalize_name(name):
    """Strip accents and suffixes for fuzzy matching."""
    import unicodedata
    normalized = unicodedata.normalize("NFD", str(name))
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Remove common suffixes
    for suffix in [" Jr.", " Sr.", " II", " III", " IV"]:
        stripped = stripped.replace(suffix, "")
    return stripped.strip().lower()


def match_candidates(candidates, batters, pitchers):
    """Match keeper candidates to master tables by name (accent-insensitive)."""
    # Pre-compute normalized names
    for df in [batters, pitchers]:
        df["_last_norm"] = df["last"].apply(_normalize_name)
        df["_first_norm"] = df["first"].apply(_normalize_name)

    matched = []
    for _, row in candidates.iterrows():
        last_n = _normalize_name(row["last"])
        first_n = _normalize_name(row["first"])
        found = False

        # Determine search order: pitchers first if position is P
        if row["position"] == "P":
            search_order = [(pitchers, "pitcher"), (batters, "batter")]
        else:
            search_order = [(batters, "batter"), (pitchers, "pitcher")]

        # Exact match (normalized)
        for df, ptype in search_order:
            mask = (df["_last_norm"] == last_n) & (df["_first_norm"] == first_n)
            player = df[mask]
            if len(player) > 0:
                matched.append({
                    "position": row["position"],
                    "first": row["first"],
                    "last": row["last"],
                    "player_type": ptype,
                    "id": player["id"].iloc[0],
                    "data": player
                })
                found = True
                break

        if found:
            continue

        # Fuzzy: last name only
        for df, ptype in search_order:
            mask = df["_last_norm"] == last_n
            player = df[mask]
            if len(player) > 0:
                matched.append({
                    "position": row["position"],
                    "first": row["first"],
                    "last": row["last"],
                    "player_type": ptype,
                    "id": player["id"].iloc[0],
                    "data": player
                })
                found = True
                break

        if not found:
            matched.append({
                "position": row["position"],
                "first": row["first"],
                "last": row["last"],
                "player_type": "unknown",
                "id": None,
                "data": pd.DataFrame()
            })

    # Clean up temp columns
    for df in [batters, pitchers]:
        df.drop(columns=["_last_norm", "_first_norm"], inplace=True, errors="ignore")

    return matched


def calculate_trajectory(player_data, predict_year):
    """Calculate trajectory score from recent 2-year ESPN_Pts history."""
    if len(player_data) == 0 or "ESPN_Pts" not in player_data.columns:
        return {"raw_score": np.nan, "ppg_score": np.nan, "trajectory": np.nan}

    recent = player_data.sort_values("year", ascending=False).head(3)
    if len(recent) < 2:
        return {"raw_score": np.nan, "ppg_score": np.nan, "trajectory": np.nan}

    # Raw points trajectory
    pts = recent["ESPN_Pts"].values
    pts_delta = pts[0] - pts[1] if len(pts) >= 2 else 0

    # Points per game trajectory
    games = recent["G"].values
    ppg = pts / np.maximum(games, 1)
    ppg_delta = ppg[0] - ppg[1] if len(ppg) >= 2 else 0

    return {
        "raw_score": pts_delta,
        "ppg_score": ppg_delta,
        "trajectory": 0.5 * pts_delta + 0.5 * ppg_delta,
        "recent_pts": pts[0],
        "recent_ppg": ppg[0],
        "recent_games": games[0],
        "prior_pts": pts[1] if len(pts) >= 2 else np.nan,
    }


def get_ml_projection(player_id, player_type, features_df, model, meta, predict_year):
    """Get ML projection for a specific player."""
    feature_cols = meta["feature_cols"]
    player_feat = features_df[(features_df["id"] == player_id) & (features_df["year"] == predict_year)]

    if len(player_feat) == 0:
        return np.nan

    for col in feature_cols:
        if col not in player_feat.columns:
            player_feat = player_feat.copy()
            player_feat[col] = np.nan

    return float(model.predict(player_feat[feature_cols])[0])


def combine_scores(trajectory_score, ml_projection, cfg):
    """Combine trajectory and ML scores with configured weights."""
    if pd.isna(trajectory_score) and pd.isna(ml_projection):
        return np.nan, "insufficient_data"

    tw = cfg["keeper"]["trajectory_weight"]
    mw = cfg["keeper"]["ml_weight"]

    if pd.isna(trajectory_score):
        return ml_projection, "ml_only"
    if pd.isna(ml_projection):
        return trajectory_score, "trajectory_only"

    return tw * trajectory_score + mw * ml_projection, "combined"


def classify_keeper(combined_score, all_scores, cfg):
    """Classify keeper recommendation based on score percentile."""
    if pd.isna(combined_score):
        return "INSUFFICIENT_DATA"

    valid_scores = [s for s in all_scores if not pd.isna(s)]
    if not valid_scores:
        return "INSUFFICIENT_DATA"

    mn, mx = min(valid_scores), max(valid_scores)
    if mx == mn:
        normalized = 0.5
    else:
        normalized = (combined_score - mn) / (mx - mn)

    thresholds = cfg["keeper"]["thresholds"]
    if normalized >= thresholds["auto_keep"]:
        return "AUTO-KEEP"
    elif normalized >= thresholds["keep"]:
        return "KEEP"
    elif normalized >= thresholds["borderline"]:
        return "BORDERLINE"
    else:
        return "CUT"


def generate_keeper_plots(keeper_results, batters, pitchers, cfg):
    """Generate trajectory plots for each keeper candidate."""
    fig_dir = ROOT / cfg["paths"]["figures"]["keeper_trajectories"]

    for _, row in keeper_results.iterrows():
        pid = row["id"]
        if pd.isna(pid):
            continue

        # Get historical data
        if row["player_type"] == "batter":
            hist = batters[batters["id"] == pid].sort_values("year")
        else:
            hist = pitchers[pitchers["id"] == pid].sort_values("year")

        if len(hist) < 2 or "ESPN_Pts" not in hist.columns:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ESPN Points over time
        ax1.plot(hist["year"], hist["ESPN_Pts"], "o-", linewidth=2)
        if not pd.isna(row.get("ml_projection")):
            ax1.scatter([hist["year"].max() + 1], [row["ml_projection"]], color="red",
                       s=100, zorder=5, label="2026 Projection")
            ax1.legend()
        ax1.set_xlabel("Year")
        ax1.set_ylabel("ESPN Points")
        ax1.set_title(f"{row['first']} {row['last']} — Points Trajectory")
        bounds = cfg["model"]["prediction_bounds"]
        ylim = bounds.get(row["player_type"], [0, 750])
        ax1.set_ylim(ylim[0], ylim[1])

        # PPG over time
        ppg = hist["ESPN_Pts"] / hist["G"].replace(0, np.nan)
        ax2.plot(hist["year"], ppg, "o-", linewidth=2, color="green")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Points Per Game")
        ax2.set_title(f"{row['first']} {row['last']} — PPG Trajectory")
        if row["player_type"] == "pitcher":
            ax2.set_ylim(0, 25)
        else:
            ax2.set_ylim(0, 4)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        fig.tight_layout()
        safe_name = f"{row['last']}_{row['first']}".replace(" ", "_").replace(".", "")
        fig.savefig(fig_dir / f"{safe_name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def generate_keeper_group_plots(keeper_results, batters, pitchers, cfg):
    """Generate grouped overlay plots: INF, OF, and Pitchers."""
    fig_dir = ROOT / cfg["paths"]["figures"]["keeper_trajectories"]
    bounds = cfg["model"]["prediction_bounds"]
    predict_year = cfg["seasons"]["predict"][0]

    # Define groups by position
    inf_positions = {"C", "1B", "2B", "3B", "SS"}
    groups = {
        "INF": keeper_results[keeper_results["position"].isin(inf_positions)],
        "OF": keeper_results[keeper_results["position"] == "OF"],
        "pitchers": keeper_results[keeper_results["position"] == "P"],
    }

    # Last 4 data years + projection year
    data_years = list(range(predict_year - 4, predict_year))

    for group_name, group_df in groups.items():
        if len(group_df) == 0:
            continue

        ylim = bounds.get("pitcher", [0, 650]) if group_name == "pitchers" else bounds.get("batter", [0, 750])
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (_, row) in enumerate(group_df.iterrows()):
            pid = row["id"]
            if pd.isna(pid):
                continue

            source = pitchers if row["player_type"] == "pitcher" else batters
            hist = source[(source["id"] == pid) & (source["year"].isin(data_years))].sort_values("year")

            if len(hist) == 0 or "ESPN_Pts" not in hist.columns:
                continue

            color = cmap(i % 20)
            label = f"{row['first']} {row['last']}"

            # Historical line
            ax.plot(hist["year"], hist["ESPN_Pts"], "o-", color=color, linewidth=1.5, label=label)

            # 2026 ML projection as star marker
            if not pd.isna(row.get("ml_projection")):
                ax.scatter([predict_year], [row["ml_projection"]], color=color,
                           marker="*", s=150, zorder=5)

        ax.set_xlim(data_years[0] - 0.5, predict_year + 0.5)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Year")
        ax.set_ylabel("ESPN Points")
        ax.set_title(f"Keeper Trajectories — {group_name}")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()
        fig.savefig(fig_dir / f"group_{group_name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def build_keeper_rankings(cfg):
    """Full keeper evaluation pipeline."""
    print("Loading data...")
    candidates = load_keeper_candidates(cfg)
    batters = pd.read_csv(ROOT / cfg["paths"]["processed"]["batters_season"])
    pitchers = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitchers_season"])

    # Load features and models
    bat_features = pd.read_csv(ROOT / cfg["paths"]["processed"]["batter_features"])
    pit_features = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitcher_features"])
    predict_year = cfg["seasons"]["predict"][0]

    bat_model, bat_meta = load_model_and_meta("batter", cfg)

    # Load pitcher models (split SP/RP or unified)
    split_pitchers = cfg["model"].get("split_pitcher_roles", False)
    if split_pitchers:
        sp_model, sp_meta = load_model_and_meta("pitcher_sp_model", cfg)
        rp_model, rp_meta = load_model_and_meta("pitcher_rp_model", cfg)
    else:
        pit_model, pit_meta = load_model_and_meta("pitcher", cfg)

    # Match candidates
    matched = match_candidates(candidates, batters, pitchers)

    # Calculate scores
    results = []
    for m in matched:
        traj = calculate_trajectory(m["data"], predict_year)

        ml_proj = np.nan
        if m["id"] is not None:
            if m["player_type"] == "batter":
                ml_proj = get_ml_projection(m["id"], "batter", bat_features, bat_model, bat_meta, predict_year)
            elif m["player_type"] == "pitcher":
                if split_pitchers:
                    # Route to SP or RP model based on is_starter flag
                    player_row = pit_features[(pit_features["id"] == m["id"]) & (pit_features["year"] == predict_year)]
                    is_sp = len(player_row) > 0 and player_row.iloc[0].get("is_starter", 0) == 1
                    if is_sp:
                        ml_proj = get_ml_projection(m["id"], "pitcher", pit_features, sp_model, sp_meta, predict_year)
                    else:
                        ml_proj = get_ml_projection(m["id"], "pitcher", pit_features, rp_model, rp_meta, predict_year)
                else:
                    ml_proj = get_ml_projection(m["id"], "pitcher", pit_features, pit_model, pit_meta, predict_year)

        results.append({
            "position": m["position"],
            "first": m["first"],
            "last": m["last"],
            "player_type": m["player_type"],
            "id": m["id"],
            "recent_pts": traj.get("recent_pts", np.nan),
            "prior_pts": traj.get("prior_pts", np.nan),
            "recent_ppg": traj.get("recent_ppg", np.nan),
            "trajectory_score": traj.get("trajectory", np.nan),
            "ml_projection": ml_proj,
        })

    results_df = pd.DataFrame(results)

    # Normalize trajectory scores
    traj_scores = results_df["trajectory_score"].dropna()
    if len(traj_scores) > 0:
        t_min, t_max = traj_scores.min(), traj_scores.max()
        if t_max != t_min:
            results_df["traj_norm"] = (results_df["trajectory_score"] - t_min) / (t_max - t_min)
        else:
            results_df["traj_norm"] = 0.5
    else:
        results_df["traj_norm"] = np.nan

    # Normalize ML projections
    ml_scores = results_df["ml_projection"].dropna()
    if len(ml_scores) > 0:
        m_min, m_max = ml_scores.min(), ml_scores.max()
        if m_max != m_min:
            results_df["ml_norm"] = (results_df["ml_projection"] - m_min) / (m_max - m_min)
        else:
            results_df["ml_norm"] = 0.5
    else:
        results_df["ml_norm"] = np.nan

    # Combined score
    tw = cfg["keeper"]["trajectory_weight"]
    mw = cfg["keeper"]["ml_weight"]
    results_df["combined_score"] = np.where(
        results_df["traj_norm"].notna() & results_df["ml_norm"].notna(),
        tw * results_df["traj_norm"] + mw * results_df["ml_norm"],
        np.where(results_df["ml_norm"].notna(), results_df["ml_norm"],
                 np.where(results_df["traj_norm"].notna(), results_df["traj_norm"], np.nan))
    )

    # Divergence flag
    results_df["signal_divergence"] = np.where(
        results_df["traj_norm"].notna() & results_df["ml_norm"].notna(),
        (np.abs(results_df["traj_norm"] - results_df["ml_norm"]) * 100 >
         cfg["keeper"]["divergence_threshold"]),
        False
    )

    # Classification
    all_scores = results_df["combined_score"].tolist()
    results_df["recommendation"] = results_df["combined_score"].apply(
        lambda s: classify_keeper(s, all_scores, cfg)
    )

    # Sort by combined score
    results_df = results_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Generate plots
    print("Generating keeper trajectory plots...")
    generate_keeper_plots(results_df, batters, pitchers, cfg)
    generate_keeper_group_plots(results_df, batters, pitchers, cfg)

    # Save
    out_path = ROOT / cfg["paths"]["outputs"]["keeper_rankings"]
    results_df.to_csv(out_path, index=False)
    print(f"\nKeeper rankings saved: {out_path}")
    print(results_df[["position", "first", "last", "recent_pts", "ml_projection",
                       "combined_score", "recommendation", "signal_divergence"]].to_string(index=False))

    return results_df
