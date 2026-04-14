"""Phase 7: Predictions, evaluation plots, draft rankings, PAR."""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_builder import load_config, ROOT
from src.feature_builder import build_shifted_dataset
from src.model_trainer import get_feature_cols, time_split, EnsembleModel, _SP_DROP_FEATURES, _RP_DROP_FEATURES

import xgboost as xgb
import lightgbm as lgb


def load_model_and_meta(model_key, cfg):
    """Load saved model and its metadata.

    Args:
        model_key: Either a player type ("batter", "pitcher") which maps to
                   "{model_key}_model", or a direct model key like "pitcher_sp_model".
    """
    # Support both "batter" -> "batter_model" and "pitcher_sp_model" -> "pitcher_sp_model"
    if model_key in ("batter", "pitcher"):
        output_key = f"{model_key}_model"
    else:
        output_key = model_key

    model_path = ROOT / cfg["paths"]["outputs"][output_key]
    meta_path = model_path.with_suffix(".meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["model_type"] == "ensemble":
        # Load XGBoost
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(model_path))
        # Load LightGBM
        lgb_path = ROOT / cfg["paths"]["outputs"][f"{output_key}_lgb"]
        lgb_model = lgb.Booster(model_file=str(lgb_path))
        # Reconstruct ensemble
        weights = np.array(meta["blend_weights"])
        intercept = meta.get("blend_intercept", 0.0)
        model = EnsembleModel(xgb_model, lgb_model, weights, intercept)
    elif meta["model_type"] == "xgboost":
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
    else:
        model = lgb.Booster(model_file=str(model_path))

    return model, meta


def predict_test_set(model, meta, test_df):
    """Generate predictions on test set."""
    feature_cols = meta["feature_cols"]
    clip_range = tuple(meta["clip_range"]) if meta.get("clip_range") else None
    preds = model.predict(test_df[feature_cols])
    if clip_range:
        preds = np.clip(preds, *clip_range)
    result = test_df[["id", "year", "last", "first", "team", "primary_pos", "target_ESPN_Pts"]].copy()
    result["predicted_pts"] = preds
    result["residual"] = result["target_ESPN_Pts"] - result["predicted_pts"]
    return result


def generate_enhanced_evaluation(test_results, player_type, cfg, year_label="2025"):
    """Extended evaluation: bias, top-N precision, position MAE, tier calibration, hits/misses."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    fig_dir = ROOT / cfg["paths"]["figures"]["model_evaluation"]
    df = test_results.copy()
    actual = df["target_ESPN_Pts"]
    predicted = df["predicted_pts"]
    residuals = actual - predicted  # actual - predicted

    # --- Summary metrics ---
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    spear, _ = spearmanr(predicted, actual)
    bias = residuals.mean()
    within_50 = (residuals.abs() <= 50).mean()
    within_100 = (residuals.abs() <= 100).mean()
    corr = np.corrcoef(predicted, actual)[0, 1]
    r2 = corr ** 2

    # Top-N precision at 25 and 50
    def topn_precision(df, n):
        top_pred = set(df.nlargest(n, "predicted_pts")["id"])
        top_actual = set(df.nlargest(n, "target_ESPN_Pts")["id"])
        return len(top_pred & top_actual) / n

    prec25 = topn_precision(df, min(25, len(df)))
    prec50 = topn_precision(df, min(50, len(df)))

    print(f"\n{'='*50}")
    print(f"{player_type.upper()} — {year_label} SEASON RETROSPECTIVE")
    print(f"{'='*50}")
    print(f"  Players evaluated: {len(df)}")
    print(f"  MAE: {mae:.1f} | RMSE: {rmse:.1f} | Spearman: {spear:.3f} | R²: {r2:.3f}")
    print(f"  Bias (actual-pred): {bias:+.1f} pts (+ = we underpredict)")
    print(f"  Within ±50 pts: {within_50:.1%} | Within ±100 pts: {within_100:.1%}")
    print(f"  Top-25 precision: {prec25:.1%} | Top-50 precision: {prec50:.1%}")

    # --- Plot 1: Scatter with player labels (top 15 by actual) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(predicted, actual, alpha=0.4, s=20, color="steelblue")
    mn = min(predicted.min(), actual.min()) - 20
    mx = max(predicted.max(), actual.max()) + 20
    ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5, label="Perfect prediction")
    top_actual = df.nlargest(15, "target_ESPN_Pts")
    for _, row in top_actual.iterrows():
        ax.annotate(row["last"], (row["predicted_pts"], row["target_ESPN_Pts"]),
                    fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Predicted ESPN Points")
    ax.set_ylabel(f"Actual {year_label} ESPN Points")
    ax.set_title(f"{player_type.title()} {year_label} Predictions vs Actual\n"
                 f"R²={r2:.3f}  Spearman={spear:.3f}  Bias={bias:+.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{player_type}_{year_label}_scatter_labeled.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Position MAE bar chart ---
    if "primary_pos" in df.columns:
        pos_mae = df.groupby("primary_pos").apply(
            lambda g: mean_absolute_error(g["target_ESPN_Pts"], g["predicted_pts"])
        ).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        pos_mae.plot(kind="barh", ax=ax, color="steelblue")
        ax.axvline(mae, color="red", linestyle="--", label=f"Overall MAE ({mae:.0f})")
        ax.set_xlabel("MAE (pts)")
        ax.set_title(f"{player_type.title()} Position-Level MAE — {year_label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{player_type}_{year_label}_position_mae.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # --- Plot 3: Tier calibration ---
    if "tier" in df.columns:
        tier_cal = df.groupby("tier").agg(
            median_actual=("target_ESPN_Pts", "median"),
            median_predicted=("predicted_pts", "median"),
            count=("id", "count")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(tier_cal))
        w = 0.35
        ax.bar(x - w/2, tier_cal["median_predicted"], w, label="Median Predicted", alpha=0.7)
        ax.bar(x + w/2, tier_cal["median_actual"], w, label="Median Actual", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Tier {t}\n(n={c})" for t, c in
                            zip(tier_cal["tier"], tier_cal["count"])])
        ax.set_ylabel("ESPN Points")
        ax.set_title(f"{player_type.title()} Tier Calibration — {year_label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{player_type}_{year_label}_tier_calibration.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # --- Table: Biggest hits and misses ---
    df["abs_residual"] = residuals.abs()
    cols = ["last", "first", "primary_pos", "predicted_pts", "target_ESPN_Pts", "residual"]
    hits = df.nsmallest(10, "abs_residual")[cols]
    misses = df.nlargest(10, "abs_residual")[cols]
    print(f"\n  TOP 10 CLOSEST PREDICTIONS:")
    print(hits.round(1).to_string(index=False))
    print(f"\n  TOP 10 BIGGEST MISSES:")
    print(misses.round(1).to_string(index=False))

    return {
        "mae": mae, "rmse": rmse, "spearman": spear, "r2": r2,
        "bias": bias, "within_50": within_50, "within_100": within_100,
        "top25_precision": prec25, "top50_precision": prec50,
    }


def generate_evaluation_plots(test_results, player_type, cfg):
    """Generate model evaluation plots."""
    fig_dir = ROOT / cfg["paths"]["figures"]["model_evaluation"]
    df = test_results.sort_values("predicted_pts", ascending=False)

    # 1. Top 50 predicted vs actual
    top50 = df.head(50)
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(top50))
    ax.bar(x, top50["predicted_pts"], alpha=0.6, label="Predicted", width=0.4)
    ax.bar([i + 0.4 for i in x], top50["target_ESPN_Pts"], alpha=0.6, label="Actual", width=0.4)
    ax.set_xlabel("Rank")
    ax.set_ylabel("ESPN Points")
    ax.set_title(f"Top 50 {player_type.title()}s: Predicted vs Actual")
    ax.legend()
    fig.savefig(fig_dir / f"{player_type}_top50_pred_vs_actual.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2. Scatter: predicted vs actual
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["predicted_pts"], df["target_ESPN_Pts"], alpha=0.5, s=20)
    mn, mx = min(df["predicted_pts"].min(), df["target_ESPN_Pts"].min()), max(df["predicted_pts"].max(), df["target_ESPN_Pts"].max())
    ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5)
    corr = np.corrcoef(df["predicted_pts"], df["target_ESPN_Pts"])[0, 1]
    r2 = corr ** 2
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Predicted Points")
    ax.set_ylabel("Actual Points")
    ax.set_title(f"{player_type.title()} Model: Predicted vs Actual")
    fig.savefig(fig_dir / f"{player_type}_scatter_pred_vs_actual.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 3. Residual distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["residual"], bins=40, alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_title(f"{player_type.title()} Residual Distribution")
    fig.savefig(fig_dir / f"{player_type}_residual_dist.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 4. Position accuracy: top 5 per position
    if "primary_pos" in df.columns:
        positions = df["primary_pos"].dropna().unique()
        pos_accuracy = []
        for pos in positions:
            pos_df = df[df["primary_pos"] == pos]
            if len(pos_df) < 5:
                continue
            top5_pred = pos_df.nlargest(5, "predicted_pts")["id"].tolist()
            top5_actual = pos_df.nlargest(5, "target_ESPN_Pts")["id"].tolist()
            overlap = len(set(top5_pred) & set(top5_actual))
            pos_accuracy.append({"position": pos, "top5_overlap": overlap, "players": len(pos_df)})
        if pos_accuracy:
            pos_df = pd.DataFrame(pos_accuracy).sort_values("top5_overlap", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(pos_df["position"], pos_df["top5_overlap"])
            ax.set_xlabel("Top 5 Overlap (out of 5)")
            ax.set_title(f"{player_type.title()} Position Accuracy")
            fig.savefig(fig_dir / f"{player_type}_position_accuracy.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    return len(df)


def generate_draft_rankings(model, meta, features_df, player_type, cfg):
    """Generate 2026 predictions from 2025 features."""
    feature_cols = meta["feature_cols"]

    # Use 2025 (predict year) features
    predict_year = cfg["seasons"]["predict"][0]
    current = features_df[features_df["year"] == predict_year].copy()

    if len(current) == 0:
        print(f"  No {player_type} data for {predict_year}")
        return pd.DataFrame()

    # Fill missing feature columns with NaN (model handles it)
    for col in feature_cols:
        if col not in current.columns:
            current[col] = np.nan

    clip_range = tuple(meta["clip_range"]) if meta.get("clip_range") else None
    preds = model.predict(current[feature_cols])
    if clip_range:
        preds = np.clip(preds, *clip_range)
    current["projected_pts_2026"] = preds

    # Add position info
    result = current[["id", "year", "last", "first", "team", "primary_pos",
                       "projected_pts_2026", "ESPN_Pts"]].copy()
    result.rename(columns={"ESPN_Pts": "pts_2025"}, inplace=True)

    # Calculate PAR by position
    result = calculate_par(result, player_type, cfg)

    # Risk flags
    result = add_risk_flags(result, features_df, cfg)

    # Sort by PAR
    result = result.sort_values("PAR", ascending=False).reset_index(drop=True)
    result["overall_rank"] = result.index + 1

    # Tier construction (based on PAR gaps > 1 SD)
    result = assign_tiers(result)

    return result


def calculate_par(rankings, player_type, cfg):
    """Points Above Replacement by position."""
    df = rankings.copy()

    if player_type == "batter":
        # Replacement level per position: ~80th percentile player at that position
        for pos in df["primary_pos"].dropna().unique():
            pos_mask = df["primary_pos"] == pos
            pos_players = df.loc[pos_mask, "projected_pts_2026"]
            if len(pos_players) >= 3:
                # Replacement = projected points of player ranked around 70th percentile at position
                replacement = pos_players.quantile(0.3)
            else:
                replacement = 0
            df.loc[pos_mask, "replacement_pts"] = replacement
    else:
        # SP and RP replacement levels
        if "is_starter" in df.columns:
            for role, mask_val in [("SP", 1), ("RP", 0)]:
                mask = df.get("is_starter", pd.Series()) == mask_val
                if mask.sum() >= 3:
                    replacement = df.loc[mask, "projected_pts_2026"].quantile(0.3)
                else:
                    replacement = 0
                df.loc[mask, "replacement_pts"] = replacement
        else:
            df["replacement_pts"] = df["projected_pts_2026"].quantile(0.3)

    df["replacement_pts"] = df["replacement_pts"].fillna(0)
    df["PAR"] = df["projected_pts_2026"] - df["replacement_pts"]
    return df


def add_risk_flags(rankings, features_df, cfg):
    """Add risk flags: regression, age, role risk."""
    df = rankings.copy()
    predict_year = cfg["seasons"]["predict"][0]

    flags = []
    for _, row in df.iterrows():
        player_flags = []
        pid = row["id"]
        feat = features_df[(features_df["id"] == pid) & (features_df["year"] == predict_year)]

        if len(feat) > 0:
            f = feat.iloc[0]
            # BABIP regression
            if "BABIP_gap" in f.index and pd.notna(f.get("BABIP_gap")) and f["BABIP_gap"] > 0.030:
                player_flags.append("BABIP_regression")
            # ERA regression for pitchers
            if "ERA_FIP_gap" in f.index and pd.notna(f.get("ERA_FIP_gap")) and f["ERA_FIP_gap"] < -0.50:
                player_flags.append("ERA_regression")
            # Experience as age proxy
            if "experience" in f.index and pd.notna(f.get("experience")) and f["experience"] >= 10:
                player_flags.append("age_risk")

        flags.append("; ".join(player_flags) if player_flags else "")

    df["risk_flags"] = flags
    return df


def assign_tiers(rankings):
    """Assign tiers based on PAR gaps > 1 SD."""
    df = rankings.copy()
    par_std = df["PAR"].std()
    tier = 1
    tiers = [tier]
    for i in range(1, len(df)):
        gap = df.iloc[i - 1]["PAR"] - df.iloc[i]["PAR"]
        if gap > par_std:
            tier += 1
        tiers.append(tier)
    df["tier"] = tiers
    return df


def _predict_split_pitchers(cfg, features, shifted):
    """Load SP and RP models, predict separately, combine results."""
    _, _, test = time_split(shifted, cfg)
    all_feature_cols = get_feature_cols(shifted)

    test_parts = []
    ranking_parts = []

    for role, mask_val, drop_feats, model_key in [
        ("pitcher_sp", 1, _SP_DROP_FEATURES, "pitcher_sp_model"),
        ("pitcher_rp", 0, _RP_DROP_FEATURES, "pitcher_rp_model"),
    ]:
        model, meta = load_model_and_meta(model_key, cfg)
        role_features = [c for c in all_feature_cols if c not in drop_feats]

        # Test evaluation for this role
        role_test = test[test["is_starter"] == mask_val]
        if len(role_test) > 0:
            # Override meta feature_cols with role-specific cols
            role_meta = dict(meta)
            role_meta["feature_cols"] = role_features
            test_results = predict_test_set(model, role_meta, role_test)
            test_parts.append(test_results)

        # Draft rankings for this role
        role_current = features[features["is_starter"] == mask_val].copy()
        role_meta = dict(meta)
        role_meta["feature_cols"] = role_features
        rankings = generate_draft_rankings(model, role_meta, role_current, "pitcher", cfg)
        if len(rankings) > 0:
            ranking_parts.append(rankings)

    # Combine
    combined_test = pd.concat(test_parts) if test_parts else pd.DataFrame()
    combined_rankings = pd.concat(ranking_parts) if ranking_parts else pd.DataFrame()

    if len(combined_rankings) > 0:
        combined_rankings = combined_rankings.sort_values("PAR", ascending=False).reset_index(drop=True)
        combined_rankings["overall_rank"] = combined_rankings.index + 1
        combined_rankings = assign_tiers(combined_rankings)

    return combined_test, combined_rankings


def build_predictions(cfg):
    """Full prediction pipeline."""
    results = {}
    split_pitchers = cfg["model"].get("split_pitcher_roles", False)

    for player_type in ["batter", "pitcher"]:
        print(f"\n{'='*50}")
        print(f"{player_type.upper()} PREDICTIONS")
        print(f"{'='*50}")

        # Load features
        if player_type == "batter":
            features = pd.read_csv(ROOT / cfg["paths"]["processed"]["batter_features"])
        else:
            features = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitcher_features"])

        shifted = build_shifted_dataset(features, cfg=cfg)

        if player_type == "pitcher" and split_pitchers and "is_starter" in features.columns:
            print("  Using separate SP/RP models...")
            test_results, rankings = _predict_split_pitchers(cfg, features, shifted)

            if len(test_results) > 0:
                n = generate_evaluation_plots(test_results, player_type, cfg)
                print(f"  Generated evaluation plots for {n} test players")

            if len(rankings) > 0:
                out_path = ROOT / cfg["paths"]["outputs"][f"draft_rankings_{player_type}s"]
                rankings.to_csv(out_path, index=False)
                print(f"  Saved {len(rankings)} {player_type} rankings")
                print(f"  Top 10:")
                print(rankings[["overall_rank", "last", "first", "primary_pos", "projected_pts_2026",
                                "PAR", "tier", "risk_flags"]].head(10).to_string(index=False))

            # Use SP meta for the report
            sp_model, sp_meta = load_model_and_meta("pitcher_sp_model", cfg)
            results[player_type] = {"rankings": rankings, "model": sp_model, "meta": sp_meta}
        else:
            model, meta = load_model_and_meta(player_type, cfg)
            _, _, test = time_split(shifted, cfg)

            if len(test) > 0:
                test_results = predict_test_set(model, meta, test)
                n = generate_evaluation_plots(test_results, player_type, cfg)
                print(f"  Generated evaluation plots for {n} test players")

            rankings = generate_draft_rankings(model, meta, features, player_type, cfg)
            if len(rankings) > 0:
                out_path = ROOT / cfg["paths"]["outputs"][f"draft_rankings_{player_type}s"]
                rankings.to_csv(out_path, index=False)
                print(f"  Saved {len(rankings)} {player_type} rankings")
                print(f"  Top 10:")
                print(rankings[["overall_rank", "last", "first", "primary_pos", "projected_pts_2026",
                                "PAR", "tier", "risk_flags"]].head(10).to_string(index=False))

            results[player_type] = {"rankings": rankings, "model": model, "meta": meta}

    # Model performance report
    report = []
    for pt in ["batter", "pitcher"]:
        m = results[pt]["meta"]
        report.append({
            "player_type": pt,
            "model_type": m["model_type"],
            **m["test_metrics"]
        })
    report_df = pd.DataFrame(report)
    report_df.to_csv(ROOT / cfg["paths"]["outputs"]["model_performance"], index=False)

    return results
