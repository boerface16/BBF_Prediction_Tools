"""Phase 6: Model training — XGBoost/LightGBM with Optuna tuning and SHAP analysis."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

from src.data_builder import load_config, ROOT
from src.feature_builder import build_shifted_dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Columns to exclude from features
META_COLS = ["id", "year", "last", "first", "team", "primary_pos", "target_ESPN_Pts"]
# Counting stats that would leak (raw counts correlate too directly with points)
# Exclude raw counting stats that directly compose the target (year N counts ≠ leak,
# but ESPN_Pts = TB+R+RBI+BB+SB-SO makes these near-perfect proxies of the same-year target).
# Keep volume indicators (PA, G, IP) and rate stats.
COUNTING_LEAK_COLS = [
    "b_hr", "b_r", "b_rbi", "b_sb", "b_k", "b_w",
    "p_k", "p_h", "p_er", "p_w", "p_hr", "p_hbp",
    "W", "L", "SV", "HD",
    "TB", "b_1b", "b_d", "b_t", "b_ab", "b_sh", "b_sf", "b_hbp", "b_iw",
    "b_cs", "b_gdp",
    "PF", "lg_OBP", "lg_SLG", "lg_ERA", "FIP_constant", "LOW_SAMPLE",
    "wRC",  # intermediate calc
    "covid_feature_year",  # used for sample weights, not a feature
]


def get_feature_cols(df):
    """Get feature columns by excluding meta and counting leak columns."""
    exclude = set(META_COLS) | set(COUNTING_LEAK_COLS)
    return [c for c in df.columns if c not in exclude]


def time_split(shifted_df, cfg, player_type="batter"):
    """Split by year: train, val, test according to config.

    Uses per-role train years if available (e.g. train_batter, train_pitcher).
    """
    train_key = f"train_{player_type}"
    train_years = cfg["seasons"].get(train_key, cfg["seasons"]["train"])
    val_years = cfg["seasons"]["validation"]
    test_years = cfg["seasons"]["test"]

    train = shifted_df[shifted_df["year"].isin(train_years)]
    val = shifted_df[shifted_df["year"].isin(val_years)]
    test = shifted_df[shifted_df["year"].isin(test_years)]

    return train, val, test


def evaluate_model(model, X, y, label="", clip_range=None):
    """Calculate MAE, RMSE, Spearman for a model's predictions."""
    preds = model.predict(X)
    if clip_range:
        preds = np.clip(preds, *clip_range)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    spearman, pval = spearmanr(y, preds)
    return {"label": label, "MAE": mae, "RMSE": rmse, "Spearman": spearman, "p_value": pval}


def topn_precision(y_true, y_pred, n=50):
    """Fraction of predicted top-N that are actually in the true top-N."""
    df = pd.DataFrame({"actual": y_true.values, "predicted": y_pred})
    actual_topn = set(df.nlargest(n, "actual").index)
    pred_topn = set(df.nlargest(n, "predicted").index)
    overlap = len(actual_topn & pred_topn)
    return overlap / n


def train_baseline(train, val, feature_cols, random_state=42):
    """Train XGBoost and LightGBM with defaults, return best."""
    X_train = train[feature_cols]
    y_train = train["target_ESPN_Pts"]
    X_val = val[feature_cols]
    y_val = val["target_ESPN_Pts"]

    # XGBoost — Huber loss to limit outlier influence
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:pseudohubererror", huber_slope=100,
        random_state=random_state, enable_categorical=False,
        early_stopping_rounds=50
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False)
    xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "XGBoost_baseline")

    # LightGBM — Huber loss
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="huber", alpha=100,
        random_state=random_state, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    lgb_metrics = evaluate_model(lgb_model, X_val, y_val, "LightGBM_baseline")

    print(f"  XGBoost:  MAE={xgb_metrics['MAE']:.1f}, Spearman={xgb_metrics['Spearman']:.3f}")
    print(f"  LightGBM: MAE={lgb_metrics['MAE']:.1f}, Spearman={lgb_metrics['Spearman']:.3f}")

    # Return the better model
    if xgb_metrics["Spearman"] >= lgb_metrics["Spearman"]:
        return xgb_model, "xgboost", xgb_metrics, lgb_metrics
    else:
        return lgb_model, "lightgbm", lgb_metrics, xgb_metrics


def tune_with_optuna(shifted_df, feature_cols, model_type, n_trials, random_state=42,
                     sample_weight_cfg=None, cv_folds=None, search_space=None):
    """Optuna hyperparameter tuning with expanding-window time-series CV.

    Args:
        shifted_df: Full shifted dataset (all years).
        feature_cols: Feature column names.
        model_type: "xgboost" or "lightgbm".
        n_trials: Number of Optuna trials.
        random_state: Random seed.
        sample_weight_cfg: COVID config dict for sample weights, or None.
        cv_folds: List of dicts with 'train_end' and 'val' keys. If None, falls back
                  to single split using the last fold.
        search_space: Optional dict overriding default param ranges (e.g. for RP).
    """
    if cv_folds is None:
        cv_folds = [{"train_end": 2022, "val": [2023]}]

    # Pre-compute fold data
    fold_data = []
    for fold in cv_folds:
        train_end = fold["train_end"]
        val_years = fold["val"]
        f_train = shifted_df[shifted_df["year"] <= train_end]
        f_val = shifted_df[shifted_df["year"].isin(val_years)]
        sw = _compute_sample_weights(f_train, {"covid": sample_weight_cfg}) if sample_weight_cfg else None
        fold_data.append((f_train[feature_cols], f_train["target_ESPN_Pts"],
                          f_val[feature_cols], f_val["target_ESPN_Pts"], sw))

    def objective(trial):
        sp = search_space or {}
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *sp.get("n_estimators", (200, 1000))),
            "max_depth": trial.suggest_int("max_depth", *sp.get("max_depth", (3, 10))),
            "learning_rate": trial.suggest_float("learning_rate", *sp.get("learning_rate", (0.01, 0.2)), log=True),
            "subsample": trial.suggest_float("subsample", *sp.get("subsample", (0.6, 1.0))),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *sp.get("colsample_bytree", (0.5, 1.0))),
            "min_child_weight": trial.suggest_int("min_child_weight", *sp.get("min_child_weight", (1, 20))),
            "reg_alpha": trial.suggest_float("reg_alpha", *sp.get("reg_alpha", (1e-8, 10.0)), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *sp.get("reg_lambda", (1e-8, 10.0)), log=True),
            "random_state": random_state,
        }
        if model_type == "xgboost":
            params["enable_categorical"] = False
            params["objective"] = "reg:pseudohubererror"
            params["huber_slope"] = trial.suggest_float("huber_slope", 50.0, 150.0)
        else:
            params["verbose"] = -1
            params["objective"] = "huber"
            params["alpha"] = trial.suggest_float("alpha", 50.0, 150.0)

        fold_rmses = []
        for X_tr, y_tr, X_vl, y_vl, sw in fold_data:
            if model_type == "xgboost":
                model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                          sample_weight=sw, verbose=False)
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                          sample_weight=sw,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            preds = model.predict(X_vl)
            fold_rmses.append(np.sqrt(mean_squared_error(y_vl, preds)))

        return np.mean(fold_rmses)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Retrain with best params on full training data
    best = study.best_params
    best["random_state"] = random_state
    # Use all training data (all folds combined = full train set passed in shifted_df minus test)
    X_full = shifted_df[feature_cols]
    y_full = shifted_df["target_ESPN_Pts"]
    sw_full = _compute_sample_weights(shifted_df, {"covid": sample_weight_cfg}) if sample_weight_cfg else None

    if model_type == "xgboost":
        best["enable_categorical"] = False
        best["objective"] = "reg:pseudohubererror"
        final_model = xgb.XGBRegressor(**best)
        final_model.fit(X_full, y_full, sample_weight=sw_full, verbose=False)
    else:
        best["verbose"] = -1
        best["objective"] = "huber"
        final_model = lgb.LGBMRegressor(**best)
        final_model.fit(X_full, y_full, sample_weight=sw_full)

    return final_model, best, study.best_value


def multi_seed_tune(shifted_df, feature_cols, model_type, n_trials, seeds,
                    sample_weight_cfg=None, cv_folds=None, val_df=None,
                    search_space=None):
    """Run Optuna tuning across multiple seeds, return model from the median-Spearman seed.

    Args:
        shifted_df: Training data (all train years).
        feature_cols: Feature column names.
        model_type: "xgboost" or "lightgbm".
        n_trials: Optuna trials per seed.
        seeds: List of random seeds to try.
        sample_weight_cfg: COVID config dict, or None.
        cv_folds: TSCV fold definitions.
        val_df: Held-out validation DataFrame for seed comparison. Must not overlap with
                training data to avoid leakage.
        search_space: Optional dict overriding default param ranges.

    Returns:
        (final_model, best_params, best_value, seed_results) where seed_results is a list
        of dicts with per-seed metrics.
    """
    if val_df is None:
        raise ValueError("val_df is required for seed comparison (must be held-out data)")

    seed_results = []
    for seed in seeds:
        model, params, best_val = tune_with_optuna(
            shifted_df, feature_cols, model_type, n_trials, seed,
            sample_weight_cfg, cv_folds, search_space
        )
        # Evaluate on held-out validation set
        preds = model.predict(val_df[feature_cols])
        y_val = val_df["target_ESPN_Pts"]
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        spearman, _ = spearmanr(y_val, preds)

        seed_results.append({
            "seed": seed,
            "val_spearman": float(spearman),
            "val_mae": float(mae),
            "val_rmse": float(rmse),
            "best_cv_rmse": float(best_val),
            "params": params,
            "model": model,
        })
        print(f"    Seed {seed}: Spearman={spearman:.3f}, MAE={mae:.1f}")

    # Pick seed with median Spearman
    sorted_by_spearman = sorted(seed_results, key=lambda x: x["val_spearman"])
    median_idx = len(sorted_by_spearman) // 2
    selected = sorted_by_spearman[median_idx]

    print(f"    Selected seed {selected['seed']} (median Spearman={selected['val_spearman']:.3f})")

    return selected["model"], selected["params"], selected["best_cv_rmse"], seed_results


def run_shap_analysis(model, X, feature_cols, fig_dir, prefix=""):
    """Generate SHAP summary and beeswarm plots."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_cols])

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X[feature_cols], plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(fig_dir / f"{prefix}shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close("all")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X[feature_cols], show=False, max_display=20)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(fig_dir / f"{prefix}shap_beeswarm.png", dpi=100, bbox_inches="tight")
    plt.close("all")

    # Feature importance table
    importance = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    return importance


def generate_pareto_plot(importance_df, fig_dir, prefix=""):
    """Pareto chart: SHAP importance bars (sorted) + cumulative % line with 80% threshold."""
    df = importance_df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    total = df["shap_importance"].sum()
    df["cumulative_pct"] = df["shap_importance"].cumsum() / total * 100

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bars
    x = range(len(df))
    ax1.bar(x, df["shap_importance"], color="steelblue", alpha=0.8)
    ax1.set_ylabel("Mean |SHAP value|", color="steelblue")
    ax1.set_xlabel("Feature")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["feature"], rotation=45, ha="right", fontsize=8)

    # Cumulative % line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, df["cumulative_pct"], color="red", marker="o", markersize=4, linewidth=2)
    ax2.set_ylabel("Cumulative %", color="red")
    ax2.set_ylim(0, 105)

    # 80% threshold line
    ax2.axhline(80, color="red", linestyle="--", alpha=0.5)
    n_80 = int((df["cumulative_pct"] <= 80).sum()) + 1
    ax2.annotate(f"80% at {n_80} features", xy=(n_80 - 1, 80),
                 xytext=(n_80 + 2, 85), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="red"),
                 color="red")

    ax1.set_title(f"{prefix.rstrip('_').title()} SHAP Feature Importance — Pareto Chart")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}pareto.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto chart saved: {prefix}pareto.png ({n_80} features reach 80%)")


class EnsembleModel:
    """Weighted blend of XGBoost and LightGBM predictions with optional intercept."""

    def __init__(self, xgb_model, lgb_model, weights, intercept=0.0):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.weights = weights  # [w_xgb, w_lgb]
        self.intercept = intercept

    def predict(self, X, clip_range=None):
        p1 = self.xgb_model.predict(X)
        p2 = self.lgb_model.predict(X)
        preds = self.weights[0] * p1 + self.weights[1] * p2 + self.intercept
        if clip_range:
            preds = np.clip(preds, *clip_range)
        return preds


def _compute_sample_weights(train_df, cfg):
    """Build sample weight array: downweight rows where feature year is COVID."""
    covid_cfg = cfg.get("covid", {})
    if covid_cfg and "covid_feature_year" in train_df.columns:
        w = np.where(train_df["covid_feature_year"] == 1, covid_cfg["sample_weight"], 1.0)
        return w
    return None


def _serialize_params(params):
    """Convert numpy types to Python natives for JSON serialization."""
    return {k: float(v) if isinstance(v, (np.floating, float))
            else int(v) if isinstance(v, (np.integer, int)) else v
            for k, v in params.items()}


def _serialize_metrics(metrics):
    """Convert metric dict values for JSON."""
    return {k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()}


# Features to drop per pitcher role (constant/irrelevant within role)
_SP_DROP_FEATURES = {"is_starter", "is_closer", "p_gs", "p_gf"}
_RP_DROP_FEATURES = {"is_starter", "p_gs"}


def _train_role_model(train, val, test, feature_cols, role_label, cfg, clip_range):
    """Train a single ensemble model for a given data subset. Returns (ensemble, meta, test)."""
    random_state = cfg["model"]["random_state"]
    n_trials = cfg["model"]["optuna_trials"]
    seeds = cfg["model"].get("optuna_seeds", [random_state])
    fig_dir = ROOT / cfg["paths"]["figures"]["feature_importance"]
    covid_cfg = cfg.get("covid", {}) or None
    # Use per-role CV folds if available (e.g. cv_folds_batter)
    base_type = "pitcher" if role_label.startswith("pitcher") else role_label
    folds_key = f"cv_folds_{base_type}"
    cv_folds = cfg["seasons"].get(folds_key, cfg["seasons"].get("cv_folds"))
    trials_each = n_trials // 2
    n_folds = len(cv_folds) if cv_folds else 1

    # Tighter search space for RP to reduce overfitting
    rp_search_space = None
    if role_label == "pitcher_rp":
        rp_search_space = {
            "n_estimators": (100, 500),
            "max_depth": (2, 5),
            "min_child_weight": (5, 30),
            "subsample": (0.5, 0.9),
            "colsample_bytree": (0.4, 0.8),
        }

    print(f"\n--- {role_label.upper()} ---")
    print(f"Features: {len(feature_cols)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Baseline
    print("Baseline models:")
    best_model, model_type, best_metrics, other_metrics = train_baseline(
        train, val, feature_cols, random_state
    )
    print(f"  Winner: {model_type}")

    # Multi-seed Optuna tuning with TSCV
    print(f"Multi-seed Optuna ({trials_each} trials × {len(seeds)} seeds, {n_folds}-fold TSCV)...")
    print(f"  XGBoost tuning...")
    xgb_model, xgb_params, _, xgb_seed_results = multi_seed_tune(
        train, feature_cols, "xgboost", trials_each, seeds, covid_cfg, cv_folds, val_df=val,
        search_space=rp_search_space
    )
    xgb_val = evaluate_model(xgb_model, val[feature_cols], val["target_ESPN_Pts"], "xgb_tuned")
    print(f"  XGBoost tuned — MAE: {xgb_val['MAE']:.1f}, Spearman: {xgb_val['Spearman']:.3f}")

    print(f"  LightGBM tuning...")
    lgb_model, lgb_params, _, lgb_seed_results = multi_seed_tune(
        train, feature_cols, "lightgbm", trials_each, seeds, covid_cfg, cv_folds, val_df=val,
        search_space=rp_search_space
    )
    lgb_val = evaluate_model(lgb_model, val[feature_cols], val["target_ESPN_Pts"], "lgb_tuned")
    print(f"  LightGBM tuned — MAE: {lgb_val['MAE']:.1f}, Spearman: {lgb_val['Spearman']:.3f}")

    # Softmax blend weights on validation Spearman (data-driven, not hardcoded)
    xgb_sp = max(xgb_val["Spearman"], 1e-6)
    lgb_sp = max(lgb_val["Spearman"], 1e-6)
    denom = np.exp(xgb_sp) + np.exp(lgb_sp)
    w_xgb = np.exp(xgb_sp) / denom
    w_lgb = np.exp(lgb_sp) / denom
    blend_weights = np.array([w_xgb, w_lgb])
    blend_intercept = 0.0
    print(f"  Blend — XGB: {w_xgb:.3f}, LGB: {w_lgb:.3f} (softmax on val Spearman)")

    ensemble = EnsembleModel(xgb_model, lgb_model, blend_weights, blend_intercept)

    # Test evaluation
    xgb_test = evaluate_model(xgb_model, test[feature_cols], test["target_ESPN_Pts"], "xgb_test", clip_range)
    lgb_test = evaluate_model(lgb_model, test[feature_cols], test["target_ESPN_Pts"], "lgb_test", clip_range)
    ens_test = evaluate_model(ensemble, test[feature_cols], test["target_ESPN_Pts"], "ensemble_test", clip_range)

    print(f"Test — XGB MAE: {xgb_test['MAE']:.1f}, LGB MAE: {lgb_test['MAE']:.1f}, "
          f"Ensemble MAE: {ens_test['MAE']:.1f}, Spearman: {ens_test['Spearman']:.3f}")

    # Top-N precision
    test_preds = ensemble.predict(test[feature_cols], clip_range=clip_range)
    y_test = test["target_ESPN_Pts"]
    for n in [25, 50, 100]:
        if len(test) >= n:
            prec = topn_precision(y_test, test_preds, n)
            print(f"  Top-{n} precision: {prec:.1%}")

    # Overfit check
    train_metrics = evaluate_model(ensemble, train[feature_cols], train["target_ESPN_Pts"], "train", clip_range)
    overfit_ratio = ens_test["MAE"] / train_metrics["MAE"] if train_metrics["MAE"] > 0 else 0
    print(f"  Train MAE: {train_metrics['MAE']:.1f}, Overfit ratio: {overfit_ratio:.2f}")

    # SHAP
    print("SHAP analysis...")
    importance = run_shap_analysis(xgb_model, test, feature_cols, fig_dir, prefix=f"{role_label}_")
    print("Top 10 features:")
    print(importance.head(10).to_string(index=False))
    generate_pareto_plot(importance, fig_dir, prefix=f"{role_label}_")

    # Summarize seed results for metadata (exclude model objects)
    def _summarize_seeds(seed_results):
        return [
            {"seed": s["seed"], "val_spearman": s["val_spearman"],
             "val_mae": s["val_mae"], "val_rmse": s["val_rmse"]}
            for s in seed_results
        ]

    xgb_spearman_range = max(s["val_spearman"] for s in xgb_seed_results) - min(s["val_spearman"] for s in xgb_seed_results)
    lgb_spearman_range = max(s["val_spearman"] for s in lgb_seed_results) - min(s["val_spearman"] for s in lgb_seed_results)
    print(f"  Seed variance — XGB Spearman range: {xgb_spearman_range:.3f}, LGB Spearman range: {lgb_spearman_range:.3f}")

    meta = {
        "model_type": "ensemble",
        "blend_weights": blend_weights.tolist(),
        "blend_intercept": float(blend_intercept),
        "clip_range": list(clip_range) if clip_range else None,
        "xgb_params": _serialize_params(xgb_params),
        "lgb_params": _serialize_params(lgb_params),
        "feature_cols": feature_cols,
        "test_metrics": _serialize_metrics(ens_test),
        "xgb_test_metrics": _serialize_metrics(xgb_test),
        "lgb_test_metrics": _serialize_metrics(lgb_test),
        "val_metrics": {
            "xgb": _serialize_metrics(xgb_val),
            "lgb": _serialize_metrics(lgb_val),
        },
        "seed_results": {
            "xgb": _summarize_seeds(xgb_seed_results),
            "lgb": _summarize_seeds(lgb_seed_results),
            "selection_method": "median_spearman",
        },
    }

    return ensemble, meta, xgb_model, lgb_model


def _save_models(xgb_model, lgb_model, meta, model_key, cfg):
    """Save XGBoost, LightGBM, and metadata to disk."""
    model_path = ROOT / cfg["paths"]["outputs"][model_key]
    xgb_model.save_model(str(model_path))

    lgb_key = model_key + "_lgb"
    lgb_path = ROOT / cfg["paths"]["outputs"][lgb_key]
    lgb_model.booster_.save_model(str(lgb_path))

    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path


def train_pipeline(player_type, cfg):
    """Full training pipeline: tune both XGBoost & LightGBM, blend into ensemble.

    For pitchers with split_pitcher_roles=True, trains separate SP and RP models.
    """
    random_state = cfg["model"]["random_state"]
    bounds = cfg["model"].get("prediction_bounds", {})

    # Load features
    if player_type == "batter":
        features = pd.read_csv(ROOT / cfg["paths"]["processed"]["batter_features"])
    else:
        features = pd.read_csv(ROOT / cfg["paths"]["processed"]["pitcher_features"])

    # Build shifted dataset (with COVID handling)
    shifted = build_shifted_dataset(features, cfg=cfg)
    all_feature_cols = get_feature_cols(shifted)

    print(f"\n{'='*50}")
    print(f"{player_type.upper()} MODEL")
    print(f"{'='*50}")
    print(f"Features: {len(all_feature_cols)}")
    print(f"Total samples: {len(shifted)}")

    # Time split (uses per-role train years if configured)
    train, val, test = time_split(shifted, cfg, player_type=player_type)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Check if we should split pitcher roles
    split_roles = (player_type == "pitcher" and cfg["model"].get("split_pitcher_roles", False)
                   and "is_starter" in shifted.columns)

    if split_roles:
        print("\n** Training separate SP and RP models **")
        results = {}
        combined_test = []

        for role, mask_val, drop_feats, role_key in [
            ("pitcher_sp", 1, _SP_DROP_FEATURES, "pitcher_sp_model"),
            ("pitcher_rp", 0, _RP_DROP_FEATURES, "pitcher_rp_model"),
        ]:
            role_train = train[train["is_starter"] == mask_val]
            role_val = val[val["is_starter"] == mask_val]
            role_test = test[test["is_starter"] == mask_val]

            # Drop constant/irrelevant features for this role
            role_features = [c for c in all_feature_cols if c not in drop_feats]
            clip_range = tuple(bounds[role]) if role in bounds else None

            ensemble, meta, xgb_m, lgb_m = _train_role_model(
                role_train, role_val, role_test, role_features, role, cfg, clip_range
            )
            _save_models(xgb_m, lgb_m, meta, role_key, cfg)
            results[role] = {"ensemble": ensemble, "meta": meta}
            combined_test.append(role_test)

        # Combined pitcher test metrics
        test_combined = pd.concat(combined_test)
        print(f"\nCombined pitcher test: {len(test_combined)} players")

        # Return the SP model as primary (for backward compat), but include both
        primary_meta = results["pitcher_sp"]["meta"]
        primary_meta["split_roles"] = True
        primary_meta["roles"] = list(results.keys())
        return results["pitcher_sp"]["ensemble"], primary_meta, test_combined, shifted

    else:
        # Single model (batters, or pitchers without split)
        clip_range = tuple(bounds[player_type]) if player_type in bounds else None
        ensemble, meta, xgb_model, lgb_model = _train_role_model(
            train, val, test, all_feature_cols, player_type, cfg, clip_range
        )
        _save_models(xgb_model, lgb_model, meta, f"{player_type}_model", cfg)

        return ensemble, meta, test, shifted
