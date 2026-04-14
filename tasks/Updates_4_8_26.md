# ML Performance Improvement Plan

## Context
The fantasy baseball model has three well-documented problems visible in notebooks 05–09:
1. A **formula bug** in ERA_minus/FIP_minus that causes both to fail EDA validation (these are top-10 SHAP features for the SP model)
2. **Data leakage**: `train_batter` in config includes 2023, which is also the validation year — the model trains on the same rows it evaluates against, causing the 20x overfit ratio (train MAE=3.5, test MAE=72)
3. **50/50 ensemble blend** ignores that LGB outperforms XGB by ~40% on validation — so the ensemble (MAE 72) is worse than both individual models (XGB 71, LGB 71)

---

## Fix 1: Correct ERA_minus and FIP_minus formulas
**File**: `src/advanced_stats.py`, lines 180–181

**Problem**: Uses `(2 - PF)` as park factor adjustment. This approximation is backwards and diverges at extremes (max_abs_diff=182 for ERA_minus). Causes FAIL in EDA validation. Both are top-10 SHAP features for the SP model, so bad signal hurts generalization.

**Fix**: Replace with the standard formula `/ PF`, and use league FIP (not ERA) as the denominator for FIP_minus.

```python
# Current (WRONG):
df["ERA_minus"] = 100 * df["ERA"] * (2 - df["PF"]) / lg_era
df["FIP_minus"] = 100 * df["FIP"] * (2 - df["PF"]) / lg_era

# Fixed:
pf = df["PF"].replace(0, np.nan)

# Compute league FIP for the year and merge it in
# lg_FIP is already calculable from league_avgs components (they're joined in)
# FIP_constant is already on df after the merge in apply_pitching_indices
fip_raw_lg = (13 * lg_agg["p_hr"] + 3 * (lg_agg["p_w"] + lg_agg["p_hbp"]) - 2 * lg_agg["p_k"]) / ip_lg
lg_agg["lg_FIP"] = fip_raw_lg + lg_agg["FIP_constant"]
# (then merge lg_FIP onto df like lg_ERA already is)

df["ERA_minus"] = 100 * (df["ERA"] / lg_era) / pf
df["FIP_minus"] = 100 * (df["FIP"] / df["lg_FIP"]) / pf
```

**Downstream**: After this fix, re-run notebooks 02 (builds pitchers_season.csv) → 05 (rebuilds pitcher_features.csv) → 06 (retrains models) → 07 (regenerates rankings) → 09 (should show PASS for ERA_minus and FIP_minus).

---

## Fix 2: Remove 2023 from train_batter (data leakage fix)
**File**: `config.yaml`, line 141

**Problem**: `train_batter` includes 2023, which is the same year as `validation: [2023]`. In the shifted dataset, year=2023 rows contain 2023 features predicting 2024 targets. `time_split()` puts these rows in BOTH `train` and `val`. Since `tune_with_optuna` retrains the final model on `shifted_df = train`, the model trains on the same rows used for seed selection — direct leakage. This is the root cause of the 20x overfit ratio.

**Fix** (1-line change in config.yaml):
```yaml
# Before:
train_batter: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
# After:
train_batter: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
```

After this: train=2015-2022 (no overlap with val), val=2023 (held out), test=2024. The final model in `tune_with_optuna` receives only `train` as `shifted_df`, so it trains on 2015-2022 only. The val set is then genuinely unseen during seed selection.

Note: `cv_folds_batter` last fold already has `train_end: 2022, val: [2023]` — this is already correct and doesn't need to change.

**Downstream**: Re-run notebook 06 only (features don't change, just the train split).

---

## Fix 3: Data-driven ensemble blend weights
**File**: `src/model_trainer.py`, lines 442–445

**Problem**: Hardcoded 50/50 blend despite LGB being substantially better for batters (val MAE 39.65 vs XGB 55.07). The ensemble ends up worse than both individuals because it over-weights the weaker model.

**Fix**: Replace hardcoded blend with softmax weighting on validation Spearman:
```python
# Replace lines 442-445:
# Instead of: blend_weights = np.array([0.5, 0.5])

xgb_sp = max(xgb_val["Spearman"], 1e-6)
lgb_sp = max(lgb_val["Spearman"], 1e-6)
# Softmax weighting
denom = np.exp(xgb_sp) + np.exp(lgb_sp)
w_xgb = np.exp(xgb_sp) / denom
w_lgb = np.exp(lgb_sp) / denom
blend_weights = np.array([w_xgb, w_lgb])
blend_intercept = 0.0
print(f"  Blend — XGB: {w_xgb:.3f}, LGB: {w_lgb:.3f} (softmax on val Spearman)")
```

Also update `meta` dict to record the computed weights (already done via `blend_weights.tolist()`).

**Downstream**: Re-run notebook 06 only.

---

## Execution order

| Step | Action | Notebooks to re-run |
|------|--------|---------------------|
| 1 | Fix ERA_minus/FIP_minus in `advanced_stats.py` | 02 → 05 → 06 → 07 → 09 |
| 2 | Remove 2023 from train_batter in `config.yaml` | 06 → 07 |
| 3 | Softmax ensemble weights in `model_trainer.py` | 06 → 07 |

All three can be coded at once, then run the full notebook sequence once: 02 → 05 → 06 → 07 → 09.

---

## Critical files
- `src/advanced_stats.py` — Fix ERA_minus/FIP_minus at lines 178–181, also update `calculate_league_averages()` to output `lg_FIP` and `apply_pitching_indices()` to accept it
- `config.yaml` — Change `train_batter` at line 141
- `src/model_trainer.py` — Replace blend weight lines 442–445

---

## Verification
After re-running all notebooks:
- Notebook 09: ERA_minus and FIP_minus should show PASS (mean_abs_diff < 1.0, corr > 0.98)
- Notebook 06 batter overfit ratio: should drop from ~20x toward 2–5x (train MAE will rise from 3.5, test MAE should fall)
- Notebook 06 ensemble MAE: should be ≤ min(XGB MAE, LGB MAE) rather than worse than both
- Notebook 06 batter Spearman on test: expect improvement from 0.588 toward 0.65+
- Notebook 07 rankings: Soto/Ohtani/Judge at top still expected, but projected points will shift
