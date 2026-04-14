# Lessons Learned

## 2026-03-18 — pybaseball function names differ from docs

**Problem:** The `statcast_data_collection.md` referenced `batting_stats_bbd` and `pitching_stats_bbd`, which do not exist in pybaseball 2.2.7.

**Fix:** The correct functions are `statcast_batter_expected_stats` and `statcast_pitcher_expected_stats`. Similarly, `statcast_pitcher_pitch_arsenal` returns only pitch speeds (wide format); `statcast_pitcher_arsenal_stats` returns the per-pitch-type data with whiff%, usage, etc.

**Rule:** Always verify pybaseball function names with `dir(pybaseball)` and test column outputs on a single year before building loops.

## 2026-03-18 — pitching.csv wp/lp/save are boolean flags, not pitcher IDs

**Problem:** Plan assumed `wp`, `lp`, `save` columns contain pitcher IDs that need matching. In reality, they are 1.0/NaN boolean flags indicating the pitcher got the decision. Separate `win`/`loss` columns exist but indicate team game outcome (0/1 for every pitcher in the game).

**Fix:** Use `df["W"] = df["wp"].fillna(0).astype(int)` instead of ID matching. The `win`/`loss` columns are team-level and should NOT be used for pitcher W/L.

**Rule:** Always inspect actual column values (unique, value_counts) before coding derivations based on assumed semantics.

## 2026-03-18 — Arsenal stats unavailable before 2017

**Problem:** `statcast_pitcher_arsenal_stats` returns empty/no data for 2015–2016. Baseball Savant didn't track per-pitch-type arsenal stats until 2017.

**Fix:** Added `arsenal_start_year: 2017` to config.yaml and adjusted year coverage validation to expect 2017–2025 for arsenal data.

## 2026-03-18 — Outer join inflates null rates on EV/barrel columns

**Problem:** Outer joining EV/barrel data (minBBE threshold) with expected stats (minPA threshold) creates many rows where one side is null. `brl_percent` and `ev95percent` null rates were 15–45% across all years.

**Fix:** This is structurally expected. Raised `max_null_rate` to 0.50 in config. The important columns (est_ba, est_slg, est_woba, xera) have low null rates.

## 2026-03-19 — Retrosheet batted ball HR_FB_pct is unreliable

**Problem:** `HR_FB_pct` from `calculate_batted_ball_rates()` in `re24_engine.py` was nearly all zeros. Home runs in Retrosheet play-by-play don't have `fly=1` set, so they're excluded from BIP (balls in play). The `hr` column in the plays data doesn't align with fly ball classification.

**Fix:** For xFIP calculation, compute league HR/FB rate using `p_hr` from season counting stats divided by `FB_count` (from `FB_pct * BIP`). This gives correct league HR/FB rates (~0.10-0.18). Don't rely on `HR_FB_pct` from the batted ball CSV for cross-stat calculations.

**Rule:** When a derived rate looks implausible (e.g., HR/FB ≈ 0), check whether the underlying event classification includes the events you expect. Retrosheet batted ball flags and outcome flags don't always overlap.

## 2026-03-20 — Replace methodology-gap derived stats with FanGraphs authoritative values

**Problem:** 12 derived/index stats differed from FanGraphs in EDA validation (notebook 09) due to methodology gaps: Retrosheet batted ball (vs BIS), our RE matrix (vs FG's), our HR/FB-based xFIP (vs FG's). Fixing the methodology to match FG exactly would require replicating their proprietary data sources.

**Fix:** Replace the problematic columns directly with FanGraphs' values in the feature pipeline (`merge_fangraphs_batters()`, `merge_fangraphs_pitchers()`). Join path: `df.id` (Retrosheet) → `id_map_2015_2025.csv` `key_retro` → `key_fangraphs` → FG leaderboard `IDfg` + `Season`. FG percentage columns are already 0–1 decimal (e.g., `GB%=0.305`, `SwStr%=0.151`) — no parsing needed.

**Rule:** When a derived stat has a known methodology gap vs an authoritative source, and the gap can't be closed without proprietary data, replace the column directly rather than trying to match their computation. Use `combine_first()` for FG overwrite so non-matched rows fall back to existing values.

## 2026-03-19 — XGBoost early_stopping_rounds moved to constructor

**Problem:** `XGBModel.fit()` no longer accepts `early_stopping_rounds` as a keyword argument in newer xgboost versions. The parameter was moved to the constructor.

**Fix:** Pass `early_stopping_rounds=50` to `xgb.XGBRegressor(...)` constructor, not to `.fit()`.

**Rule:** When upgrading XGBoost, check for API changes in `fit()` parameters — several were moved to the constructor (early_stopping_rounds, callbacks, etc.).

## 2026-03-20 — Multi-seed Optuna reveals high variance in small training sets

**Problem:** Pitcher SP model Spearman bounced 0.455 → 0.261 → 0.353 across single-seed runs. With 5-seed tournament (multi_seed_tune), XGB Spearman ranged 0.727–0.992 on val and LGB ranged 0.825–0.984 — a 0.265 and 0.159 range respectively. Root cause: only 893 SP training rows.

**Fix:** Implemented `multi_seed_tune()` with median-Spearman selection. This stabilizes the selected model (0.455 test Spearman = matches best prior run) but doesn't reduce the underlying variance. The high variance signals the model architecture itself is near its ceiling for this data size.

**Rule:** When Optuna results are unstable across runs, run 5+ seeds and take the median before concluding a change helped or hurt. For training sets under ~1000 rows, expect Spearman variance > 0.10 across seeds — this is a data limitation, not a tuning problem.

## 2026-03-20 — Stale column references after feature pipeline changes

**Problem:** Notebook 09 Cell 5 referenced `('xFIP', 'xFIP_fg', 'xFIP')` in `pit_feat_rate_pairs`, but `xFIP` was removed from `pitcher_features` during the FG feature swap. Since there's no collision on merge, `fg_pit`'s `xFIP` keeps its name (no `_fg` suffix), so `xFIP_fg` doesn't exist → `KeyError`.

**Fix:** Remove the stale tuple. The `xFIP_minus` vs `xFIP-` comparison in `pit_feat_idx_pairs` still validates xFIP-derived data.

**Rule:** When removing or renaming columns in the feature pipeline, grep all notebooks for references to the old column name AND any `_fg`/`_sc` suffixed variants that relied on merge collisions.

## 2026-04-14 — espn_api missing from requirements.txt caused ModuleNotFoundError in notebook 11

**Problem:** `from espn_api.baseball import League` raised `ModuleNotFoundError` in notebook 11. The package was installed in the environment but was never declared in `requirements.txt`.

**Fix:** `espn_api` was already installed (v0.46.0) — `pip install espn_api` confirmed it. Added `espn_api>=0.46` to `requirements.txt` so the dependency is tracked.

**Rule:** Any third-party import used in a notebook must be listed in `requirements.txt`. When a new data source or API client is introduced, update the requirements file in the same step.
