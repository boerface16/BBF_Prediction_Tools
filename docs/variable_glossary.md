# Variable Glossary

Complete reference for all columns used across the fantasy baseball pipeline. Each variable lists its definition, formula (where applicable), source file/function, and whether it applies to batters (B), pitchers (P), or both.

---

## Player Identity & Metadata

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `id` | Retrosheet player ID (primary key) | — | `data_builder.aggregate_batting/pitching` (B/P) |
| `year` | Season year, extracted from game date | `str(date)[:4]` | `data_builder.extract_year` (B/P) |
| `last` | Player last name | — | `data_builder.merge_player_info` (B/P) |
| `first` | Player first name | — | `data_builder.merge_player_info` (B/P) |
| `team` | Primary team for the season (most games) | — | `data_builder.merge_player_info` (B/P) |
| `primary_pos` | Primary position (C, 1B, 2B, 3B, SS, LF, CF, RF, DH, P, UTIL) | Position with most games played | `data_builder.get_primary_position` (B/P) |
| `G` | Games played (count of unique game IDs) | `nunique(gid)` | `data_builder.aggregate_batting/pitching` (B/P) |
| `LOW_SAMPLE` | Flag: player below minimum PA/IP threshold | See thresholds in `config.yaml` | `data_builder.apply_thresholds` (B/P) |

---

## Raw Counting Stats — Batting

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `b_pa` | Plate appearances | `sum(b_pa)` | `data_builder.aggregate_batting` (B) |
| `b_ab` | At-bats | `sum(b_ab)` | `data_builder.aggregate_batting` (B) |
| `b_r` | Runs scored | `sum(b_r)` | `data_builder.aggregate_batting` (B) |
| `b_h` | Hits | `sum(b_h)` | `data_builder.aggregate_batting` (B) |
| `b_d` | Doubles | `sum(b_d)` | `data_builder.aggregate_batting` (B) |
| `b_t` | Triples | `sum(b_t)` | `data_builder.aggregate_batting` (B) |
| `b_hr` | Home runs | `sum(b_hr)` | `data_builder.aggregate_batting` (B) |
| `b_rbi` | Runs batted in | `sum(b_rbi)` | `data_builder.aggregate_batting` (B) |
| `b_sh` | Sacrifice hits (bunts) | `sum(b_sh)` | `data_builder.aggregate_batting` (B) |
| `b_sf` | Sacrifice flies | `sum(b_sf)` | `data_builder.aggregate_batting` (B) |
| `b_hbp` | Hit by pitch | `sum(b_hbp)` | `data_builder.aggregate_batting` (B) |
| `b_w` | Walks (BB) | `sum(b_w)` | `data_builder.aggregate_batting` (B) |
| `b_iw` | Intentional walks | `sum(b_iw)` | `data_builder.aggregate_batting` (B) |
| `b_k` | Strikeouts | `sum(b_k)` | `data_builder.aggregate_batting` (B) |
| `b_sb` | Stolen bases | `sum(b_sb)` | `data_builder.aggregate_batting` (B) |
| `b_cs` | Caught stealing | `sum(b_cs)` | `data_builder.aggregate_batting` (B) |
| `b_gdp` | Grounded into double play | `sum(b_gdp)` | `data_builder.aggregate_batting` (B) |
| `b_1b` | Singles (derived) | `b_h - b_d - b_t - b_hr` | `advanced_stats.calculate_batter_rates` (B) |
| `TB` | Total bases | `b_1b + 2*b_d + 3*b_t + 4*b_hr` | `advanced_stats.calculate_batter_rates` (B) |

---

## Raw Counting Stats — Pitching

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `p_ipouts` | Outs recorded (IP × 3) | `sum(p_ipouts)` | `data_builder.aggregate_pitching` (P) |
| `p_bfp` | Batters faced (BF) | `sum(p_bfp)` | `data_builder.aggregate_pitching` (P) |
| `p_h` | Hits allowed | `sum(p_h)` | `data_builder.aggregate_pitching` (P) |
| `p_d` | Doubles allowed | `sum(p_d)` | `data_builder.aggregate_pitching` (P) |
| `p_t` | Triples allowed | `sum(p_t)` | `data_builder.aggregate_pitching` (P) |
| `p_hr` | Home runs allowed | `sum(p_hr)` | `data_builder.aggregate_pitching` (P) |
| `p_r` | Runs allowed | `sum(p_r)` | `data_builder.aggregate_pitching` (P) |
| `p_er` | Earned runs | `sum(p_er)` | `data_builder.aggregate_pitching` (P) |
| `p_w` | Walks allowed (BB) | `sum(p_w)` | `data_builder.aggregate_pitching` (P) |
| `p_iw` | Intentional walks issued | `sum(p_iw)` | `data_builder.aggregate_pitching` (P) |
| `p_k` | Strikeouts | `sum(p_k)` | `data_builder.aggregate_pitching` (P) |
| `p_hbp` | Hit batters | `sum(p_hbp)` | `data_builder.aggregate_pitching` (P) |
| `p_gs` | Games started | `sum(p_gs)` | `data_builder.aggregate_pitching` (P) |
| `p_gf` | Games finished | `sum(p_gf)` | `data_builder.aggregate_pitching` (P) |
| `W` | Wins | `sum(wp flag)` | `data_builder.aggregate_pitching` (P) |
| `L` | Losses | `sum(lp flag)` | `data_builder.aggregate_pitching` (P) |
| `SV` | Saves | `sum(save flag)` | `data_builder.aggregate_pitching` (P) |
| `HD` | Holds (not available in Retrosheet; set to 0) | `0` | `data_builder.aggregate_pitching` (P) |
| `IP` | Innings pitched | `p_ipouts / 3` | `advanced_stats.calculate_pitcher_rates` (P) |

---

## Basic Rate Stats — Batting

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `AVG` | Batting average | `b_h / b_ab` | `advanced_stats.calculate_batter_rates` (B) |
| `OBP` | On-base percentage | `(b_h + b_w + b_hbp) / b_pa` | `advanced_stats.calculate_batter_rates` (B) |
| `SLG` | Slugging percentage | `TB / b_ab` | `advanced_stats.calculate_batter_rates` (B) |
| `OPS` | On-base plus slugging | `OBP + SLG` | `advanced_stats.calculate_batter_rates` (B) |
| `ISO` | Isolated power | `SLG - AVG` | `advanced_stats.calculate_batter_rates` (B) |
| `BABIP` | Batting average on balls in play | `(b_h - b_hr) / (b_ab - b_k - b_hr + b_sf)` | `advanced_stats.calculate_batter_rates` (B) |
| `K_pct` | Strikeout rate | `b_k / b_pa` | `advanced_stats.calculate_batter_rates` (B) |
| `BB_pct` | Walk rate | `b_w / b_pa` | `advanced_stats.calculate_batter_rates` (B) |
| `BB_K_pct` | Walk rate minus strikeout rate | `BB_pct - K_pct` | `advanced_stats.calculate_batter_rates` (B) |
| `HR_PA` | Home run rate per plate appearance | `b_hr / b_pa` | `advanced_stats.calculate_batter_rates` (B) |
| `SB_pct` | Stolen base success rate | `b_sb / (b_sb + b_cs)` | `advanced_stats.calculate_batter_rates` (B) |
| `SB_rate` | Stolen bases per plate appearance | `b_sb / b_pa` | `feature_builder.build_batter_features` (B) |

---

## Basic Rate Stats — Pitching

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `ERA` | Earned run average | `(p_er * 9) / IP` | `advanced_stats.calculate_pitcher_rates` (P) |
| `WHIP` | Walks + hits per inning pitched | `(p_h + p_w) / IP` | `advanced_stats.calculate_pitcher_rates` (P) |
| `K9` | Strikeouts per 9 innings | `(p_k * 9) / IP` | `advanced_stats.calculate_pitcher_rates` (P) |
| `BB9` | Walks per 9 innings | `(p_w * 9) / IP` | `advanced_stats.calculate_pitcher_rates` (P) |
| `HR9` | Home runs per 9 innings | `(p_hr * 9) / IP` | `advanced_stats.calculate_pitcher_rates` (P) |
| `K_pct` | Strikeout rate (pitcher) | `p_k / p_bfp` | `advanced_stats.calculate_pitcher_rates` (P) |
| `BB_pct` | Walk rate (pitcher) | `p_w / p_bfp` | `advanced_stats.calculate_pitcher_rates` (P) |
| `K_BB_pct` | K% minus BB% (pitcher) | `K_pct - BB_pct` | `advanced_stats.calculate_pitcher_rates` (P) |
| `BABIP_allowed` | Pitcher BABIP (balls in play) | `(p_h - p_hr) / (p_bfp - p_k - p_hr - p_w - p_hbp)` | `advanced_stats.calculate_pitcher_rates` (P) |
| `LOB_pct` | Left on base percentage | `(p_h + p_w + p_hbp - p_r) / (p_h + p_w + p_hbp - 1.4*p_hr)` | `advanced_stats.calculate_pitcher_rates` (P) |
| `is_starter` | Starter flag (1 if ≥ 50% of appearances are starts) | `(p_gs >= G * 0.5).astype(int)` | `advanced_stats.calculate_pitcher_rates` (P) |
| `is_closer` | Closer flag | `(is_starter == 0) & (SV >= 10)` | `feature_builder.build_pitcher_features` (P) |

---

## Park Factors & League Averages

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `PF` | Park factor (5-year rolling average) | `rpg_at_park / lg_rpg` (rolling 5-year mean) | `advanced_stats.calculate_park_factors` (B/P) |
| `lg_OBP` | League average OBP for the season | `(lg_H + lg_BB + lg_HBP) / lg_PA` | `advanced_stats.calculate_league_averages` (B) |
| `lg_SLG` | League average SLG for the season | `lg_TB / lg_AB` | `advanced_stats.calculate_league_averages` (B) |
| `lg_ERA` | League average ERA for the season | `(lg_ER * 9) / lg_IP` | `advanced_stats.calculate_league_averages` (P) |
| `FIP_constant` | FIP constant (season-level) | `lg_ERA - (13*lg_HR + 3*(lg_BB+lg_HBP) - 2*lg_K) / lg_IP` | `advanced_stats.calculate_league_averages` (P) |

---

## Advanced Batting Stats

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `OPS_plus` | OPS+ (park- and league-adjusted OPS; 100 = league average) | `100 * (OBP/lg_OBP + SLG/lg_SLG - 1) / PF` | `advanced_stats.apply_ops_plus` (B) |
| `wOBA` | Weighted on-base average (RE24-derived weights) | `(wBB*(BB-IBB) + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR) / PA` | `re24_engine.calculate_woba_wrc` (B) |
| `wRAA` | Weighted runs above average | `((wOBA - lg_wOBA) / wOBA_scale) * PA` | `re24_engine.calculate_woba_wrc` (B) |
| `wRC` | Weighted runs created | `wRAA + (lg_runs_per_out * 3/9) * PA` | `re24_engine.calculate_woba_wrc` (B) |
| `wRC_plus` | Weighted runs created plus (park-adjusted; 100 = league avg) | `((wRAA/PA + runs_pa) / runs_pa) * 100 / PF` | `re24_engine.calculate_woba_wrc` (B) |

---

## Advanced Pitching Stats

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `FIP` | Fielding independent pitching | `(13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_constant` | `advanced_stats.apply_pitching_indices` (P) |
| `ERA_minus` | ERA- (park/league adjusted; 100 = league avg, lower is better) | `100 * ERA * (2 - PF) / lg_ERA` | `advanced_stats.apply_pitching_indices` (P) |
| `FIP_minus` | FIP- (park/league adjusted; 100 = league avg, lower is better) | `100 * FIP * (2 - PF) / lg_ERA` | `advanced_stats.apply_pitching_indices` (P) |
| `xFIP` | Expected FIP (replaces HR with expected HR from FB rate) | `(13 * (FB_count * lg_HR/FB) + 3*(BB+HBP) - 2*K) / IP + FIP_constant` | `feature_builder.build_pitcher_features` (P) |
| `xFIP_minus` | xFIP- (park/league adjusted) | `100 * xFIP * (2 - PF) / lg_ERA` | `feature_builder.build_pitcher_features` (P) |

---

## Run Expectancy (RE24) & Linear Weights

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `RE24` | Run expectancy over 24 base-out states (batter: total run value; pitcher: negated) | `runs_scored + RE_end - RE_start` (summed per season) | `re24_engine.calculate_re24 / aggregate_re24_players` (B/P) |
| `RE24_against` | Raw RE24 against (pitcher, before negation) | `sum(RE24)` per pitcher-season | `re24_engine.aggregate_re24_players` (P) |
| `RE24_efficiency` | RE24 per unit of available run expectancy | `RE24 / Available_Runs` | `re24_engine.aggregate_re24_players` (B) |
| `Available_Runs` | Sum of RE_start across all PAs (run opportunity) | `sum(RE_start)` | `re24_engine.aggregate_re24_players` (B) |
| `wOBA_scale` | Scaling factor to align raw linear weights to OBP scale | `league_OBP / league_wOBA_raw` | `re24_engine.derive_woba_weights` (B) |
| `runs_per_out` | Average run cost of an out (negated rv_out) | `-rv_out` | `re24_engine.derive_woba_weights` (B) |

---

## Batted Ball Rates

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `GB_pct` | Ground ball rate | `ground_balls / BIP` | `re24_engine.calculate_batted_ball_rates` (B/P) |
| `FB_pct` | Fly ball rate | `fly_balls / BIP` | `re24_engine.calculate_batted_ball_rates` (B/P) |
| `LD_pct` | Line drive rate | `line_drives / BIP` | `re24_engine.calculate_batted_ball_rates` (B) |
| `HR_FB_pct` | Home run to fly ball rate | `HR / fly_balls` | `re24_engine.calculate_batted_ball_rates` (B/P) |
| `BIP` | Balls in play count | `count of PAs with ground + fly + line > 0` | `re24_engine.calculate_batted_ball_rates` (B/P) |

---

## Statcast Metrics — Batting

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `brl_percent` | Barrel rate (% of batted ball events that are barrels) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `ev95percent` | Hard hit rate (% of batted balls ≥ 95 mph) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `avg_hit_speed` | Average exit velocity (mph) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `anglesweetspotpercent` | Sweet spot rate (% of batted balls in 8-32 degree launch angle) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `est_ba` | Expected batting average (xBA, from Statcast) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `est_slg` | Expected slugging (xSLG, from Statcast) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `est_woba` | Expected wOBA (xwOBA, from Statcast) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B/P) |
| `sprint_speed` | Sprint speed (ft/s, from Statcast) | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `est_ba_minus_ba_diff` | xBA minus BA differential | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |
| `est_slg_minus_slg_diff` | xSLG minus SLG differential | — (from Statcast) | `feature_builder.merge_statcast_batters` (B) |

---

## Statcast Metrics — Pitching

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `brl_percent_allowed` | Barrel rate allowed (renamed from `brl_percent`) | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `ev95percent_against` | Hard hit rate against (renamed from `ev95percent`) | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `avg_ev_against` | Average exit velocity against (renamed from `avg_hit_speed`) | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `era` | Statcast ERA (from Statcast expected stats) | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `xera` | Expected ERA (xERA, from Statcast) | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `era_minus_xera_diff` | ERA minus xERA differential | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `weighted_whiff_pct` | Usage-weighted whiff% across all pitch types (SwStr% proxy) | `sum(whiff_percent * pitch_usage / 100)` | `feature_builder.merge_statcast_pitchers` (P) |
| `avg_hard_hit_pct` | Average hard hit % across pitch types | `mean(hard_hit_percent)` | `feature_builder.merge_statcast_pitchers` (P) |
| `swstr_pct` | Swinging strike rate — strongest predictor of K rate | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `csw_pct` | Called strikes + swinging strikes % — broader stuff metric | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `zone_pct` | Rate at which pitcher throws pitches inside the strike zone | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `chase_pct` | Rate at which batters swing at pitches outside the zone | — (from Statcast) | `feature_builder.merge_statcast_pitchers` (P) |
| `ff_avg_speed` | Four-seam fastball average velocity (mph) | — (from Statcast pitch arsenal) | `feature_builder.merge_statcast_pitchers` (P) |
| `ff_velo_delta` | Year-over-year change in fastball velocity | `groupby("id")["ff_avg_speed"].diff()` | `feature_builder.build_pitcher_features` (P) |

---

## Year-over-Year Delta Features

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `K_pct_delta` | Year-over-year change in K% | `groupby("id")["K_pct"].diff()` | `feature_builder.build_batter_features` (B) |
| `BB_pct_delta` | Year-over-year change in BB% | `groupby("id")["BB_pct"].diff()` | `feature_builder.build_batter_features` (B) |
| `ISO_delta` | Year-over-year change in ISO | `groupby("id")["ISO"].diff()` | `feature_builder.build_batter_features` (B) |
| `BABIP_delta` | Year-over-year change in BABIP | `groupby("id")["BABIP"].diff()` | `feature_builder.build_batter_features` (B) |
| `K9_delta` | Year-over-year change in K/9 | `groupby("id")["K9"].diff()` | `feature_builder.build_pitcher_features` (P) |
| `ERA_delta` | Year-over-year change in ERA | `groupby("id")["ERA"].diff()` | `feature_builder.build_pitcher_features` (P) |
| `WHIP_delta` | Year-over-year change in WHIP | `groupby("id")["WHIP"].diff()` | `feature_builder.build_pitcher_features` (P) |
| `BB9_delta` | Year-over-year change in BB/9 | `groupby("id")["BB9"].diff()` | `feature_builder.build_pitcher_features` (P) |

---

## Regression & Gap Features

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `career_BABIP` | Career expanding mean BABIP (lagged 1 year, min 2 seasons) | `expanding().mean().shift(1)` on BABIP | `feature_builder.add_career_rolling` (B) |
| `career_HR_FB_pct` | Career expanding mean HR/FB% (lagged 1 year) | `expanding().mean().shift(1)` on HR_FB_pct | `feature_builder.add_career_rolling` (B) |
| `career_BABIP_allowed` | Career expanding mean BABIP allowed (pitcher) | `expanding().mean().shift(1)` on BABIP_allowed | `feature_builder.add_career_rolling` (P) |
| `career_ERA` | Career expanding mean ERA (pitcher) | `expanding().mean().shift(1)` on ERA | `feature_builder.add_career_rolling` (P) |
| `BABIP_gap` | Current BABIP vs career average (regression signal) | Batter: `BABIP - career_BABIP`; Pitcher: `BABIP_allowed - career_BABIP_allowed` | `feature_builder.build_batter/pitcher_features` (B/P) |
| `HR_FB_gap` | Current HR/FB% vs career average | `HR_FB_pct - career_HR_FB_pct` | `feature_builder.build_batter_features` (B) |
| `xBA_AVG_gap` | xBA minus actual AVG (luck/skill gap) | `est_ba - AVG` | `feature_builder.build_batter_features` (B) |
| `xSLG_SLG_gap` | xSLG minus actual SLG | `est_slg - SLG` | `feature_builder.build_batter_features` (B) |
| `ERA_FIP_gap` | ERA minus FIP (sequencing luck indicator) | `ERA - FIP` | `feature_builder.build_pitcher_features` (P) |
| `LOB_pct_dev` | LOB% deviation from league average (.720) | `LOB_pct - 0.720` | `feature_builder.build_pitcher_features` (P) |

---

## Team Context

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `team_rpg` | Team runs per game for the season | `team_total_runs / team_games` | `feature_builder.build_batter_features` (B) |
| `team_win_pct` | Team win percentage for the season | `team_wins / team_games` | `feature_builder.build_pitcher_features` (P) |
| `experience` | Number of prior qualifying seasons (0-indexed) | `cumcount()` per player | `feature_builder.build_batter/pitcher_features` (B/P) |

---

## ESPN Fantasy Points (Target Variable)

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `ESPN_Pts` | ESPN H2H fantasy points (current season) | Batter: `TB + R + RBI + BB + SB - SO`; Pitcher: `IP*3 + K + W*5 + SV*5 + HD*2 - H - ER*2 - BB - L*2` | `espn_points_mapper.calculate_batter/pitcher_points` (B/P) |
| `target_ESPN_Pts` | Next-season ESPN points (ML target, year N+1) | Shifted from next year's `ESPN_Pts`; 2020 targets pro-rated to 162 games | `feature_builder.build_shifted_dataset` (B/P) |

---

## Shifted Dataset & Model Training

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `covid_feature_year` | Flag: feature year is 2020 (noisy features) | `(year == 2020).astype(int)` | `feature_builder.build_shifted_dataset` (B/P) |

### Excluded from Features (META_COLS)

These columns are present in the shifted dataset but excluded from model input:

| Variable | Reason for Exclusion | Source |
|---|---|---|
| `id` | Player identifier, not a feature | `model_trainer.META_COLS` |
| `year` | Time identifier, not a feature | `model_trainer.META_COLS` |
| `last` | Name, not a feature | `model_trainer.META_COLS` |
| `first` | Name, not a feature | `model_trainer.META_COLS` |
| `team` | Categorical identifier | `model_trainer.META_COLS` |
| `primary_pos` | Categorical identifier | `model_trainer.META_COLS` |
| `target_ESPN_Pts` | The target variable itself | `model_trainer.META_COLS` |

### Excluded from Features (COUNTING_LEAK_COLS)

Raw counting stats excluded to prevent leakage (they compose the ESPN_Pts formula directly):

`b_hr`, `b_r`, `b_rbi`, `b_sb`, `b_k`, `b_w`, `p_k`, `p_h`, `p_er`, `p_w`, `p_hr`, `p_hbp`, `W`, `L`, `SV`, `HD`, `TB`, `b_1b`, `b_d`, `b_t`, `b_ab`, `b_sh`, `b_sf`, `b_hbp`, `b_iw`, `b_cs`, `b_gdp`, `PF`, `lg_OBP`, `lg_SLG`, `lg_ERA`, `FIP_constant`, `LOW_SAMPLE`, `wRC`, `covid_feature_year`

---

## Keeper Evaluator

| Variable | Definition | Formula | Source |
|---|---|---|---|
| `position` | ESPN roster position from keeper candidates file | — | `keeper_evaluator.load_keeper_candidates` |
| `player_type` | Matched as `batter` or `pitcher` | — | `keeper_evaluator.match_candidates` |
| `recent_pts` | Most recent season ESPN_Pts | — | `keeper_evaluator.calculate_trajectory` |
| `prior_pts` | Second most recent season ESPN_Pts | — | `keeper_evaluator.calculate_trajectory` |
| `recent_ppg` | Most recent season points per game | `ESPN_Pts / G` | `keeper_evaluator.calculate_trajectory` |
| `raw_score` | Year-over-year raw ESPN_Pts change | `pts[0] - pts[1]` | `keeper_evaluator.calculate_trajectory` |
| `ppg_score` | Year-over-year PPG change | `ppg[0] - ppg[1]` | `keeper_evaluator.calculate_trajectory` |
| `trajectory_score` | Blended trajectory score | `0.5 * raw_score + 0.5 * ppg_score` | `keeper_evaluator.calculate_trajectory` |
| `ml_projection` | ML model predicted ESPN_Pts for next season | `model.predict(features)` | `keeper_evaluator.get_ml_projection` |
| `traj_norm` | Min-max normalized trajectory score (0–1) | `(trajectory_score - min) / (max - min)` | `keeper_evaluator.build_keeper_rankings` |
| `ml_norm` | Min-max normalized ML projection (0–1) | `(ml_projection - min) / (max - min)` | `keeper_evaluator.build_keeper_rankings` |
| `combined_score` | Weighted blend of normalized trajectory and ML scores | `0.5 * traj_norm + 0.5 * ml_norm` | `keeper_evaluator.build_keeper_rankings` |
| `signal_divergence` | Flag: trajectory and ML signals disagree by > 25 points (normalized) | `abs(traj_norm - ml_norm) * 100 > 25` | `keeper_evaluator.build_keeper_rankings` |
| `recommendation` | Keeper classification | AUTO-KEEP (≥0.80) / KEEP (≥0.60) / BORDERLINE (≥0.40) / CUT (<0.40) | `keeper_evaluator.classify_keeper` |

---

## Configuration Parameters

Key thresholds and weights from `config.yaml`:

| Parameter | Value | Purpose |
|---|---|---|
| `batter_pa` | 200 | Minimum PA for batter qualification |
| `pitcher_ip_outs` | 150 (50 IP) | Minimum outs for pitcher qualification |
| `batter_pa_2020` | 70 | Reduced PA threshold for COVID-shortened 2020 |
| `pitcher_ip_outs_2020` | 50 (~17 IP) | Reduced IP threshold for 2020 |
| `covid.sample_weight` | 0.5 | Downweight for 2020 feature-year training rows |
| `covid.full_season_games` | 162 | Full-season games for 2020 target pro-rating |
| `covid.actual_games` | 60 | Actual 2020 games played |
| `keeper.trajectory_weight` | 0.50 | Weight for trajectory score in combined keeper score |
| `keeper.ml_weight` | 0.50 | Weight for ML projection in combined keeper score |
| `keeper.divergence_threshold` | 25 | Normalized point threshold for divergence flag |
| `model.optuna_trials` | 200 | Optuna hyperparameter search trials (split between XGB and LGB) |
| `model.optuna_seeds` | [42, 123, 456, 789, 1010] | Random seeds for multi-seed tuning (median Spearman selected) |
| `model.random_state` | 42 | Random seed for reproducibility |
| `model.split_pitcher_roles` | true | Train separate SP and RP models |
| `model.prediction_bounds.batter` | [0, 750] | Prediction clipping range for batters |
| `model.prediction_bounds.pitcher_sp` | [0, 650] | Prediction clipping range for SP |
| `model.prediction_bounds.pitcher_rp` | [0, 350] | Prediction clipping range for RP |
