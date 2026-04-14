# ML Model Input Features

Features that enter the XGBoost/LightGBM ensemble models. Feature lists are defined in `config.yaml` under `features.batter.model` and `features.pitcher.model`. Comment out a feature with `#` to disable it without deleting. Meta and counting columns are included in feature CSVs but excluded from model inputs by `get_feature_cols()` in `src/model_trainer.py`.

## Batter Model — 31 Active Features

### Volume
| Feature | Description | Status |
|---------|-------------|--------|
| `b_pa` | Plate appearances | active |
| `G` | Games played | active |

### Rate
| Feature | Description | Status |
|---------|-------------|--------|
| `AVG` | Batting average | active |
| `ISO` | Isolated power (SLG - AVG) | active |
| `BABIP` | Batting average on balls in play | active |
| `K_pct` | Strikeout rate | active |
| `BB_pct` | Walk rate | active |
| `BB_K_pct` | BB% minus K% | active |
| `SB_pct` | Stolen base success rate | active |

### Advanced
| Feature | Description | Status |
|---------|-------------|--------|
| `wOBA` | Weighted on-base average | active |
| `wRC_plus` | Weighted runs created plus (100 = league avg) | active |
| `RE24` | Run expectancy change per PA | active |

### Batted Ball
| Feature | Description | Status |
|---------|-------------|--------|
| `GB_pct` | Ground ball rate | active |
| `LD_pct` | Line drive rate | active |
| `HR_FB_pct` | HR per fly ball rate | active |

### Statcast
| Feature | Description | Status |
|---------|-------------|--------|
| `brl_percent` | Barrel rate | active |
| `ev95percent` | Hard hit rate (exit velo >= 95 mph) | active |
| `anglesweetspotpercent` | Sweet spot rate (launch angle 8-32) | active |
| `est_ba` | Expected batting average (xBA) | active |
| `est_slg` | Expected slugging (xSLG) | active |
| `est_woba` | Expected weighted OBA (xwOBA) | active |
| `sprint_speed` | Sprint speed (ft/s) | active |

### Regression Signals
| Feature | Description | Status |
|---------|-------------|--------|
| `BABIP_gap` | BABIP minus career expanding mean | active |
| `xBA_AVG_gap` | xBA minus AVG (overperformance signal) | active |
| `xSLG_SLG_gap` | xSLG minus SLG (overperformance signal) | active |

### Context
| Feature | Description | Status |
|---------|-------------|--------|
| `team_rpg` | Team runs per game | active |
| `experience` | Prior qualifying seasons (cumulative count) | active |
| `SB_rate` | Stolen bases per PA | active |
| `career_BABIP` | Career expanding mean BABIP | active |

### Year-over-Year Deltas
| Feature | Description | Status |
|---------|-------------|--------|
| `K_pct_delta` | Change in K% | active |
| `BB_pct_delta` | Change in BB% | active |
| `ISO_delta` | Change in ISO | active |
| `BABIP_delta` | Change in BABIP | active |

---

## Pitcher Model — 34 Active Features

### Volume
| Feature | Description | Status |
|---------|-------------|--------|
| `p_ipouts` | Outs recorded | counting |
| `p_bfp` | Batters faced | counting |
| `G` | Games played | active |
| `p_gf` | Games finished | active |

### Rate
| Feature | Description | Status |
|---------|-------------|--------|
| `ERA` | Earned run average | active |
| `WHIP` | Walks + hits per inning pitched | active |
| `K9` | Strikeouts per 9 innings | active |
| `BB9` | Walks per 9 innings | active |
| `HR9` | Home runs per 9 innings | active |
| `K_BB_pct` | K% minus BB% | active |
| `BABIP_allowed` | Pitcher BABIP | active |
| `LOB_pct` | Left on base percentage | active |

### Advanced
| Feature | Description | Status |
|---------|-------------|--------|
| `ERA_minus` | Park/league-adjusted ERA (100 = league avg, lower is better) | active |
| `xFIP_minus` | Park/league-adjusted xFIP (100 = league avg, lower is better) | active |

### Role
| Feature | Description | Status |
|---------|-------------|--------|
| `is_starter` | Starter flag (1 = starter) | active |
| `is_closer` | Closer flag (1 = >= 10 saves and non-starter) | active |

### Statcast
| Feature | Description | Status |
|---------|-------------|--------|
| `xera` | Expected ERA | active |
| `brl_percent_allowed` | Barrel rate allowed | active |
| `ev95percent_against` | Hard hit rate against (>= 95 mph) | active |
| `weighted_whiff_pct` | Usage-weighted whiff% across pitch types | active |
| `avg_hard_hit_pct` | Average hard hit percentage | active |
| `swstr_pct` | Swinging strike rate | active |
| `csw_pct` | Called + swinging strike % | active |
| `zone_pct` | % pitches in strike zone | active |
| `chase_pct` | % swings at pitches outside zone | active |
| `ff_avg_speed` | Four-seam fastball avg velocity (mph) | active |

### Batted Ball
| Feature | Description | Status |
|---------|-------------|--------|
| `GB_pct` | Ground ball rate | active |

### Regression Signals
| Feature | Description | Status |
|---------|-------------|--------|
| `BABIP_gap` | BABIP allowed minus career expanding mean | active |
| `ERA_FIP_gap` | ERA minus FIP (luck/defense signal) | active |

### Context
| Feature | Description | Status |
|---------|-------------|--------|
| `team_win_pct` | Team win percentage | active |
| `experience` | Prior qualifying seasons (cumulative count) | active |
| `PF` | Park factor | active |

### Year-over-Year Deltas
| Feature | Description | Status |
|---------|-------------|--------|
| `K9_delta` | Change in K/9 | active |
| `ERA_delta` | Change in ERA | active |
| `WHIP_delta` | Change in WHIP | active |
| `BB9_delta` | Change in BB/9 | active |
| `ff_velo_delta` | Change in fastball velocity | active |

---

**Source:** Feature lists defined in `config.yaml` under `features`. Built by `src/feature_builder.py` → `build_batter_features()` and `build_pitcher_features()`. Filtered by `src/model_trainer.py` → `get_feature_cols()`.

---

## External Projection Sources (FanGraphs — used in notebook 10)

These files are **not model inputs** — they are used post-prediction in the draft dashboard (Cell 6 of notebook 10) to add market context (ADP) and cross-system consensus to the model's `overall_rank`.

### Steamer Projections (`data/fg_predictions/steamer/`)
Pipe-delimited markdown format. Top 50 batters and pitchers.
- **Batters:** HR, R, RBI, SB, AVG, OBP, SLG, wOBA, wRC+, ADP
- **Pitchers:** IP, W, SO, K/9, ERA, FIP, WHIP, ADP

### ZiPS Projections (`data/fg_predictions/zips/`)
Tab-separated FanGraphs export format. Top 50 batters and pitchers.
- Same stat columns as Steamer. ADP included.

### ZiPS-DC Projections (`data/fg_predictions/zips_dc/`)
Tab-separated FanGraphs export format. Playing-time adjusted version of ZiPS.
- Same stat columns as Steamer/ZiPS. ADP included.

### FanGraphs Article Lists (`data/fg_predictions/articles/`)
Plain-text ranked lists (expert consensus). Provides `article_rank` used to cross-reference editorial opinion vs model rank and ADP.
- `top_50_hitters_FG.md`: `1.  Name -- Team, Pos`
- `top_50_pitchers_FG.md`: `1.  Name (SP/RP)`

### Consensus Columns Produced
After averaging Steamer + ZiPS + ZiPS-DC: `fg_hr`, `fg_r`, `fg_rbi`, `fg_sb`, `fg_avg`, `fg_wrcplus`, `fg_adp` (batters); `fg_ip`, `fg_so`, `fg_k9`, `fg_era`, `fg_whip`, `fg_adp` (pitchers).
