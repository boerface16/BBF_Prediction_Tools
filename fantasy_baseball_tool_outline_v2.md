# Fantasy Baseball H2H Points League Prediction Tool
## Project Outline v2 — Python + Retrosheet Raw Data + ML
### Target: Draft Day Rankings | Training: 2015–2023 | Test: 2024–2025

---

> **Before running this pipeline:** Complete the Statcast data
> collection setup documented in `statcast_data_collection.md`.
> That script (`src/statcast_pull.py`) must be run once from the
> command line to build the four Statcast CSV files this pipeline
> reads from. It is never called from within any pipeline notebook.

---

## SECTION 0 — Understanding the Target: ESPN H2H Points Scoring

Before any data is pulled or model is built, the ESPN scoring system must be
treated as the ground truth. Every statistical choice in this tool flows from
mapping real-world stats to fantasy point output.

### 0.1 — ESPN H2H Points Scoring (Batters)

| Stat              | ESPN Points |
|-------------------|-------------|
| Runs Scored (R)   | +1          |
| Total Bases (TB)  | +1 per base |
| RBI               | +1          |
| Walks (BB)        | +1          |
| Stolen Bases (SB) | +1          |
| Strikeouts (K)    | -1          |

> NOTE: Scoring is based on Total Bases, not individual hit types.
> TB = (1×1B) + (2×2B) + (3×3B) + (4×HR). This means a HR is worth
> +4 TB points plus +1 R plus +1 RBI = +6 points minimum on a solo HR.
> HBP and Caught Stealing are NOT scored in this league.

### 0.2 — ESPN H2H Points Scoring (Pitchers)

| Stat                  | ESPN Points |
|-----------------------|-------------|
| Innings Pitched (IP)  | +3 per IP   |
| Strikeouts (K)        | +1          |
| Wins (W)              | +5          |
| Saves (SV)            | +5          |
| Holds (HD)            | +2          |
| Hits Allowed (H)      | -1          |
| Earned Runs (ER)      | -2          |
| Walks Issued (BB)     | -1          |
| Losses (L)            | -2          |

> NOTE: Complete Games, Shutouts, and No-Hitters are NOT scored in
> this league. HBP allowed is NOT scored in this league.

---

## SECTION 1 — Data Sources & Architecture

### 1.1 — Raw Data Files (Primary Source: Your Retrosheet Files)

All core statistics are calculated directly from the following files.
No FanGraphs leaderboard pulls are used for stats that can be derived here.

| File | Contents | Key Use |
|------|----------|---------|
| `allplayers.csv` | Player info, team, season | Player ID mapping, position lookup |
| `gameinfo.csv` | Game-level info, teams, attendance, umpires | Park factors, game context |
| `teamstats.csv` | Team batting, pitching, fielding, line scores, lineups | League averages, team run environment |
| `batting.csv` | Batting stats by player by game | All batter rate/counting stats |
| `pitching.csv` | Pitching stats by player by game | All pitcher rate/counting stats |
| `fielding.csv` | Fielding stats by player by position by game | Positional context, defensive value |
| `plays.csv` | Parsed play-by-play for all available games | RE24, base-out states, run values |

### 1.2 — Supplemental Data Source: Statcast (Baseball Savant via pybaseball)

Statcast metrics cannot be calculated from Retrosheet event files —
they require pitch-tracking and ball-tracking hardware data that
Retrosheet does not contain.

**These metrics are collected in the PREREQUISITE step** (see the
Prerequisite section at the top of this document), which must be
completed before the main pipeline is run for the first time.
The prerequisite step produces four files and is never called
from within the main pipeline:

```
data/raw/statcast/statcast_batters_2015_2025.csv
data/raw/statcast/statcast_pitchers_2015_2025.csv
data/raw/statcast/statcast_pitchers_arsenal_2015_2025.csv
data/raw/id_map.csv
```

From this point forward in the main pipeline, Statcast data is
read directly from those CSVs. No pybaseball calls appear in
any main pipeline notebook or script.

**Statcast coverage window:** 2015–2025. This aligns exactly with
the training window (2015–2023) and test window (2024–2025).
There is no missing Statcast data problem in this project.

**Metrics available after the prerequisite step completes:**

Batters: xwOBA, xBA, xSLG, Barrel%, Hard Hit%, Average Exit
Velocity, Sweet Spot%, Sprint Speed

Pitchers: xERA, Barrel% allowed, Hard Hit% allowed, SwStr%,
CSW%, Chase%, Zone%, Fastball velocity, Spin rate, Extension,
Whiff% per pitch type (from arsenal file)



### 1.3 — What Is NOT Pulled From FanGraphs

The following stats are commonly sourced from FanGraphs in other tools
but are calculated from raw files in this project:

| Metric | Calculated From |
|--------|----------------|
| OPS, SLG, OBP | batting.csv |
| OPS+, ERA+ | batting.csv + teamstats.csv (league averages) |
| wOBA | batting.csv (formula provided by user) |
| wRAA | batting.csv (formula provided by user) |
| WAR (batter) | batting.csv + fielding.csv (formula provided by user) |
| FIP, xFIP | pitching.csv + plays.csv |
| WHIP | pitching.csv |
| K%, BB%, K-BB% | batting.csv / pitching.csv |
| RE24 | plays.csv |

> NOTE: wOBA, wRAA, and WAR require linear weights that are recalculated
> each season from the run environment in teamstats.csv and plays.csv.
> The user will provide the formulas for these three metrics. Do not
> substitute published FanGraphs values — calculate them from the raw data
> so they are consistent with the rest of the pipeline.

### 1.4 — RE24 Coverage Policy

**plays.csv does not cover every game in every season.** Before using RE24
as a model feature for any player-season:

1. Count the number of games with play-by-play data for that player-season
2. Count the total games that player appeared in (from batting.csv)
3. Calculate coverage rate: `PBP_Games / Total_Games`
4. **If coverage rate < 1.0 (any gap at all), exclude RE24 from that
   player-season's ML feature row.** Do not impute, do not scale.
5. RE24 is still calculated and reported where coverage is complete —
   it is only excluded from the ML training/prediction pipeline when
   coverage is incomplete.
6. Log all excluded player-seasons to `outputs/re24_exclusions.csv`
   so you can audit which players were affected each year.

**Why this is the right call:** Scaling or imputing RE24 would introduce
estimates into a metric that is specifically valued for its precision.
A scaled RE24 is not RE24 — it is a guess. The model is better served
by a clean feature that is sometimes absent than a noisy feature that
is always present.

**RE24 is still reported in the player output tables** regardless of
whether it was used as an ML feature — it is informative for human review
even when excluded from the model.

### 1.5 — Minimum Sample Thresholds

Rows below these thresholds are excluded from the ML training and test
sets. They are retained in the player database for historical reference
but flagged as `LOW_SAMPLE`.

| Group | Threshold | Exception |
|-------|-----------|-----------|
| Batters | 200 PA | 2020 season: 100 PA (60-game season) |
| Pitchers | 45 IP | 2020 season: 25 IP (60-game season) |

All other shortened seasons (strike years: 1981, 1994–95) use the same
proportional logic if they fall within the 2015–2025 window.
Neither 1981 nor 1994–95 falls in the training or test window, so this
is noted for completeness only.

### 1.6 — Project Folder Structure

```
fantasy_baseball_tool/
│
├── data/
│   ├── raw/
│   │   ├── retrosheet/
│   │   │   ├── allplayers.csv
│   │   │   ├── gameinfo.csv
│   │   │   ├── teamstats.csv
│   │   │   ├── batting.csv
│   │   │   ├── pitching.csv
│   │   │   ├── fielding.csv
│   │   │   └── plays.csv
│   │   ├── statcast/
│   │   │   ├── statcast_batters_2015_2025.csv
│   │   │   ├── statcast_pitchers_2015_2025.csv
│   │   │   └── statcast_pitchers_arsenal_2015_2025.csv
│   │   └── id_map.csv
│   ├── processed/
│   │   ├── batters_master.csv
│   │   └── pitchers_master.csv
│   └── keepers_candidates.txt
│
├── notebooks/
│   ├── 01_data_build.ipynb
│   ├── 02_advanced_stats_calculator.ipynb
│   ├── 03_re24_engine.ipynb
│   ├── 04_espn_points_mapper.ipynb
│   ├── 05_feature_engineering_and_viz.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_predictions_vs_actual.ipynb
│   └── 08_keeper_evaluator.ipynb
│
├── src/
│   ├── data_builder.py
│   ├── advanced_stats.py
│   ├── re24_engine.py
│   ├── espn_points_mapper.py
│   ├── feature_builder.py
│   ├── model_trainer.py
│   ├── predictor.py
│   └── keeper_evaluator.py
│
├── outputs/
│   ├── draft_rankings_batters.csv
│   ├── draft_rankings_pitchers.csv
│   ├── model_performance_report.csv
│   ├── re24_exclusions.csv
│   ├── keeper_rankings.csv
│   └── figures/
│       ├── feature_importance/
│       ├── scatter_trends/
│       ├── model_evaluation/
│       └── keeper_trajectories/
│
└── requirements.txt
```

### 1.7 — Data Flow Architecture

```
[batting.csv]  [pitching.csv]  [fielding.csv]  [plays.csv]  [teamstats.csv]
      |               |               |              |               |
      v               v               v              v               v
[Advanced Stats Calculator]      [RE24 Engine]     [League Avg Builder]
(OPS, SLG, OPS+, wOBA,          (RE Matrix,        (Park Factors,
 wRAA, WAR, FIP, WHIP,           per-PA RE24,       wOBA weights,
 ERA+, K%, BB%)                  Available Runs)    ERA+ baseline)
      |               |               |
      +---------------+---------------+
                      |
             [Master Player Table]
             (one row = player-season)
                      |
         +------------+------------+
         |                         |
  [Batter Table]           [Pitcher Table]
         |                         |
         +------------+------------+
                      |
             [Statcast Merge]
             (2015–2025 only,
              Baseball Savant)
                      |
             [ESPN Points Mapper]
             (applies scoring formula to
              each player's stat line to
              produce the ML target variable:
              ESPN_Pts per player per season)
                      |
             [ML Feature Matrix]
                      |
         +------------+------------+
         |                         |
  [XGBoost Model]         [LightGBM Model]
         |                         |
         +-----[Best Performer]----+
                      |
             [Draft Rankings Output]
```

---

## SECTION 2 — Explicit Stat Calculation Inventory

This section is the authoritative reference for every statistic in this
tool. It is divided into three parts:

- **Part A:** Stats calculated entirely from the raw Retrosheet files
- **Part B:** Stats pulled from Baseball Savant (Statcast — cannot be
  derived from Retrosheet)
- **Part C:** Formulas the user must provide before coding begins

No stat appears in the ML feature set or output tables unless it is
accounted for in one of these three parts.

---

### PART A — Stats Calculated From Raw Retrosheet Files

#### A.1 — Columns Assumed Present in Each Raw File

These are the columns the calculation engine depends on. If any column
is named differently or is absent in your actual files, flag it before
coding — every formula below requires exact column name matches.

**batting.csv** (per player per game):
`player_id, game_id, year, team, AB, R, H, 2B, 3B, HR, RBI, SH, SF,
HBP, BB, IBB, SO, SB, CS, GIDP`

**pitching.csv** (per player per game):
`player_id, game_id, year, team, GS, IP_outs, BF, H, 2B, 3B, HR, R,
ER, BB, IBB, SO, HBP, WP, BK, SH, SF, GDP, W, L, SV, HD,
inherited_runners, inherited_scored`

> IP_outs = innings pitched recorded as outs (e.g., 6.1 IP = 19 outs).
> Converted to decimal IP in the pipeline: IP = IP_outs / 3.

**fielding.csv** (per player per position per game):
`player_id, game_id, year, team, position, PO, A, E, DP, PB`

**plays.csv** (per plate appearance):
`game_id, year, inning, half_inning, batter_id, pitcher_id,
base_state_start, outs_start, base_state_end, outs_end,
runs_scored_on_play, event_type, batted_ball_type`

**teamstats.csv** (per team per game):
`game_id, year, team, R, H, 2B, 3B, HR, RBI, BB, SO, SB, CS, HBP,
SF, SH, AB, IP_outs, ER, BF`

**gameinfo.csv** (per game):
`game_id, year, date, home_team, away_team, home_score, away_score,
park_id, attendance`

**allplayers.csv** (per player per team per season):
`player_id, year, team, first_name, last_name, position, age`

---

#### A.2 — Batter Counting & Rate Stats (From batting.csv)

All stats aggregated per player per season by summing game-level rows,
then applying the formulas below.

| Stat | Formula | Source |
|------|---------|--------|
| PA | AB + BB + HBP + SF + SH | batting.csv |
| 1B | H − 2B − 3B − HR | batting.csv |
| TB | (1B×1) + (2B×2) + (3B×3) + (HR×4) | derived |
| AVG | H / AB | batting.csv |
| OBP | (H + BB + HBP) / (AB + BB + HBP + SF) | batting.csv |
| SLG | TB / AB | derived |
| OPS | OBP + SLG | derived |
| ISO | SLG − AVG | derived |
| BABIP | (H − HR) / (AB − SO − HR + SF) | batting.csv |
| K% | SO / PA | derived |
| BB% | BB / PA | derived |
| BB%−K% | BB% − K% | derived |
| HR/PA | HR / PA | derived |
| SB% | SB / (SB + CS) | batting.csv |
| wOBA | See C.1 — derived from RE Matrix weights | plays.csv + batting.csv |
| wRAA | See C.2 — (wOBA − lg_wOBA) / wOBA_scale × PA | derived from wOBA |
| wRC | See C.3 — wRAA-based + league R/PA adjustment | derived from wRAA |
| wRC+ | See C.3 — park + league adjusted, 100 = avg | derived from wRC |
| oWAR | See C.5 — offensive WAR (defense excluded in v1) | plays.csv + batting.csv + fielding.csv |
| ESPN_Pts | (TB×1)+(R×1)+(RBI×1)+(BB×1)+(SB×1)+(SO×−1) | derived |

#### A.3 — Batter Park & League Adjusted Stats
(batting.csv + teamstats.csv + gameinfo.csv)

**Step 1 — Park Factor** (5-year rolling average per park):

A single-season park factor is noisy — sample size at any one park
in one year is limited. A 5-year rolling average smooths out
year-to-year variance while still tracking real ballpark changes
(new fences, altitude, etc.).

```
Park_Factor_Year_N =
    mean of single-season park factors for years N, N-1, N-2, N-3, N-4

Single-season Park_Factor =
    (Runs scored by both teams in home games at this park)
    / (Runs scored by both teams in away games for this team)
    -- sourced from gameinfo.csv + teamstats.csv run totals

If fewer than 5 seasons of data exist for a park (new stadium):
    use all available seasons. Log the number of seasons used
    in the park_factor_lookup table.
```

**Step 2 — League Averages** (once per season, from teamstats.csv):
```
League_OBP = Σ(H + BB + HBP) / Σ(AB + BB + HBP + SF)
League_SLG = Σ(TB) / Σ(AB)
             where TB is recalculated from teamstats.csv hit columns
```

**Step 3 — OPS+:**
```
OPS+ = 100 × [(OBP / League_OBP) + (SLG / League_SLG) − 1]
             / Park_Factor
```
100 = league average. Above 100 = better than average.

#### A.4 — Batter Batted Ball Rates (From plays.csv)

plays.csv contains a `batted_ball_type` column that records the
type of each batted ball using single-character Retrosheet event
codes. The standard codes are:

```
G = Ground Ball
F = Fly Ball
L = Line Drive
P = Popup
```

These single-character codes are used to count batted ball events.
In the formulas below, the variable names GB, FB, LD, PU refer to
the COUNT of plays where `batted_ball_type` equals G, F, L, and P
respectively — they are not column names. They are derived counts
produced by grouping plays.csv by batter_id, year, and
batted_ball_type, then pivoting.

Calculated per batter per season:

| Stat | Formula | Notes |
|------|---------|-------|
| GB% | GB / (GB+FB+LD+PU) | Ground ball rate |
| FB% | FB / (GB+FB+LD+PU) | Fly ball rate |
| LD% | LD / (GB+FB+LD+PU) | Line drive rate |
| PU% | PU / (GB+FB+LD+PU) | Popup rate |
| HR/FB% | HR / FB | Home runs per fly ball — HR from batting.csv |

#### A.5 — Pitcher Counting & Rate Stats (From pitching.csv)

All stats aggregated per player per season by summing game-level rows.

| Stat | Formula | Source |
|------|---------|--------|
| IP | IP_outs / 3 | pitching.csv |
| ERA | (ER × 9) / IP | derived |
| WHIP | (BB + H) / IP | pitching.csv |
| K/9 | (SO × 9) / IP | derived |
| BB/9 | (BB × 9) / IP | derived |
| HR/9 | (HR × 9) / IP | derived |
| K% | SO / BF | derived |
| BB% | BB / BF | derived |
| K−BB% | K% − BB% | derived |
| BABIP_allowed | (H−HR) / (BF−SO−HR−BB) | derived |
| LOB% | (H+BB+HBP−ER) / (H+BB+HBP−(1.4×HR)) | derived |
| ESPN_Pts | (IP×3)+(SO×1)+(W×5)+(SV×5)+(HD×2)+(ER×−2)+(H×−1)+(BB×−1)+(L×−2) | derived |

#### A.6 — Pitcher Batted Ball Rates (From plays.csv)

Calculated by filtering plays.csv to rows where pitcher_id matches,
then counting batted_ball_type codes — same method as A.4.

| Stat | Formula | Notes |
|------|---------|-------|
| GB% | GB / (GB+FB+LD+PU) | Ground ball rate allowed |
| FB% | FB / (GB+FB+LD+PU) | Fly ball rate allowed |
| LD% | LD / (GB+FB+LD+PU) | Line drive rate allowed |
| HR/FB% | HR / FB | HR per fly ball allowed |

#### A.7 — Pitcher Park & League Adjusted Stats
(pitching.csv + teamstats.csv + gameinfo.csv)

Uses the same Park_Factor calculated in A.3.

**League_ERA** (once per season, from teamstats.csv):
```
League_ERA = (Σ ER × 9) / Σ IP
```

**ERA+:**
```
ERA+ = 100 × (League_ERA / ERA) × Park_Factor
```
100 = league average. Above 100 = better than average.

**FIP Constant** (once per season, from teamstats.csv):
```
FIP_League = lgERA − (((13×lgHR) + (3×(lgBB + lgHBP)) − (2×lgK)) / lgIP)
```

The constant brings FIP onto the ERA scale. League average ERA and
league average FIP are equal by design. The constant is generally
around 3.10 but is recalculated each season from teamstats.csv so
it reflects the actual run environment of that year.

**FIP:**
```
FIP = ((13×HR) + (3×(BB + HBP)) − (2×K)) / IP + FIP_League
```

The weights reflect the relative run-prevention value of each event:
home runs (13) are weighted most heavily, walks and HBP (3) less so,
and strikeouts (2) as a negative contribution.

**xFIP:**

xFIP is identical to FIP except it replaces the pitcher's actual HR
total with an expected HR total derived from fly balls allowed
multiplied by the league average HR/FB rate. This removes HR variance
due to luck or park factors.

```
lgHR_per_FB = lgHR / lgFB
              -- calculated from teamstats.csv + plays.csv for that season

xFIP = ((13×(FB × lgHR_per_FB)) + (3×(BB + HBP)) − (2×K)) / IP + FIP_League
```

The same FIP_League constant used in FIP applies here unchanged.

#### A.8 — RE24 & Available Runs (From plays.csv)

Full methodology detailed in Section 3. Summary of outputs:

| Stat | Calculated From | Condition |
|------|----------------|-----------|
| RE Matrix (24 cells) | plays.csv — all PA in season | Recalculated per year |
| RE24 per batter (season) | plays.csv — sum of all PA | Coverage = 100% only |
| RE24 per pitcher (season) | plays.csv — sum of all PA | Coverage = 100% only |
| Total_Available_Runs | RE start-state value, summed per player | Coverage = 100% only |
| RE24_Efficiency | RE24 / Total_Available_Runs | Coverage = 100% only |

---

### PART B — Stats From Baseball Savant (Statcast, 2015–2025 Only)

These metrics require pitch-tracking and ball-tracking hardware data.
They cannot be derived from Retrosheet event files under any
circumstance. Pulled directly from Baseball Savant as pre-built CSVs.

**Batter Statcast Stats:**

| Stat | What It Measures |
|------|-----------------|
| xwOBA | Expected wOBA based on contact quality — removes fielding luck |
| xBA | Expected batting average from exit velocity and launch angle |
| xSLG | Expected slugging from contact quality |
| Barrel% | % of batted balls meeting Statcast barrel criteria (EV + LA combo) |
| Hard Hit% | % of batted balls at 95+ mph exit velocity |
| Avg Exit Velocity | Mean exit velocity across all batted ball events |
| Sweet Spot% | % of batted balls at 8–32° launch angle |
| Sprint Speed | Feet per second in a player's fastest running situations |

**Pitcher Statcast Stats:**

| Stat | What It Measures |
|------|-----------------|
| xERA | Expected ERA from quality of contact allowed |
| Barrel% allowed | Barrels per PA against |
| Hard Hit% allowed | Hard contact rate (95+ mph EV) allowed |
| SwStr% | Swinging strike rate — strongest predictor of K rate |
| CSW% | Called + swinging strike % — broader stuff metric |
| Whiff% | Swinging strike rate per pitch type |
| Chase% | Rate at which batters swing at pitches outside the zone |
| Zone% | Rate at which pitcher throws pitches inside the zone |
| Fastball velocity | Primary fastball average velocity |
| Spin rate | Per pitch type — affects movement and whiff generation |
| Extension | Release point extension in feet toward home plate |

---

### PART C — Confirmed Formulas

All four previously open items are now confirmed. No formula decisions
remain outstanding. This section is the authoritative reference for
every derived metric that requires more than raw column arithmetic.

---

#### C.1 — wOBA (Confirmed: Derived From RE Matrix Each Season)

wOBA is calculated entirely from plays.csv using annually-derived
linear weights. No external weight tables are used. Weights change
each season because run-scoring environments change.

**Formula:**
```
wOBA = (wBB×uBB + wHBP×HBP + w1B×1B + w2B×2B + w3B×3B + wHR×HR)
       / (AB + uBB + SF + HBP)

where:
  uBB = BB − IBB    (unintentional walks only)
  1B  = H − 2B − 3B − HR
```

**Weight Derivation — 4 Stages (run once per season year):**

Stage 1 — Build the RE Matrix (see Section 3 for full methodology):
From plays.csv, compute RE(base_state, outs) — the average runs
scored from each of the 24 states through end of half-inning.

Stage 2 — Compute Run Values Above Average per Event:
For every plate appearance:
```
run_value = runs_scored_on_play + RE(state_after) − RE(state_before)
            (if third out is made: RE(state_after) = 0)
```
Average run_value across all instances of each event type:
```
rv_uBB ≈ +0.29    (unintentional walks)
rv_HBP ≈ +0.31    (hit by pitch)
rv_1B  ≈ +0.44    (singles)
rv_2B  ≈ +0.74    (doubles)
rv_3B  ≈ +1.01    (triples)
rv_HR  ≈ +1.39    (home runs)
rv_out ≈ −0.26    (all outs)
```
Note: values above are approximate season-neutral examples.
Actual values are recalculated from plays.csv each season year.

Stage 3 — Re-center Relative to Outs (outs = 0):
```
lw_uBB = rv_uBB − rv_out
lw_HBP = rv_HBP − rv_out
lw_1B  = rv_1B  − rv_out
lw_2B  = rv_2B  − rv_out
lw_3B  = rv_3B  − rv_out
lw_HR  = rv_HR  − rv_out
```

Stage 4 — Scale to OBP:
```
raw_league_wOBA = (lw_uBB×lg_uBB + lw_HBP×lg_HBP + lw_1B×lg_1B
                 + lw_2B×lg_2B + lw_3B×lg_3B + lw_HR×lg_HR)
                 / (lg_AB + lg_uBB + lg_SF + lg_HBP)

league_OBP_noIBB = (lg_H + lg_uBB + lg_HBP)
                   / (lg_AB + lg_uBB + lg_SF + lg_HBP)

wOBA_scale = league_OBP_noIBB / raw_league_wOBA

wBB  = lw_uBB × wOBA_scale
wHBP = lw_HBP × wOBA_scale
w1B  = lw_1B  × wOBA_scale
w2B  = lw_2B  × wOBA_scale
w3B  = lw_3B  × wOBA_scale
wHR  = lw_HR  × wOBA_scale
```

wOBA_scale and the final weights are stored in a
`season_constants.csv` lookup table, one row per year.
All downstream metrics (wRAA, wRC, wRC+, WAR) read from this table.

---

#### C.2 — wRAA (Confirmed)

```
wRAA = ((wOBA − league_wOBA) / wOBA_scale) × PA
```

league_wOBA = PA-weighted average wOBA across all qualified batters
that season (calculated after wOBA is computed for all players).
wOBA_scale = from season_constants.csv.

Scale reference (these are estimates — actual values derive from data):
Excellent ≥ 40 | Great ≥ 20 | Above Average ≥ 10 | Average = 0
Below Average ≤ −5 | Poor ≤ −10 | Awful ≤ −20

wRAA is always league-centered at 0 regardless of season year.
Ten wRAA = approximately +1 win.

---

#### C.3 — wRC and wRC+ (Confirmed — Added to Batter Stat Set)

wRC (Weighted Runs Created):
```
wRC = (((wOBA − league_wOBA) / wOBA_scale) + (league_R / league_PA)) × PA
```

wRC+ (Park and League Adjusted — 100 = league average):
```
wRC+ = 100 × (
    ( (wRAA / PA + league_R / league_PA)
    + (league_R / league_PA − park_factor × (league_R / league_PA)) )
    / (league_wRC / league_PA)
)
```

Where:
- league_R / league_PA = league runs per plate appearance
  (from teamstats.csv)
- park_factor = park factor ratio (1.00 = neutral), from A.3
- league_wRC / league_PA = league-wide wRC per PA
  (calculated after wRC is computed for all players)

wRC+ inputs all come from plays.csv, batting.csv, teamstats.csv,
and gameinfo.csv. No external data required.

---

#### C.4 — SIERA (Confirmed)

All inputs calculable from plays.csv and pitching.csv.

```
SIERA = 6.145
      − 16.986 × (SO/PA)
      + 11.434 × (BB/PA)
      − 1.858  × ((GB − FB − LD) / PA)
      + 7.653  × (SO/PA)²
      − 6.664  × (LD/PA)²
      + 10.130 × (GB/PA) × (BB/PA)
      − 5.195  × (SO/PA) × (BB/PA)
```

Where PA = batters faced (BF). GB, FB, LD = seasonal counts
from plays.csv filtered to rows where pitcher_id matches.

---

#### C.5 — WAR (Confirmed Formula and All Sub-Components)

**Full Formula:**
```
WAR = (Batting_Runs + Baserunning_Runs + Fielding_Runs
       + Positional_Adjustment + League_Adjustment
       + Replacement_Runs)
      / Runs_Per_Win

Runs_Per_Win = 9 + (League_ERA / 2)
               -- recalculated per season from teamstats.csv
```

---

**Sub-component 1 — Batting Runs:**
```
Batting_Runs = wRAA
             + (lgR/PA − (PF × lgR/PA)) × PA
             + (lgR/PA − AL_or_NL_nonpitcher_wRC/PA) × PA
```
- lgR/PA = league runs per PA from teamstats.csv
- PF = park factor from gameinfo.csv (same as A.3)
- AL_or_NL_nonpitcher_wRC/PA = league-specific non-pitcher
  wRC per PA (accounts for DH rule differences by league)

---

**Sub-component 2 — Baserunning Runs:**
```
Baserunning_Runs = UBR + wSB + wGDP
```

**wSB (Weighted Stolen Base Runs):**
```
runSB  = run value of a successful steal (from RE Matrix transitions)
runCS  = 2 × RunsPerOut + 0.075
         where RunsPerOut = League_R / (League_Outs_recorded)

lgwSB  = league average wSB per (1B + uBB + HBP − IBB)
         -- calculated across all players in that season

wSB    = (SB × runSB) + (CS × runCS) − (lgwSB × (1B + uBB + HBP − IBB))
```

**UBR (Ultimate Base Running):**
UBR values the non-SB baserunning decisions a player makes.
For each baserunning opportunity in plays.csv:

1. Identify the batting event type and base-out state
2. Look up the league-average run expectancy change for that
   event + state combination across all players in that season
   (e.g., runner on 2B, single hit — average change = 0.7 runs)
3. Calculate the actual run expectancy change for this specific play
   using the RE Matrix
4. UBR for that play = actual RE change − average RE change for
   that situation
5. Sum all UBR plays per player per season

**wGDP (Weighted Grounded Into Double Play):**
GDP run value is derived from the RE Matrix change when a double
play occurs vs. the average expected outcome from that base-out
state. Calculated from plays.csv GDP events, compared to league
average GDP rate in GDP-eligible situations.

---

**Sub-component 3 — Fielding Runs:**
For version 1 of this tool, Fielding_Runs = 0 for all players.
WAR is labeled **oWAR** (Offensive WAR) in all outputs to be
explicit that defensive value is excluded. This is the recommended
approach for v1 — adding a noisy defensive estimate would be worse
than omitting it cleanly.

This can be upgraded to a full defensive model in a future version.

---

**Sub-component 4 — Positional Adjustment:**
Fixed run values per position per 150 games (confirmed):
```
C:     +12.5    SS:    +7.5
2B:    +2.5     3B:    +2.5
CF:    +2.5     LF/RF: −7.5
1B:    −12.5    DH:    −17.5
```
Prorated to actual games played:
```
Positional_Adjustment = (position_value / 150) × G
```

---

**Sub-component 5 — League Adjustment:**
Accounts for DH rule and AL vs NL offensive environment differences.
```
League_Adjustment = (lgR/PA − league_specific_nonpitcher_wRC/PA) × PA
```
For seasons after 2022 (universal DH adopted), this adjustment
approaches zero. For 2015–2021, AL and NL are computed separately.

---

**Sub-component 6 — Replacement Runs:**
```
Replacement_Runs = (−20 / 600) × PA    (= −0.0333 runs per PA)
```
Standard replacement level: a freely available replacement player
produces approximately −20 runs per 600 PA below average.

---

**Calculation Order for WAR (must follow this sequence):**
1. Build RE Matrix for the season (Section 3)
2. Derive wOBA weights → store in season_constants.csv (C.1)
3. Calculate wOBA per player (C.1)
4. Calculate league_wOBA, wOBA_scale (C.1)
5. Calculate wRAA per player (C.2)
6. Calculate wRC and wRC+ per player (C.3)
7. Calculate Batting_Runs (C.5 sub-component 1)
8. Calculate runSB, runCS, lgwSB → wSB per player (C.5 sub-component 2)
9. Calculate UBR per player from plays.csv (C.5 sub-component 2)
10. Calculate wGDP per player (C.5 sub-component 2)
11. Sum Baserunning_Runs (C.5 sub-component 2)
12. Apply Positional Adjustment (C.5 sub-component 4)
13. Apply League Adjustment (C.5 sub-component 5)
14. Apply Replacement Runs (C.5 sub-component 6)
15. Calculate Runs_Per_Win from teamstats.csv
16. Divide total run components by Runs_Per_Win → oWAR

---

## SECTION 3 — RE24 Engine (From plays.csv)

### 3.1 — What RE24 Measures

RE24 (Run Expectancy Based on 24 Base-Out States) measures the change
in run expectancy a batter or pitcher creates across every plate
appearance. There are 24 possible states: 8 base configurations
(empty, 1st, 2nd, 3rd, 1st+2nd, 1st+3rd, 2nd+3rd, loaded) × 3 out
counts (0, 1, 2).

Each state has an expected number of runs that will score by end of
inning, derived from historical play-by-play data. A batter's RE24 for
a given PA is the change in that run expectancy plus any runs that
scored on the play.

### 3.2 — Building the Run Expectancy Matrix From plays.csv

Step 1: Parse plays.csv to identify, for each plate appearance:
- The base-out state at the START of the PA
- The base-out state at the END of the PA
- The number of runs that scored on the play

Step 2: Group all PA endings by their starting base-out state.
For each of the 24 states, average the total runs scored from that
point to the end of the inning. This produces the RE Matrix — a
24-cell table of expected run values.

Step 3: Recalculate this matrix separately for each season year
(2015–2025). Run environments change year to year, and a 2015 RE
matrix is not appropriate for 2023 data.

```
RE_Matrix[base_state][out_count] = 
    mean(runs scored from this state to end of inning)
    -- across all instances of this state in that season
```

### 3.3 — RE24 Per Batter Per Season

For each plate appearance in plays.csv:
```
RE24_PA = (RE_Matrix[end_state][end_outs] + runs_scored_on_play)
          − RE_Matrix[start_state][start_outs]
```

Sum all PA-level RE24 values per player per season:
```
RE24_Season = sum(RE24_PA) for all PA by that player in that season
```

### 3.4 — Available Runs at Start of PA

This is a new metric specific to this tool. For each plate appearance,
the "available runs" is the run expectancy at the START of the PA —
i.e., how many runs were potentially on the table when the batter
stepped to the plate.

```
Available_Runs_PA = RE_Matrix[start_state][start_outs]
```

Sum across all PA in a season:
```
Total_Available_Runs_Season = sum(Available_Runs_PA) for all PA
```

### 3.5 — RE24 Efficiency Ratio

By combining RE24 and Total Available Runs, we can measure how well a
batter converted his opportunities into actual run value:

```
RE24_Efficiency = RE24_Season / Total_Available_Runs_Season
```

**Interpretation:**
- A positive RE24_Efficiency means the batter added more run value than
  the situations he inherited were worth — he improved his team's run
  environment relative to what was already there
- A negative RE24_Efficiency means the batter destroyed run value
  relative to his opportunities
- This metric is most meaningful for cleanup hitters and middle-of-order
  bats who consistently bat with runners on base
- Leadoff hitters will have low Total_Available_Runs (they often bat
  with bases empty) but can still have high RE24_Efficiency

**Reporting:** Both RE24 and RE24_Efficiency are reported in the player
output tables. RE24 is used as the ML feature (see Section 5 for
coverage rules). RE24_Efficiency is a contextual reporting metric.

### 3.6 — RE24 for Pitchers

For pitchers, RE24 is the negative of the batter RE24 they allow:
```
Pitcher_RE24_PA = −(Batter_RE24_PA for that PA)
```

A pitcher who induces a double play in a high-leverage state generates
a large positive RE24 (good). A pitcher who walks the bases loaded
generates a large negative RE24 (bad).

Sum per pitcher per season as with batters.

### 3.7 — RE24 Coverage Check (Implementation)

Before any RE24 value enters the ML pipeline:

```python
# For each player-season:
pbp_games = count of games in plays.csv where player appears
total_games = count of games in batting.csv (or pitching.csv) where player appears
coverage = pbp_games / total_games

if coverage < 1.0:
    re24_ml_eligible = False
    log to re24_exclusions.csv
else:
    re24_ml_eligible = True
```

RE24 values are still stored in the master table for all players.
The `re24_ml_eligible` flag controls whether RE24 is passed to the
feature matrix in Section 5.

---

## SECTION 4 — Building the ESPN Points Mapper

### 4.1 — Purpose

The ESPN Points Mapper converts a player's real-world stat line into
an estimated fantasy point total. This is the bridge between baseball
statistics and the ML target variable.

### 4.2 — Batter Points Formula

```
ESPN_Pts_Batter =
    (TB × 1) +
    (R  × 1) +
    (RBI × 1) +
    (BB  × 1) +
    (SB  × 1) +
    (SO  × -1)
```

Where TB = (1B × 1) + (2B × 2) + (3B × 3) + (HR × 4),
calculated from hit-type columns in batting.csv.

HBP and CS are not scored and are excluded from the formula.

All input stats come from batting.csv, aggregated per player per season.

### 4.3 — Pitcher Points Formula

```
ESPN_Pts_Pitcher =
    (IP  × 3)  +
    (K   × 1)  +
    (W   × 5)  +
    (SV  × 5)  +
    (HD  × 2)  +
    (ER  × -2) +
    (H_allowed × -1) +
    (BB_allowed × -1) +
    (L   × -2)
```

All input stats come from pitching.csv, aggregated per player per season.

CG, SHO, No-Hitter, and HBP allowed are NOT scored and are excluded
from the formula entirely.

---

## SECTION 5 — Feature Engineering & Exploratory Visualizations

### 5.2 — Batter Feature Set

**Group A: Contact Quality (Statcast — 2015–2025)**
- Barrel%
- Hard Hit%
- xwOBA
- xBA
- xSLG
- Average Exit Velocity
- Sweet Spot%

**Group B: Plate Discipline (From batting.csv)**
- BB%
- K%
- BB% − K% (net discipline score)
- OBP
- ISO

**Group C: Speed & Baserunning (Statcast + batting.csv)**
- Sprint Speed (Statcast)
- SB Attempt Rate (from batting.csv)
- SB Success Rate / SB% (from batting.csv)

**Group D: Situational & Contextual**
- RE24 *(only if re24_ml_eligible = True for that player-season)*
- RE24_Efficiency *(same coverage condition as RE24)*
- wOBA (calculated from batting.csv via RE Matrix weights)
- wRAA
- wRC+ (park and league adjusted)
- OPS+
- Lineup position (averaged across season — from plays.csv batting order)
- Team runs per game (from teamstats.csv — lineup quality proxy)

**Group E: Health & Durability**
- PA total for the season
- Age at season start
- Seasons in MLB (experience proxy)

**Group F: Regression Signals — Luck Detection**
- BABIP vs career BABIP (gap = luck indicator)
- HR/FB% vs career HR/FB%
- xBA − AVG (Statcast luck gap)
- xSLG − SLG (Statcast luck gap for power)
- oWAR

### 5.3 — Pitcher Feature Set

**Group A: Stuff (Statcast — 2015–2025)**
- SwStr%
- CSW% (Called + Swinging Strike%)
- Whiff% per primary pitch type
- Fastball velocity
- Fastball velocity trend (current year vs prior year)

**Group B: Command & Control (From pitching.csv)**
- K-BB%
- BB%
- K%
- WHIP

**Group C: ERA Estimators (From pitching.csv + RE Matrix)**
- SIERA
- xFIP
- FIP
- xERA (Statcast)
- ERA+

**Group D: Batted Ball Profile (Statcast + pitching.csv)**
- GB%
- Barrel% allowed
- Hard Hit% allowed
- HR/FB% (deviation from career = regression signal)

**Group E: Role & Workload**
- Starter vs Reliever binary flag (from pitching.csv GS column)
- IP total for season
- BF (batters faced)
- Team win% (affects SP Win opportunities)
- Closer role flag (SV > 3 in prior season)

**Group F: Regression Signals**
- LOB% (strand rate) deviation from league average
- BABIP allowed vs career BABIP allowed
- ERA vs FIP gap
- ERA vs SIERA gap

### 5.4 — Exploratory Visualization Suite (Section 5 Graphs)

All graphs are generated for the TEST years (2024–2025) and saved to
`outputs/figures/scatter_trends/`. Each graph labels the top 5 players
by the metric shown.

The top 5 players are annotated directly on the plot with their last name
and a marker color distinct from the rest of the field. Trend lines use
OLS linear regression with a shaded 95% confidence band.

**Batter Scatter Plots (each = one graph, test years)**

| Graph | X-Axis | Y-Axis | Purpose |
|-------|--------|--------|---------|
| B-01 | xwOBA | ESPN Fantasy Points | Core quality-to-production relationship |
| B-02 | Barrel% | HR Total | Barrel rate as HR predictor |
| B-03 | BB% − K% | OBP | Discipline index vs on-base rate |
| B-04 | Sprint Speed | SB Total | Speed metric vs stolen base output |
| B-05 | BABIP gap (actual − career) | AVG change YoY | Luck detection: does BABIP gap predict regression? |
| B-06 | xBA − AVG | Next-year AVG change | Statcast luck gap validation |
| B-07 | RE24 | ESPN Fantasy Points | Situational value vs total output |
| B-08 | RE24_Efficiency | wRAA | Opportunity conversion vs run value |
| B-09 | wOBA | wRAA | Sanity check — these should track closely |
| B-10 | OPS+ | ESPN Fantasy Points | Park-adjusted performance vs fantasy output |

**Pitcher Scatter Plots**

| Graph | X-Axis | Y-Axis | Purpose |
|-------|--------|--------|---------|
| P-01 | SwStr% | K% | Stuff metric vs strikeout rate |
| P-02 | K-BB% | FIP | Command vs fielding-independent ERA |
| P-03 | FIP − ERA (gap) | Next-year ERA change | Regression signal validation |
| P-04 | SIERA | Next-year ERA | SIERA as forward-looking ERA predictor |
| P-05 | Barrel% allowed | ER total | Contact quality allowed vs run damage |
| P-06 | LOB% | ERA | Strand rate luck indicator |
| P-07 | ERA+ | ESPN Fantasy Points | Adjusted ERA vs fantasy output |
| P-08 | WHIP | ESPN Fantasy Points | WHIP vs total fantasy production |
| P-09 | Pitcher RE24 | ESPN Fantasy Points | Situational pitching value vs output |
| P-10 | xERA | Actual ERA next year | Statcast ERA estimator validation |

**Histogram / Distribution Plots**

- Distribution of ESPN Fantasy Points by position (box plots)
- Distribution of RE24 for batters (test years), with coverage flag
  shown as different color
- Histogram of BABIP luck gaps across all qualified batters
- Histogram of ERA − FIP gaps across all qualified pitchers

---

## SECTION 6 — Machine Learning Pipeline

### 6.1 — Beginner-Friendly ML Overview

The ML model does one job: given a player's stats from Year N, predict
how many ESPN fantasy points they will produce in Year N+1.

This is a regression problem. The output is a number (projected points),
not a category.

### 6.2 — Target Variable

```
Target = ESPN_Fantasy_Points in Year N+1
         (calculated by the ESPN Points Mapper in Section 4)
```

One row in the training data = one player-season.
The model learns from Year N features → Year N+1 target.

**Training set:** 2015–2022 (features from year N, target from year N+1;
the latest training pair uses 2022 features → 2023 target)
**Validation set:** 2023 (features from 2023, target from 2024 — used
to tune model hyperparameters)
**Test set:** 2024 (features from 2024, target from 2025 actuals)

> This is a strict time-based split. No future data ever enters training.
> 2025 is the production prediction year.

### 6.3 — Model Selection: XGBoost vs LightGBM

Both models are trained and evaluated side-by-side on the validation set.
The better-performing model becomes the production tool.

**XGBoost (eXtreme Gradient Boosting)**
- Builds an ensemble of decision trees, each correcting the errors of
  the previous one
- Strong performance on tabular data with mixed feature types
- Handles missing values natively (important for RE24 gaps and players
  with partial Statcast coverage)
- Slightly slower to train than LightGBM but well-documented for beginners
- Key hyperparameters to tune: n_estimators, max_depth, learning_rate,
  subsample, colsample_bytree

**LightGBM (Light Gradient Boosting Machine)**
- Same boosting concept as XGBoost but uses a leaf-wise tree growth
  strategy that is faster and often more accurate on large datasets
- Also handles missing values natively
- Can be more prone to overfitting on small datasets — tune
  num_leaves carefully
- Key hyperparameters to tune: num_leaves, learning_rate, n_estimators,
  min_child_samples, feature_fraction

**Head-to-Head Comparison Protocol:**
1. Train both models with default hyperparameters first (baseline)
2. Run both through cross-validation on the training set
3. Evaluate both on the validation set (2023 data)
4. Tune the better-performing model's hyperparameters using GridSearchCV
   or Optuna (beginner-friendly tuning library)
5. Re-evaluate tuned model on the test set (2024 data) — this is the
   final honest accuracy number
6. The model with lower MAE and higher Spearman rank correlation on the
   test set becomes the production model
7. Document both results transparently — do not cherry-pick

**Separate models are trained for batters and pitchers.**
A single combined model is not appropriate because the feature sets and
scoring dynamics are fundamentally different.

### 6.4 — Handling Missing Features in the Model

Both XGBoost and LightGBM handle NaN values natively — they learn
which direction to split when a value is missing, rather than requiring
imputation. This is why these two models were chosen.

Specific missing data cases:
- RE24 absent (coverage < 1.0): leave as NaN — model handles it
- Statcast absent (pre-2015 data not used, so this does not occur in
  the 2015–2023 training window)
- Players with LOW_SAMPLE flag: excluded from training entirely
- First-year players with no prior-season features: these are production
  predictions only, not training rows (no Year N-1 to look back on)

### 6.5 — Feature Importance: Expected vs Actual

After training, extract actual feature importance using SHAP
(SHapley Additive exPlanations). Compare to the expected importance
rankings documented before training begins.

**Pre-training expected importance (document these before you train):**

Batters (expected rank order):
1. xwOBA
2. Barrel%
3. wOBA / wRAA
4. BB% − K%
5. Sprint Speed (for stolen base contribution)
6. RE24 (where eligible)
7. Hard Hit%
8. OPS+
9. BABIP gap (regression signal)
10. Age

Pitchers (expected rank order):
1. SwStr% / CSW%
2. K-BB%
3. SIERA
4. Barrel% allowed
5. FIP − ERA gap (regression signal)
6. IP projection (role/workload)
7. xERA
8. LOB% deviation
9. ERA+
10. Starter/Reliever flag

**Post-training output:**
- Generate SHAP summary plot (bar chart of mean absolute SHAP values)
- Generate SHAP beeswarm plot (shows direction and magnitude per feature)
- Create side-by-side table: Expected Rank | Actual SHAP Rank | Delta
- Discuss surprises: which features outperformed expectations, which
  underperformed, and what that implies about which stats actually
  drive fantasy point production

---

## SECTION 7 — Model Evaluation: Predictions vs Actual Rankings

### 7.1 — Overview

After training and testing, the model's predictions are compared against
actual ESPN fantasy points earned in the test period (2024–2025). All
evaluation graphs are saved to `outputs/figures/model_evaluation/`.

### 7.2 — Core Accuracy Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| MAE (Mean Absolute Error) | Average point prediction error per player | Minimize |
| RMSE (Root Mean Squared Error) | Penalizes large errors more heavily | Minimize |
| Spearman Rank Correlation | How well predicted rankings match actual rankings | > 0.65 for starters |
| Top-30 Precision | % of top-30 predicted players who finished top-30 | Maximize |
| Position-Adjusted Accuracy | Same metrics, split by position | Per-position targets |

### 7.3 — Visualization: Top 50 Predicted Players (Batters + Pitchers)

Two separate graphs — one for the top 50 predicted batters, one for
the top 50 predicted pitchers. Each graph shows:

- X-axis: Predicted ESPN points rank (1–50, left to right)
- Y-axis: Actual ESPN points rank (1–50+ actual finish)
- Each player = one dot
- Dots on the diagonal = perfect prediction
- Dots above the diagonal = player outperformed prediction (breakout)
- Dots below the diagonal = player underperformed prediction (bust)
- Color coding: Green = within 10 rank spots; Yellow = within 20;
  Red = missed by more than 20 ranks
- Player names labeled for the 10 biggest misses in each direction

**Graph filenames:**
- `model_evaluation/top50_batters_predicted_vs_actual.png`
- `model_evaluation/top50_pitchers_predicted_vs_actual.png`

### 7.4 — Position-Adjusted Accuracy: Top 5 Per Position

For each position (C, 1B, 2B, 3B, SS, OF, SP, RP):
- Identify the top 5 predicted players at that position
- Show their actual finish rank at that position
- Display as a small grouped bar chart per position:
  - One bar = predicted rank
  - Adjacent bar = actual rank
  - Color the bar red if the player missed the top 5 actual, green if hit

Positions covered:
- Batters: C, 1B, 2B, 3B, SS, OF (6 position groups)
- Pitchers: SP, RP (2 groups)
- Total: 8 position charts, each showing top 5 predicted vs actual

**Graph filename:** `model_evaluation/position_accuracy_top5.png`
(all 8 positions tiled into one figure)

### 7.5 — Hit Rate by Tier (Top 30 Only)

For both batters and pitchers independently, evaluate prediction quality
across three tiers. Each tier = 10 players.

**Tier Definitions:**
- Tier 1: Predicted rank 1–10 (elite)
- Tier 2: Predicted rank 11–20 (strong)
- Tier 3: Predicted rank 21–30 (solid)

**For each tier, calculate:**
- Hit rate: % of predicted players in that tier who actually finished
  in the same tier (10-player window)
- Near-miss rate: % who finished within one tier (±10 spots)
- Miss rate: % who finished outside 2 tiers

**Graph:** Stacked bar chart per tier showing Hit / Near-Miss / Miss
proportions for batters and pitchers side by side.

**Graph filename:** `model_evaluation/tier_hit_rate_top30.png`

### 7.6 — Bust Detection (Top 30)

A bust = a player the model predicted in the top 30 who finished outside
the top 30 in actual ESPN points.

Report the top 30 predicted players for both batters and pitchers.
For each bust, record:
- Player name
- Predicted rank
- Actual rank
- Predicted ESPN points
- Actual ESPN points
- Likely cause (audit manually): injury, role change, age decline,
  regression from inflated prior-year luck stats, or model miss

**Graph:** Horizontal bar chart showing predicted vs actual points for
all busts, sorted by severity of miss. Bars colored red for actual
performance vs blue for predicted.

**Graph filename:** `model_evaluation/busts_top30_batters.png`
and `model_evaluation/busts_top30_pitchers.png`

### 7.7 — Breakout Detection (20 Players: 10 Batters + 10 Pitchers)

A breakout = a player the model did NOT predict in the top 30 who
actually finished in the top 30.

**Additional filter for breakout candidates:**
- Age ≤ 29 at the start of the season
- Not predicted in the top 30 by the model

**If fewer than 10 qualifying breakout batters or pitchers exist for a
given test year:**
- Report however many qualify (even 0)
- Log: "X breakout batters identified; Y breakout pitchers identified"
- Do not pad the list with players who don't meet the criteria

**For each breakout, record:**
- Player name
- Age
- Predicted rank
- Actual rank (top 30 finish)
- Which features signaled the breakout in hindsight
  (check SHAP values post-hoc for that player)
- Lesson: what should the model have weighted more heavily?

**Graph:** Same horizontal bar chart format as busts, but green bars.
**Graph filenames:** `model_evaluation/breakouts_batters.png`
and `model_evaluation/breakouts_pitchers.png`

---

## SECTION 8 — Draft Day Rankings Tool (2026 Season)

### 8.0 — Using 2025 Data as Input for 2026 Predictions

The draft is for the 2026 season. The model was trained on 2015–2022
and validated on 2023, with final accuracy measured against 2024
features → 2025 actuals (the test set). For the production draft tool,
2025 season data is used as the input feature set to generate 2026
projected fantasy points.

This is the standard and correct production use of a trained ML model:
- **Test set:** 2024 features → 2025 actuals (measures model accuracy)
- **Production prediction:** 2025 features → 2026 projected points (draft tool)

These are entirely separate uses. The test set validates the model.
The production prediction applies the validated model to new data.
There is no data leakage — 2025 actuals are never used as a training
target.

**What must be in place before generating draft rankings:**
1. 2025 full season data present in all Retrosheet files
   (batting.csv, pitching.csv, plays.csv, etc.)
2. 2025 Statcast data present in statcast CSVs
   (covered by the 2015–2025 pull in the prerequisite step)
3. Full stat calculation pipeline run on 2025 data — all metrics in
   Section 2 calculated for 2025 player-seasons
4. 2025 RE Matrix built (if plays.csv 2025 coverage is complete)
5. 2025 player-season feature rows assembled per Section 5
6. Trained model (best of XGBoost vs LightGBM from Section 6) applied
   to 2025 feature rows → outputs 2026 projected ESPN fantasy points

### 8.1 — Output Format

The final draft-day output is two ranked CSVs (one batters, one pitchers):

```
Draft_Rank | Player | Team | Position | Projected_ESPN_Points_2026 |
Points_per_Game_2025 | PAR_Score | Tier | Key_Stat_1 | Key_Stat_2 |
Injury_Flag | Regression_Flag | Age_Flag | Model_Confidence
```

### 8.2 — Positional Scarcity: Points Above Replacement (PAR)

Raw projected points are not the right draft currency. You must adjust
for what the player is worth RELATIVE to what you could get for free.

**Method:**
1. Project all players at each position
2. Define replacement level: the average projected points of the
   player ranked just outside a typical rostered pool at that position
   - Example: 12-team league, 1 catcher per team → replacement = #13 C
3. PAR = Projected_Points − Replacement_Level_Points_at_Position
4. Rank all players by PAR, not raw points

### 8.3 — Draft Tier Construction

Tiers are built per position using PAR score gaps:
- A tier break occurs where there is a statistically significant drop
  in PAR between adjacent players (gap > 1 standard deviation of the
  PAR distribution at that position)
- Tier 1: Elite — draft at any cost in rounds 1–4
- Tier 2: Strong — target in rounds 5–10
- Tier 3: Solid — target in rounds 11–18
- Tier 4: Streamers / waiver tier

### 8.4 — Risk Flags

| Flag | Trigger Condition |
|------|------------------|
| Age Risk | Player age ≥ 33 at season start |
| Health Risk | 2+ IL stints in prior season (from allplayers.csv) |
| Role Risk | Reliever: SV = 0 in prior season (unproven closer) |
| Regression Risk | ERA << FIP, or BABIP >> career, or xBA << AVG |
| Bounce-Back | ERA >> SIERA, or BABIP << career, or xBA >> AVG |

---

## SECTION 9 — Keeper Evaluation Module

### 9.0 — What This Module Does

You are allowed to keep 7 players from your prior roster. This module:
1. Reads a tab-separated text file of your candidate keeper players
2. Automatically looks up each player's ESPN fantasy point history
   (prior 2 full seasons) from the master player table built in Section 1
3. Calculates a Keeper Trajectory Score — a combined ranking that
   accounts for both raw point totals AND points-per-game trend
4. Outputs a ranked list of all candidates so you can make an informed
   keep/cut decision for your 7 slots
5. Applies the trained ML model to each keeper candidate's most recent
   season feature row to generate a forward-looking projected fantasy
   point total for the coming season

**On using the ML model for keeper evaluation:**

The ML model was trained on the full qualified player pool (2015–2023)
to predict next-season ESPN fantasy points. It is appropriate to apply
it to keeper candidates because it uses the same feature set as the
draft tool — xwOBA, Barrel%, SIERA, BABIP luck gaps, age, role flags,
and so on. This gives keeper evaluation the same analytical depth as
the draft rankings.

The Trajectory Score and the ML projection answer different questions:
- Trajectory Score: Has this player been improving or declining over
  the past two seasons in terms of actual fantasy point output?
- ML Projection: Given the player's underlying skill metrics this
  season, what does the model expect them to produce next season?

These two signals can diverge. A player with a strong Trajectory Score
may have a poor ML projection if their recent point gains were driven
by luck (high BABIP, inflated HR/FB%). Conversely, a player with a
weak Trajectory Score may have a strong ML projection if their
underlying metrics are healthy but their recent totals were suppressed
by injury.

**Pros of including ML in keeper evaluation:**
- Embeds regression-to-the-mean logic automatically via luck-detection
  features — the model is not fooled by an unsustainable lucky season
- Handles injury-shortened seasons better than raw point totals —
  a player who missed 60 games but was excellent when healthy will
  score poorly on Trajectory but well on ML
- Consistent methodology with the draft tool — the same model that
  ranks free agents also evaluates your keepers on equal footing
- Age curve is embedded — a 32-year-old trending up will be discounted
  relative to a 26-year-old with the same recent trend

**Cons of including ML in keeper evaluation:**
- The model was trained on the full player pool, not a 10-player
  keeper pool — the relative ranking within a small candidate set
  carries less statistical confidence than a full draft board ranking
- ML projections are less transparent than the Trajectory Score —
  a human can immediately see why a trajectory is good or bad;
  explaining why the model ranked a player 3rd vs 5th requires SHAP
- The model cannot account for off-season trades, role changes, or
  injuries that occur after the season ends — these require manual
  flag overrides in the output

**Resolution:** Both signals are computed independently and then
combined in Section 9.5 (Combined Keeper Score). Neither dominates
by default. The final ranking is transparent — both component scores
are visible in the output alongside the combined score.

### 9.1 — Input File Specification

**File type:** Tab-separated (.tsv or .txt)
**File name:** `keepers_candidates.txt`
**Location:** `data/` folder

**Required columns (exact names, tab-separated):**
```
Position    FirstName    LastName
```

**Rules:**
- Position values must match ESPN position codes: C, 1B, 2B, 3B, SS, OF, SP, RP
- You may list more than 7 candidates — the tool ranks all of them
- Accented characters, hyphens in names: supported with normalization
- If a player cannot be matched in the master table: flagged
  `LOOKUP_FAILED`, tool continues without crashing

### 9.2 — Data Lookup From Master Table

For each player in the input file:
1. Match on FirstName + LastName against allplayers.csv player IDs
2. Pull two prior season stat rows from batters_master.csv or
   pitchers_master.csv depending on Position value
3. Apply the ESPN Points Mapper (Section 4) to each season's stats
4. Output: ESPN_Points_YearN1, ESPN_Points_YearN2, Games_YearN1,
   Games_YearN2 (or IP for pitchers)

### 9.3 — Keeper Trajectory Score

**Step 1 — Points Per Game (PPG)**
```
PPG_YearN1 = ESPN_Points_YearN1 / Games_YearN1
PPG_YearN2 = ESPN_Points_YearN2 / Games_YearN2

For pitchers: PPG = ESPN_Points / IP (points per inning)
```

**Step 2 — Trend Deltas**
```
Raw_Points_Delta = ESPN_Points_YearN1 − ESPN_Points_YearN2
PPG_Delta        = PPG_YearN1 − PPG_YearN2
```

**Step 3 — Min-Max Normalization (within candidate pool only)**
```
Raw_Delta_Norm = (Raw_Delta − min) / (max − min) × 100
PPG_Delta_Norm = (PPG_Delta − min) / (max − min) × 100
```

**Step 4 — Composite Score**
```
Keeper_Trajectory_Score = (Raw_Delta_Norm × 0.5) + (PPG_Delta_Norm × 0.5)
```

Equal 50/50 weighting. Adjustable by user to reward durability (raise
Raw weight) or efficiency (raise PPG weight).

### 9.4 — Keeper Trajectory Scatter Plots

Two scatter plots generated and saved to
`outputs/figures/keeper_trajectories/`:

**Plot 1 — Raw Points Trend**
- X-axis: ESPN Points Year N-2
- Y-axis: ESPN Points Year N-1
- One dot per player candidate
- Diagonal reference line = no change
- Dots above line = improved; below = declined
- Players labeled by last name
- Color by position
- OLS trend line across all candidates with confidence band

**Plot 2 — PPG Trend**
- X-axis: PPG Year N-2
- Y-axis: PPG Year N-1
- Same format as Plot 1
- This plot highlights players who improved efficiency even if raw
  totals dropped due to injury/reduced playing time

**Graph filenames:**
- `keeper_trajectories/keeper_raw_points_trend.png`
- `keeper_trajectories/keeper_ppg_trend.png`

### 9.5 — Combined Keeper Score

The Combined Keeper Score merges the Trajectory Score (Section 9.3)
and the ML Projected Points (Section 9.0, step 5) into a single
final ranking number.

**Step 1 — Normalize ML Projected Points (within candidate pool):**
```
ML_Norm = (ML_Projected_Points − min(ML_Projected_Points))
          / (max(ML_Projected_Points) − min(ML_Projected_Points))
          × 100
```

This places each player's ML projection on a 0–100 scale relative
to the other keeper candidates — the same scale as the Trajectory
Score.

**Step 2 — Combine:**
```
Combined_Keeper_Score = (Keeper_Trajectory_Score × 0.5)
                      + (ML_Norm × 0.5)
```

Equal 50/50 weighting by default. Both weights are adjustable:
- Raise Trajectory weight if you trust recent point history more
  than model projections (e.g., stable veteran players)
- Raise ML weight if you want regression signals to dominate
  (e.g., you suspect a player had an unsustainable lucky season)

**Step 3 — Final Keeper Rank:**
Players are ranked 1 to N by Combined_Keeper_Score descending.
The top 7 by Combined_Keeper_Score are the tool's recommended keepers
before any manual flag overrides are applied.

**When the two signals diverge significantly (gap > 25 points):**
Flag the player as `REVIEW` in the output. Large divergences indicate
a meaningful disagreement between recent history and underlying skill
projections — these players warrant a manual look before deciding.

---

### 9.6 — Output: Keeper Rankings Report

**File:** `outputs/keeper_rankings.csv`

```
Keeper_Rank | Player | Position | ESPN_Points_YearN2 |
ESPN_Points_YearN1 | Raw_Points_Delta | PPG_YearN2 |
PPG_YearN1 | PPG_Delta | Keeper_Trajectory_Score |
ML_Projected_Points | ML_Norm | Combined_Keeper_Score |
Signal_Divergence_Flag | Regression_Flag | Injury_Flag | Recommendation
```

**New columns vs prior version:**

| Column | Description |
|--------|-------------|
| ML_Projected_Points | Raw projected ESPN points from the trained model |
| ML_Norm | ML projection normalized 0–100 within the candidate pool |
| Combined_Keeper_Score | Final composite score (Trajectory 50% + ML 50%) |
| Signal_Divergence_Flag | REVIEW if Trajectory and ML_Norm differ by > 25 pts |

### 9.7 — Recommendation Logic

Applied in order, first match wins. Uses Combined_Keeper_Score:

```
IF Combined_Keeper_Score >= 75 AND no Injury_Flag:
    → AUTO-KEEP

ELSE IF Combined_Keeper_Score >= 50 AND ML_Projected_Points >= positional_average:
    → KEEP

ELSE IF Combined_Keeper_Score >= 35 OR ML_Projected_Points in top half at position:
    → BORDERLINE

ELSE:
    → CUT
```

Players flagged `REVIEW` (signal divergence > 25) always display
both component scores prominently in the output so the decision
is informed, not hidden.

### 9.8 — Edge Cases

| Situation | Handling |
|-----------|---------|
| Rookie — only 1 year of data | Trajectory Score uses PPG only; Raw Delta = N/A |
| Mid-season team change | Combined season totals used; note appended |
| Position switch (e.g., OF → DH) | Input file position used for scoring |
| Injury season (< threshold PA/IP) | PPG weighted more heavily; LOW_SAMPLE flagged |
| SP → RP conversion | Role flag added; IP-based PPG recalculated per role |
| Name lookup fails | Row flagged LOOKUP_FAILED; manual override column provided |
| ML model cannot score player (missing features) | ML_Norm = N/A; Combined Score uses Trajectory only; flagged |

### 9.9 — Console Output Example

```
=======================================================================
  KEEPER EVALUATION REPORT
=======================================================================
  Candidates evaluated: 10    Keeper slots: 7

  RNK  PLAYER               POS  TRAJ   ML_PTS  ML_NRM  COMBINED  REC
  ---  -------------------  ---  -----  ------  ------  --------  ---------
  1    Alvarez, Yordan      OF   91.2   510     96.1    93.7      AUTO-KEEP
  2    Freeman, Freddie     1B   83.7   455     88.4    86.1      AUTO-KEEP
  3    Witt, Bobby          SS   79.4   435     82.1    80.8      AUTO-KEEP
  4    Soto, Juan           OF   74.1   420     76.3    75.2      AUTO-KEEP
  5    Cole, Gerrit         SP   70.3   400     71.5    70.9      KEEP
  6    Semien, Marcus       2B   63.8   355     60.2    62.0      KEEP
  7    Riley, Austin        3B   55.2   340     55.8    55.5      KEEP
  8    Contreras, William   C    48.1   300     44.3    46.2      BORDERLINE
  9    Nootbaar, Lars       OF   38.9   275     35.1    37.0      BORDERLINE  ⚑ REVIEW
  10   Díaz, Alexis         RP   22.4   205     18.6    20.5      CUT

  ⚑ REVIEW flags indicate Trajectory vs ML_Norm gap > 25 points.
    Nootbaar: Trajectory 38.9 vs ML_Norm 35.1 — gap within threshold,
    flagged for low combined score near BORDERLINE/CUT boundary.

  Suggested keeps (slots 1–7): Alvarez, Freeman, Witt, Soto, Cole, Semien, Riley
=======================================================================
```

---

## APPENDIX A — Key Stat Correlation Cheat Sheet

Stats known to correlate with future fantasy point production.
These are not assumptions — each has published baseball research support.

| Stat | Predicts | Strength |
|------|----------|----------|
| xwOBA | Future OBP + SLG events | Very Strong |
| Barrel% | Future HR rate | Strong |
| K-BB% (pitchers) | Future ERA / K totals | Very Strong |
| SIERA | Future ERA | Stronger than ERA itself |
| SwStr% | Future K rate | Very Strong |
| Sprint Speed | SB conversion | Strong |
| LOB% deviation from mean | ERA regression direction | Strong |
| BABIP deviation from career | AVG regression direction | Strong |
| HR/FB% deviation from career | HR regression direction | Moderate–Strong |
| Hard Hit% allowed | Future ERA | Moderate–Strong |
| FIP − ERA gap | ERA regression direction | Strong |
| xBA − AVG | Next-year AVG direction | Moderate–Strong |

---

## APPENDIX B — Tools & Libraries Reference

| Library | Purpose | Install |
|---------|---------|---------|
| pandas | Data manipulation | pip install pandas |
| numpy | Numerical calculations | pip install numpy |
| pybaseball | Statcast data pull from Baseball Savant | pip install pybaseball |
| xgboost | Primary ML model candidate | pip install xgboost |
| lightgbm | Primary ML model candidate | pip install lightgbm |
| scikit-learn | Preprocessing, metrics, cross-validation | pip install scikit-learn |
| shap | Model explainability (feature importance) | pip install shap |
| optuna | Hyperparameter tuning | pip install optuna |
| matplotlib | Charting | pip install matplotlib |
| seaborn | Statistical visualization | pip install seaborn |
| jupyter | Notebook interface | pip install jupyter |

---

## APPENDIX C — Key Decisions Log

This log records every deliberate design choice made during planning,
so that future-you knows why things were built the way they were.

| Decision | Choice Made | Reason |
|----------|-------------|--------|
| Primary data source | Retrosheet raw files | Full control, no FanGraphs dependency |
| Statcast source | pybaseball → Baseball Savant | Standalone prerequisite script, not in main pipeline |
| RE24 missing data | Exclude from ML if coverage < 100% | Scaled RE24 is not RE24; precision over completeness |
| Training window | 2015–2023 | Aligns with Statcast; avoids pre-tracking-era gap |
| Test window | 2024–2025 | Strict future holdout; no leakage |
| Production prediction input | 2025 features → 2026 projected points | Standard ML production use; separate from test set |
| Batter PA threshold | 200 PA (100 in 2020) | Reliable rate stats; 2020 exception for 60-game season |
| Pitcher IP threshold | 45 IP | Reliable rate stats |
| Park factor method | 5-year rolling average | Smooths single-season noise; new parks use all available seasons |
| Batted ball codes | G/F/L/P from plays.csv batted_ball_type | Renamed to GB/FB/LD/PU as derived count variables |
| FIP league constant name | FIP_League | Clarifies it is a league-level constant, not player-level |
| Model type | XGBoost vs LightGBM, best wins | Both handle NaN natively; best accuracy on tabular data |
| Feature importance language | Removed pre-assigned labels | SHAP determines actual importance after training |
| Closer flag threshold | SV > 3 in prior season | More reliable than SV > 0 for established closer role |
| Pitcher scoring | IP×3, K×1, W×5, SV×5, HD×2, ER×−2, H×−1, BB×−1, L×−2 | Confirmed 2026 league scoring. No CG/SHO/NH/HBP. |
| Batter scoring | TB×1, R×1, RBI×1, BB×1, SB×1, K×-1 | Confirmed 2026 league scoring. TB-based. No HBP, no CS penalty. |
| Trajectory score weighting | 50% raw points / 50% PPG | Equal credit to volume and efficiency |
| Keeper ML integration | ML projection added as step 5 in Section 9.0 | Uses full feature set; handles injury seasons better than raw totals |
| Combined Keeper Score | (Trajectory × 0.5) + (ML_Norm × 0.5) | ML_Norm scaled 0–100 within candidate pool before combining |
| Signal divergence flag | REVIEW if Trajectory vs ML_Norm gap > 25 | Surfaces meaningful disagreements for human review |
| wOBA weights | Derived from RE Matrix each season | Self-contained from plays.csv; no external table dependency |
| wRAA formula | (wOBA − lg_wOBA) / wOBA_scale × PA | Confirmed standard form; 0 = league average always |
| wRC+ | Added to batter stat set | Park + league adjusted offensive metric; 100 = average |
| SIERA formula | Confirmed standard published form | All inputs calculable from plays.csv + pitching.csv |
| oWAR (v1) | Fielding_Runs = 0; labeled oWAR | Defensive model deferred to v2; offensive components fully confirmed |
| WAR baserunning | UBR + wSB + wGDP from RE Matrix | All three components derived from plays.csv |
| Positional adjustment | Standard published values per 150G | C:+12.5, SS:+7.5, 2B/3B/CF:+2.5, LF/RF:−7.5, 1B:−12.5, DH:−17.5 |
| Replacement level | −20 runs per 600 PA | Standard published value confirmed |
| Breakout age filter | Age ≤ 29 at season start | True breakouts are young; older players are known quantities |
| Breakout threshold | Top 30 finish (not in top 30 predicted) | Focused on actionable surprises |
| Bust / tier analysis | Top 30 only, 3 tiers of 10 | Actionable draft-range focus |

---

*Version 2 — updated to reflect all confirmed design decisions.
Rerun the full pipeline at the start of each new season with the
prior season's data added. Model accuracy compounds with more training rows.*
