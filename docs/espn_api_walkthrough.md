Step-by-Step Guide: Pulling ESPN Fantasy Baseball Data with espn-api
This guide uses the cwendt94/espn-api Python package (890+ stars, last updated March 2026) to pull fantasy baseball data from your ESPN league — no Selenium, no browser automation, no broken xpaths. Just clean Python API calls.

Step 1 — Install the Package
pip install espn_api
That's it. No Chrome, no Selenium, no webdriver-manager needed.

Requirements: Python ≥ 3.8

Step 2 — Find Your League ID
Log in to ESPN Fantasy Baseball
Navigate to your league
Look at the URL — it will look like:
4.  https://fantasy.espn.com/baseball/league?leagueId=12345678
Your league ID is the number after leagueId= (e.g., 12345678)
Step 3 — Get Your Authentication Cookies (Private Leagues Only)
If your league is public, skip to Step 4. For private leagues, you need two cookies:

Open espn.com in Chrome while logged in
Open DevTools → Application tab → Cookies → espn.com
Find and copy these two values:
SWID — looks like {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
espn_s2 — a very long alphanumeric string
Tip: Even for public leagues, logging in enables the recent_activity() feature.

Step 4 — Connect to Your League
from espn_api.baseball import League
 
# Public league
league = League(league_id=12345678, year=2025)
 
# Private league
league = League(
    league_id=12345678,
    year=2025,
    espn_s2='YOUR_LONG_ESPN_S2_COOKIE_HERE',
    swid='{YOUR-SWID-HERE}'
)
Once initialized, the league object is fully loaded with all your league data.

Step 5 — Explore Your League Data
5.1 League Info
# Basic league properties
print(league.current_week)          # Current scoring period
print(league.currentMatchupPeriod)  # Current matchup period
print(league.scoring_type)          # e.g., 'H2H_CATEGORY', 'H2H_POINTS'
5.2 Teams & Standings
# All teams
for team in league.teams:
    print(f"{team.team_name}: {team.wins}-{team.losses}-{team.ties} (Standing: {team.standing})")
 
# Sorted standings
for team in league.standings():
    print(f"{team.standing}. {team.team_name} ({team.wins}-{team.losses})")
5.3 Team Rosters — Every Player on Every Team
for team in league.teams:
    print(f"\n=== {team.team_name} ===")
    for player in team.roster:
        print(f"  {player.name} | {player.position} | {player.proTeam} | "
              f"Lineup: {player.lineupSlot} | Injury: {player.injuryStatus}")


5.4 Scoreboard — Current Matchups
# Current matchup period
scoreboard = league.scoreboard()
for matchup in scoreboard:
    print(f"{matchup.home_team.team_name} vs {matchup.away_team.team_name}")
    print(f"  Home Score: {matchup.home_score}  |  Away Score: {matchup.away_score}")
 
# Specific matchup period
scoreboard_wk3 = league.scoreboard(matchupPeriod=3)
5.5 Free Agents
# Top 50 free agents by % owned
free_agents = league.free_agents(size=50)
for fa in free_agents:
    print(f"{fa.name} | {fa.position} | {fa.proTeam} | "
          f"Owned: {fa.percent_owned}% | Points: {fa.total_points}")
 
# Filter by position
catchers = league.free_agents(position='C', size=25)
starting_pitchers = league.free_agents(position='SP', size=25)
outfielders = league.free_agents(position='OF', size=25)
Available position filters: C, 1B, 2B, 3B, SS, OF, LF, CF, RF, DH, UTIL, P, SP, RP, IF, 2B/SS, 1B/3B

5.6 Box Scores — Detailed Player Stats per Matchup
# Current matchup period
box_scores = league.box_scores()
 
# Specific matchup period
box_scores = league.box_scores(matchup_period=5, scoring_period=5)
 
for box in box_scores:
    print(f"\n{box.home_team.team_name} vs {box.away_team.team_name}")
    print("  Home Lineup:")
    for player in box.home_lineup:
        print(f"    {player.name} ({player.lineupSlot}): {player.points}")
5.7 Recent Activity — Adds, Drops, Trades
# Last 25 transactions
activity = league.recent_activity(size=25)
for act in activity:
    print(act)
 
# Filter by type
adds = league.recent_activity(msg_type='FA')
waivers = league.recent_activity(msg_type='WAIVER')
trades = league.recent_activity(msg_type='TRADED')
Step 6 — Pull Season Stats for All Players
Each Player object has a .stats dictionary keyed by scoring period ID. Period 0 contains season totals.

import pandas as pd
 
all_players = []
 
for team in league.teams:
    for player in team.roster:
        # Season totals are at scoring period 0
        season = player.stats.get(0, {})
        breakdown = season.get('breakdown', {})
 
        all_players.append({
            'Team': team.team_name,
            'Player': player.name,
            'Position': player.position,
            'MLB_Team': player.proTeam,
            'Lineup_Slot': player.lineupSlot,
            'Total_Points': player.total_points,
            'Projected_Points': player.projected_total_points,
            'Pct_Owned': player.percent_owned,
            # Batting stats
            'AB': breakdown.get('AB', 0),
            'H': breakdown.get('H', 0),
            'HR': breakdown.get('HR', 0),
            'R': breakdown.get('R', 0),
            'RBI': breakdown.get('RBI', 0),
            'SB': breakdown.get('SB', 0),
            'AVG': breakdown.get('AVG', 0),
            'OBP': breakdown.get('OBP', 0),
            'OPS': breakdown.get('OPS', 0),
            'B_BB': breakdown.get('B_BB', 0),
            'B_SO': breakdown.get('B_SO', 0),
            # Pitching stats
            'W': breakdown.get('W', 0),
            'L': breakdown.get('L', 0),
            'ERA': breakdown.get('ERA', 0),
            'WHIP': breakdown.get('WHIP', 0),
            'K': breakdown.get('K', 0),
            'SV': breakdown.get('SV', 0),
            'HLD': breakdown.get('HLD', 0),
            'QS': breakdown.get('QS', 0),
            'K/9': breakdown.get('K/9', 0),
        })
 
df = pd.DataFrame(all_players)
df.to_csv('fantasy_baseball_season_stats.csv', index=False)
print(f"Exported {len(df)} players to fantasy_baseball_season_stats.csv")
All available stat keys (from the source code):


Step 7 — Pull Stats by Scoring Period (Week-by-Week)
# Loop through all scoring periods for a specific player
for team in league.teams:
    for player in team.roster:
        if player.name == "Shohei Ohtani":
            print(f"\n{player.name} — Week-by-Week Stats:")
            for period, stats in sorted(player.stats.items()):
                if period == 0:
                    continue  # skip season totals
                pts = stats.get('points', 0)
                proj = stats.get('projected_points', 0)
                print(f"  Period {period}: {pts} pts (projected: {proj})")
Step 8 — Loop Through Multiple Seasons
seasons = {}
 
for year in [2021, 2022, 2023, 2024, 2025]:
    try:
        lg = League(
            league_id=12345678,
            year=year,
            espn_s2='YOUR_ESPN_S2',
            swid='{YOUR-SWID}'
        )
        seasons[year] = lg
        print(f"{year}: {len(lg.teams)} teams loaded")
    except Exception as e:
        print(f"{year}: Failed — {e}")
Note: Historical data is only available for 2018+ seasons via the API. Pre-2018 data may return errors.

Step 9 — Export Free Agents to CSV
import pandas as pd
 
fa_list = []
for pos in ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP', 'DH']:
    agents = league.free_agents(position=pos, size=100)
    for fa in agents:
        season = fa.stats.get(0, {})
        breakdown = season.get('breakdown', {})
        fa_list.append({
            'Player': fa.name,
            'Position': fa.position,
            'MLB_Team': fa.proTeam,
            'Pct_Owned': fa.percent_owned,
            'Total_Points': fa.total_points,
            'Projected_Points': fa.projected_total_points,
            'HR': breakdown.get('HR', 0),
            'RBI': breakdown.get('RBI', 0),
            'R': breakdown.get('R', 0),
            'SB': breakdown.get('SB', 0),
            'AVG': breakdown.get('AVG', 0),
            'W': breakdown.get('W', 0),
            'ERA': breakdown.get('ERA', 0),
            'WHIP': breakdown.get('WHIP', 0),
            'K': breakdown.get('K', 0),
            'SV': breakdown.get('SV', 0),
        })
 
fa_df = pd.DataFrame(fa_list).drop_duplicates(subset='Player')
fa_df.to_csv('free_agents.csv', index=False)
print(f"Exported {len(fa_df)} free agents")
Step 10 — Debug Mode
If something isn't working, enable debug mode to see the raw ESPN API requests and responses:

league = League(league_id=12345678, year=2025, debug=True)
This prints all HTTP requests and JSON responses to the console. Pipe to a file if needed:

python your_script.py > debug_output.txt 2>&1
Quick Reference
from espn_api.baseball import League
 
league = League(league_id=YOUR_ID, year=2025, espn_s2='...', swid='{...}')
 
league.teams                        # List of Team objects
league.standings()                  # Teams sorted by standing
league.scoreboard(matchupPeriod=N)  # Matchups for period N
league.free_agents(position='SP')   # Free agents filtered by position
league.box_scores(matchup_period=N) # Box scores for period N
league.recent_activity(size=25)     # Recent adds/drops/trades
 
team.roster                         # List of Player objects on team
team.schedule                       # List of Matchup objects
team.wins / team.losses / team.ties # Record
 
player.stats[0]['breakdown']        # Season total stat breakdown dict
player.stats[N]['points']           # Points for scoring period N
player.total_points                 # Shortcut for season total points
⚠️ Known Limitations
Issue

Details

Baseball is "in development"

The wiki docs say "Doc Coming Soon!" — the module works but documentation is sparse. Use the source code and this guide as reference.

In-progress seasons

Some users report issues pulling data for active/current baseball seasons. Completed seasons work reliably.

Pre-2018 data

Historical data before 2018 may not be accessible via the API.

Undocumented ESPN API

ESPN can change or break the underlying API at any time. The package maintainers have historically adapted quickly (last release: March 2026).

Free agents require 2019+

The free_agents() and box_scores() methods raise exceptions for years before 2019.