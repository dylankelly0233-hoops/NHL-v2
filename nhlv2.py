import streamlit as st
import pandas as pd
import requests
import numpy as np
import difflib
from datetime import datetime
from bs4 import BeautifulSoup
import re

# --- CONFIGURATION ---
LEAGUE_AVG_TOTAL = 6.2
HOME_ICE_ADVANTAGE = 0.2  # Adjustment for home team strength

# ⬇️ PASTE YOUR API KEY HERE (keep it inside the quotes) ⬇️
ODDS_API_KEY = "48179aea30c6c0450cd29f216d62e34e" 

# --- 1. DATA FETCHING ---

@st.cache_data(ttl=3600)
def get_schedule(date_str):
    """Fetches the schedule and team details for the given date."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    try:
        r = requests.get(url).json()
        games = []
        for day in r.get('gameWeek', []):
            if day['date'] == date_str:
                for game in day['games']:
                    games.append({
                        'home_team': game['homeTeam']['abbrev'],
                        'away_team': game['awayTeam']['abbrev'],
                        'home_id': game['homeTeam']['id'],
                        'away_id': game['awayTeam']['id'],
                        'home_name': game['homeTeam']['placeName']['default'],
                        'away_name': game['awayTeam']['placeName']['default']
                    })
        return games
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []

@st.cache_data(ttl=3600)
def get_projected_starters():
    """Scrapes DailyFaceoff for projected starters."""
    url = "https://www.dailyfaceoff.com/starting-goalies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    starters = {}
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        matchups = soup.find_all('div', class_='starting-goalies_matchup')
        
        for match in matchups:
            teams = match.find_all('span', class_='logo_ticker')
            goalie_cards = match.find_all('h4', class_='name')
            
            if len(teams) >= 2 and len(goalie_cards) >= 2:
                away_team = teams[0].text.strip()
                home_team = teams[1].text.strip()
                
                # Clean messy JSON names if they appear
                def clean_name(raw_text):
                    if "default" in raw_text:
                        matches = re.findall(r"['\"]default['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw_text)
                        if matches: return " ".join(matches)
                    return raw_text

                starters[away_team] = clean_name(goalie_cards[0].text.strip())
                starters[home_team] = clean_name(goalie_cards[1].text.strip())
        return starters
    except Exception as e:
        print(f"Error scraping starters: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_vegas_odds(api_key, region='us', market='h2h,totals'):
    """
    Fetches odds from The Odds API. 
    """
    if not api_key:
        return {}
    
    url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?apiKey={api_key}&regions={region}&markets={market}&oddsFormat=american"
    try:
        r = requests.get(url).json()
        odds_map = {}
        for game in r:
            home_team = game.get('home_team')
            # Initialize defaults
            if home_team not in odds_map:
                odds_map[home_team] = {'total': 6.5, 'home_ml': -110, 'away_ml': -110}

            for bookmaker in game.get('bookmakers', []):
                found_total = False
                found_h2h = False
                
                for market_data in bookmaker.get('markets', []):
                    # Parse Totals
                    if market_data['key'] == 'totals' and not found_total:
                        if len(market_data['outcomes']) > 0:
                            line = market_data['outcomes'][0].get('point')
                            odds_map[home_team]['total'] = line
                            found_total = True
                    
                    # Parse Moneyline (h2h)
                    if market_data['key'] == 'h2h' and not found_h2h:
                        for outcome in market_data['outcomes']:
                            if outcome['name'] == home_team:
                                odds_map[home_team]['home_ml'] = outcome['price']
                            else:
                                odds_map[home_team]['away_ml'] = outcome['price']
                        found_h2h = True
                
                if found_total and found_h2h:
                    break
        return odds_map
    except Exception as e:
        st.warning(f"Could not fetch Vegas odds (Check API Key): {e}")
        return {}

@st.cache_data(ttl=86400)
def get_all_roster_goalies(teams_playing):
    """
    Iterates through the teams playing TODAY and fetches their full active roster.
    This ensures backups (like Quick) are found even if they aren't stats leaders.
    """
    goalies = []
    
    # 1. Fetch Stats Leaders (Base Data for GSAx)
    stats_map = {}
    try:
        # limit=-1 gets ALL goalies with stats
        stats_url = "https://api-web.nhle.com/v1/goalie-stats-leaders/current?categories=goalsAgainstAverage&limit=-1"
        r = requests.get(stats_url).json()
        for g in r.get('goalsAgainstAverage', []):
            name_key = f"{g['firstName']} {g['lastName']}"
            stats_map[name_key] = g['value'] # Store GAA for sim
    except:
        pass

    # 2. Fetch Rosters for active teams
    for team_abbr in teams_playing:
        try:
            roster_url = f"https://api-web.nhle.com/v1/roster/{team_abbr}/current"
            r = requests.get(roster_url).json()
            
            # The roster is split into categories usually
            roster_goalies = r.get('goalies', [])
            
            for g in roster_goalies:
                name = f"{g['firstName']['default']} {g['lastName']['default']}"
                
                # Try to get real GAA, otherwise simulate based on "Backup" status
                if name in stats_map:
                    gaa = stats_map[name]
                    # Simple GSAx Simulation based on GAA
                    if gaa < 2.5: gsax = round(np.random.uniform(0.3, 0.9), 2)
                    elif gaa < 3.0: gsax = round(np.random.uniform(-0.1, 0.3), 2)
                    else: gsax = round(np.random.uniform(-0.6, -0.1), 2)
                else:
                    # No stats found? Likely a backup/callup
                    gaa = 3.00
                    gsax = -0.25 # Slight negative penalty for unknowns
                    
                goalies.append({'Name': name, 'Team': team_abbr, 'GSAx': gsax})
                
        except Exception as e:
            # Fallback if roster fails
            pass
            
    return pd.DataFrame(goalies)

def reconcile_starters(starters_dict, goalie_df):
    if goalie_df.empty: return starters_dict, goalie_df
    official_names = goalie_df['Name'].tolist()
    final_starters = {}
    new_rows = []
    
    for team, scraped_name in starters_dict.items():
        # 1. Exact Match
        if scraped_name in official_names:
            final_starters[team] = scraped_name
        else:
            # 2. Fuzzy Match
            matches = difflib.get_close_matches(scraped_name, official_names, n=1, cutoff=0.6)
            if matches:
                final_starters[team] = matches[0]
            else:
                # 3. Force Add (if scraper found someone roster didn't)
                new_rows.append({'Name': scraped_name, 'Team': team, 'GSAx': 0.00})
                final_starters[team] = scraped_name
                
    if new_rows:
        goalie_df = pd.concat([goalie_df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Generic Fallbacks
    goalie_df = pd.concat([goalie_df, pd.DataFrame([
        {'Name': 'Average Goalie', 'Team': 'NHL', 'GSAx': 0.00},
        {'Name': 'Backup/Rookie', 'Team': 'NHL', 'GSAx': -0.40}
    ])], ignore_index=True)
    
    return final_starters, goalie_df.drop_duplicates(subset=['Name']).sort_values('Name')

def get_simulated_ratings(active_teams):
    data = []
    for t in active_teams:
        # Placeholder for your real team ratings
        off_rating = np.random.uniform(2.9, 3.4) 
        def_rating = np.random.uniform(2.9, 3.4) 
        data.append({'team': t, 'off_rating': off_rating, 'def_rating': def_rating})
    return pd.DataFrame(data).set_index('team')

def match_vegas_odds(home_team_name, odds_map):
    """Fuzzy matches NHL API team name to Odds API team name."""
    default = {'total': 6.5, 'home_ml': -110, 'away_ml': -110}
    if not odds_map: return default
    
    # Exact match check
    if home_team_name in odds_map:
        return odds_map[home_team_name]
    
    # Fuzzy match check (e.g. "NY Rangers" vs "New York Rangers")
    keys = list(odds_map.keys())
    matches = difflib.get_close_matches(home_team_name, keys, n=1, cutoff=0.5)
    if matches:
        return odds_map[matches[0]]
    
    return default

# --- 2. MATH HELPERS ---

def implied_probability(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def probability_to_american(probability):
    if probability <= 0.5:
        return int(round(-100 * ((1 / probability) - 1))) if probability > 0 else 20000
    else:
        return int(round((probability / (1 - probability)) * -100)) if probability < 1 else -20000

def calculate_moneyline_projection(h_stats, a_stats, h_gsax, a_gsax):
    # Base expected goals
    exp_h_goals = (h_stats['off_rating'] + a_stats['def_rating']) / 2
    exp_a_goals = (a_stats['off_rating'] + h_stats['def_rating']) / 2
    
    # Home Ice Adjustment
    exp_h_goals += HOME_ICE_ADVANTAGE / 2
    exp_a_goals -= HOME_ICE_ADVANTAGE / 2
    
    # Goalie Impact
    h_strength = exp_h_goals + (h_gsax * 0.5) 
    a_strength = exp_a_goals + (a_gsax * 0.5) 
    
    # Pythagorean Expectation
    exponent = 2.0
    home_win_prob = (h_strength ** exponent) / ((h_strength ** exponent) + (a_strength ** exponent))
    
    return home_win_prob

def get_gsax(goalie_name, goalie_df):
    try:
        return goalie_df.loc[goalie_df['Name'] == goalie_name, 'GSAx'].values[0]
    except:
        return 0.0

# --- 3. MAIN APP ---

def main():
    st.set_page_config(page_title="NHL Edge Finder", page_icon="🏒", layout="wide")
    
    st.title("🏒 NHL Edge Finder: Totals & Moneyline")

    # -- Sidebar --
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_date = st.date_input("Game Date", datetime.now())
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.divider()
        st.subheader("💰 Vegas Integration")
        
        # KEY LOGIC: Check code variable first, then text input
        if ODDS_API_KEY and len(ODDS_API_KEY) > 5:
            api_key = ODDS_API_KEY
            st.success(f"API Key Loaded! 🔒")
        else:
            api_key = st.text_input("The Odds API Key", type="password")
        
        st.divider()
        st.subheader("🎯 Betting Strategy")
        totals_edge_threshold = st.slider("Min Edge (Totals)", 0.1, 1.5, 0.5, 0.1)
        ml_edge_threshold = st.slider("Min Edge (Moneyline %)", 1.0, 15.0, 5.0, 0.5)
        
        load_btn = st.button("🚀 Run Model", type="primary")

    # -- LOGIC --
    if load_btn:
        with st.spinner("Fetching Rosters, Odds & Schedule..."):
            games = get_schedule(date_str)
            if not games:
                st.error("No games found for this date.")
                st.session_state['data_loaded'] = False
            else:
                # 1. Get Projected Starters (DailyFaceoff)
                starters_dict = get_projected_starters()
                
                # 2. Get ALL Goalies from Rosters of playing teams (Fixes Backup/Quick issue)
                teams_playing = set([g['home_team'] for g in games] + [g['away_team'] for g in games])
                raw_goalies = get_all_roster_goalies(teams_playing)
                
                final_starters, final_goalie_db = reconcile_starters(starters_dict, raw_goalies)
                ratings = get_simulated_ratings(teams_playing)
                
                # 3. Get Vegas Odds
                vegas_odds = get_vegas_odds(api_key) if api_key else {}
                
                st.session_state['games'] = games
                st.session_state['starters'] = final_starters
                st.session_state['goalie_db'] = final_goalie_db
                st.session_state['ratings'] = ratings
                st.session_state['vegas_odds'] = vegas_odds
                st.session_state['data_loaded'] = True
                
                if not api_key:
                    st.warning("⚠️ No API Key provided. You will need to input Vegas lines manually.")

    # -- DASHBOARD --
    if st.session_state.get('data_loaded', False):
        games = st.session_state['games']
        goalie_db = st.session_state['goalie_db']
        ratings = st.session_state['ratings']
        starters = st.session_state['starters']
        vegas_odds = st.session_state['vegas_odds']
        
        goalie_names = goalie_db['Name'].tolist()

        st.subheader(f"📊 Market Analysis for {date_str}")
        
        for game in games:
            home = game['home_team']
            home_full = game['home_name']
            away = game['away_team']
            gid = game['home_id']

            # Get Stats
            try:
                h_stats = ratings.loc[home]
                a_stats = ratings.loc[away]
                base_total = (h_stats['off_rating'] + a_stats['def_rating'])/2 + \
                             (a_stats['off_rating'] + h_stats['def_rating'])/2
            except:
                h_stats, a_stats = None, None
                base_total = LEAGUE_AVG_TOTAL

            # Get Vegas
            vegas_data = match_vegas_odds(home_full, vegas_odds)
            auto_total = vegas_data.get('total', 6.5)
            auto_h_ml = vegas_data.get('home_ml', -110)
            auto_a_ml = vegas_data.get('away_ml', -110)
            
            with st.container(border=True):
                st.markdown(f"#### {away} @ {home}")
                c1, c2, c3 = st.columns([1.5, 1.2, 1.3])
                
                # --- GOALIES ---
                a_start = starters.get(away, "Average Goalie")
                h_start = starters.get(home, "Average Goalie")
                
                # Safe Indexing
                try: a_idx = goalie_names.index(a_start)
                except: a_idx = goalie_names.index("Average Goalie") if "Average Goalie" in goalie_names else 0
                try: h_idx = goalie_names.index(h_start)
                except: h_idx = goalie_names.index("Average Goalie") if "Average Goalie" in goalie_names else 0

                with c1:
                    st.caption("🥅 Goaltending")
                    sel_a_goalie = st.selectbox(f"{away} G", goalie_names, index=a_idx, key=f"a_{gid}")
                    sel_h_goalie = st.selectbox(f"{home} G", goalie_names, index=h_idx, key=f"h_{gid}")
                    a_gsax = get_gsax(sel_a_goalie, goalie_db)
                    h_gsax = get_gsax(sel_h_goalie, goalie_db)

                # --- TOTALS ---
                my_total_proj = base_total - h_gsax - a_gsax
                
                with c2:
                    st.caption("📊 Totals")
                    # Vegas input defaults to 'auto_total' which comes from API
                    vegas_line = st.number_input("Line", value=float(auto_total), step=0.5, key=f"v_{gid}")
                    
                    total_edge = my_total_proj - vegas_line
                    st.metric("Projected", f"{my_total_proj:.2f}")
                    
                    if abs(total_edge) >= totals_edge_threshold:
                        if total_edge > 0: st.success(f"**OVER** {vegas_line} (+{abs(total_edge):.2f})")
                        else: st.error(f"**UNDER** {vegas_line} ({total_edge:.2f})")
                    else:
                        st.caption(f"No Total Edge ({total_edge:.2f})")

                # --- MONEYLINE ---
                with c3:
                    st.caption("💵 Moneyline")
                    col_h, col_a = st.columns(2)
                    with col_h:
                        v_h_ml = st.number_input(f"{home} ML", value=int(auto_h_ml), step=10, key=f"ml_h_{gid}")
                    with col_a:
                        v_a_ml = st.number_input(f"{away} ML", value=int(auto_a_ml), step=10, key=f"ml_a_{gid}")

                    if h_stats is not None:
                        proj_h_prob = calculate_moneyline_projection(h_stats, a_stats, h_gsax, a_gsax)
                        proj_a_prob = 1 - proj_h_prob
                        
                        st.write(f"**My Win %:** {home} {proj_h_prob*100:.1f}%")

                        implied_h = implied_probability(v_h_ml)
                        implied_a = implied_probability(v_a_ml)
                        
                        edge_h = (proj_h_prob - implied_h) * 100
                        edge_a = (proj_a_prob - implied_a) * 100
                        
                        if edge_h >= ml_edge_threshold: st.success(f"**BET {home}** (+{edge_h:.1f}%)")
                        elif edge_a >= ml_edge_threshold: st.success(f"**BET {away}** (+{edge_a:.1f}%)")
                        else: st.caption("No ML Value")

if __name__ == "__main__":
    main()
