import streamlit as st
import pandas as pd
import requests
import numpy as np
import difflib
import io
import re
from datetime import datetime
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
LEAGUE_AVG_TOTAL = 6.2
HOME_ICE_ADVANTAGE = 0.2  # Adjustment for home team strength

# --- 1. DATA FETCHING (Cached) ---

@st.cache_data(ttl=3600)
def get_schedule(date_str):
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
                
                def clean_name(raw_text):
                    if "default" in raw_text:
                        matches = re.findall(r"['\"]default['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw_text)
                        if matches:
                            return " ".join(matches)
                    return raw_text

                starters[away_team] = clean_name(goalie_cards[0].text.strip())
                starters[home_team] = clean_name(goalie_cards[1].text.strip())
                
        return starters
    except Exception as e:
        print(f"Error scraping starters: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_vegas_odds(api_key, region='us', market='h2h,totals'):
    """Fetches Moneyline (h2h) and Totals from The Odds API."""
    if not api_key:
        return {}
    
    # We now fetch both h2h (moneyline) and totals
    url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?apiKey={api_key}&regions={region}&markets={market}&oddsFormat=american"
    try:
        r = requests.get(url).json()
        odds_map = {}
        for game in r:
            home_team = game.get('home_team')
            # Initialize dict for this team
            if home_team not in odds_map:
                odds_map[home_team] = {'total': 6.5, 'home_ml': -110, 'away_ml': -110}

            for bookmaker in game.get('bookmakers', []):
                # We prioritize the first valid bookmaker we find
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
                            # Determine if outcome is for home or away team
                            if outcome['name'] == home_team:
                                odds_map[home_team]['home_ml'] = outcome['price']
                            else:
                                odds_map[home_team]['away_ml'] = outcome['price']
                        found_h2h = True
                
                if found_total and found_h2h:
                    break
        return odds_map
    except Exception as e:
        st.warning(f"Could not fetch Vegas odds: {e}")
        return {}

@st.cache_data(ttl=86400)
def get_active_goalies_db():
    url = "https://api-web.nhle.com/v1/goalie-stats-leaders/current?categories=goalsAgainstAverage&limit=250"
    try:
        r = requests.get(url).json()
        goalies = []
        for g in r.get('goalsAgainstAverage', []):
            name = f"{g['firstName']} {g['lastName']}"
            team = g['teamAbbrev']
            # Simulated GSAx
            gaa = g['value']
            if gaa < 2.5: gsax = round(np.random.uniform(0.3, 0.9), 2)
            elif gaa < 3.1: gsax = round(np.random.uniform(-0.1, 0.25), 2)
            else: gsax = round(np.random.uniform(-0.6, -0.1), 2)
            goalies.append({'Name': name, 'Team': team, 'GSAx': gsax})
        return pd.DataFrame(goalies)
    except Exception:
        return pd.DataFrame(columns=['Name', 'Team', 'GSAx'])

def reconcile_starters(starters_dict, goalie_df):
    if goalie_df.empty: return starters_dict, goalie_df
    official_names = goalie_df['Name'].tolist()
    final_starters = {}
    new_rows = []
    
    for team, scraped_name in starters_dict.items():
        if scraped_name in official_names:
            final_starters[team] = scraped_name
        else:
            matches = difflib.get_close_matches(scraped_name, official_names, n=1, cutoff=0.6)
            if matches:
                final_starters[team] = matches[0]
            else:
                new_rows.append({'Name': scraped_name, 'Team': team, 'GSAx': 0.00})
                final_starters[team] = scraped_name
                
    if new_rows:
        goalie_df = pd.concat([goalie_df, pd.DataFrame(new_rows)], ignore_index=True)
    
    goalie_df = pd.concat([goalie_df, pd.DataFrame([
        {'Name': 'Average Goalie', 'Team': 'NHL', 'GSAx': 0.00},
        {'Name': 'Backup/Rookie', 'Team': 'NHL', 'GSAx': -0.40}
    ])], ignore_index=True)
    
    return final_starters, goalie_df.drop_duplicates(subset=['Name']).sort_values('Name')

def get_simulated_ratings(active_teams):
    data = []
    for t in active_teams:
        # Simulated Team Strength (Goals For/Against expectation)
        off_rating = np.random.uniform(2.9, 3.4) 
        def_rating = np.random.uniform(2.9, 3.4) 
        data.append({'team': t, 'off_rating': off_rating, 'def_rating': def_rating})
    return pd.DataFrame(data).set_index('team')

def match_vegas_odds(home_team_name, odds_map):
    """Fuzzy matches and returns dict with total and MLs."""
    default = {'total': 6.5, 'home_ml': -110, 'away_ml': -110}
    if not odds_map: return default
    
    if home_team_name in odds_map:
        return odds_map[home_team_name]
    
    keys = list(odds_map.keys())
    matches = difflib.get_close_matches(home_team_name, keys, n=1, cutoff=0.5)
    if matches:
        return odds_map[matches[0]]
    
    return default

# --- 2. MATH HELPERS ---

def implied_probability(american_odds):
    """Converts American odds to implied probability (0.0 to 1.0)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def probability_to_american(probability):
    """Converts probability (0.0 to 1.0) to American odds."""
    if probability <= 0.5:
        return int(round(-100 * ((1 / probability) - 1))) if probability > 0 else 20000
    else:
        return int(round((probability / (1 - probability)) * -100)) if probability < 1 else -20000

def calculate_moneyline_projection(h_stats, a_stats, h_gsax, a_gsax):
    """
    Project Home Win Probability based on:
    1. Team Offense/Defense Ratings
    2. Home Ice Advantage
    3. Goalie GSAx Impact
    """
    # Expected Goals for Home vs Away (Raw)
    exp_h_goals = (h_stats['off_rating'] + a_stats['def_rating']) / 2
    exp_a_goals = (a_stats['off_rating'] + h_stats['def_rating']) / 2
    
    # Adjust for Home Ice
    exp_h_goals += HOME_ICE_ADVANTAGE / 2
    exp_a_goals -= HOME_ICE_ADVANTAGE / 2
    
    # Apply Goalie GSAx to the "Strength"
    # Home Strength = Exp Goals + Home Goalie Impact
    h_strength = exp_h_goals + (h_gsax * 0.5) 
    a_strength = exp_a_goals + (a_gsax * 0.5) 
    
    # Pythagorean Expectation with exponent ~2 (standard for hockey)
    exponent = 2.0
    home_win_prob = (h_strength ** exponent) / ((h_strength ** exponent) + (a_strength ** exponent))
    
    return home_win_prob

def get_gsax(goalie_name, goalie_df):
    try:
        return goalie_df.loc[goalie_df['Name'] == goalie_name, 'GSAx'].values[0]
    except:
        return 0.0

# --- 4. MAIN APP ---

def main():
    st.set_page_config(page_title="NHL Edge Finder", page_icon="🏒", layout="wide")
    
    st.title("🏒 NHL Edge Finder: Totals & Moneyline")
    st.markdown("Reverse engineer the total & moneyline, compare to live lines, find the edge.")

    # -- Sidebar --
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_date = st.date_input("Game Date", datetime.now())
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.divider()
        st.subheader("💰 Vegas Integration")
        odds_api_key = st.text_input("The Odds API Key (Optional)", type="password", help="Get a free key at the-odds-api.com")
        
        st.divider()
        st.subheader("🎯 Betting Strategy")
        totals_edge_threshold = st.slider("Min Edge (Totals)", 0.1, 1.5, 0.5, 0.1)
        ml_edge_threshold = st.slider("Min Edge (Moneyline %)", 1.0, 15.0, 5.0, 0.5, help="Difference in implied probability (%) to trigger a ML bet.")
        
        load_btn = st.button("🚀 Run Model", type="primary")

    # -- LOGIC --
    if load_btn:
        with st.spinner("Crunching numbers & scraping lines..."):
            games = get_schedule(date_str)
            if not games:
                st.error("No games found.")
                st.session_state['data_loaded'] = False
            else:
                starters_dict = get_projected_starters()
                raw_goalies = get_active_goalies_db()
                final_starters, final_goalie_db = reconcile_starters(starters_dict, raw_goalies)
                
                active_teams = set([g['home_team'] for g in games] + [g['away_team'] for g in games])
                ratings = get_simulated_ratings(active_teams)
                
                # Fetch Vegas Odds (Now includes ML)
                vegas_odds = get_vegas_odds(odds_api_key) if odds_api_key else {}
                
                st.session_state['games'] = games
                st.session_state['starters'] = final_starters
                st.session_state['goalie_db'] = final_goalie_db
                st.session_state['ratings'] = ratings
                st.session_state['vegas_odds'] = vegas_odds
                st.session_state['data_loaded'] = True

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

            # 1. Base Stats
            try:
                h_stats = ratings.loc[home]
                a_stats = ratings.loc[away]
                
                # Totals Calculation
                base_total = (h_stats['off_rating'] + a_stats['def_rating'])/2 + \
                             (a_stats['off_rating'] + h_stats['def_rating'])/2
            except:
                h_stats, a_stats = None, None
                base_total = LEAGUE_AVG_TOTAL

            # 2. Get Vegas Data
            vegas_data = match_vegas_odds(home_full, vegas_odds)
            auto_total = vegas_data.get('total', 6.5)
            auto_h_ml = vegas_data.get('home_ml', -110)
            auto_a_ml = vegas_data.get('away_ml', -110)
            
            with st.container(border=True):
                st.markdown(f"#### {away} @ {home}")
                
                # Layout: Goalies | Totals | Moneyline
                c1, c2, c3 = st.columns([1.5, 1.2, 1.3])
                
                # --- GOALIES ---
                a_start = starters.get(away, "Average Goalie")
                h_start = starters.get(home, "Average Goalie")
                
                try: a_idx = goalie_names.index(a_start)
                except: a_idx = goalie_names.index("Average Goalie")
                try: h_idx = goalie_names.index(h_start)
                except: h_idx = goalie_names.index("Average Goalie")

                with c1:
                    st.caption("🥅 Goaltending")
                    sel_a_goalie = st.selectbox(f"{away} G", goalie_names, index=a_idx, key=f"a_{gid}")
                    sel_h_goalie = st.selectbox(f"{home} G", goalie_names, index=h_idx, key=f"h_{gid}")
                    a_gsax = get_gsax(sel_a_goalie, goalie_db)
                    h_gsax = get_gsax(sel_h_goalie, goalie_db)

                # --- TOTALS LOGIC ---
                my_total_proj = base_total - h_gsax - a_gsax
                total_edge = my_total_proj - float(auto_total)
                
                with c2:
                    st.caption("📊 Totals")
                    vegas_line = st.number_input("Line", value=float(auto_total), step=0.5, key=f"v_{gid}")
                    st.metric("Projected", f"{my_total_proj:.2f}")
                    
                    if abs(total_edge) >= totals_edge_threshold:
                        if total_edge > 0:
                            st.success(f"**OVER** {vegas_line} (+{abs(total_edge):.2f})")
                        else:
                            st.error(f"**UNDER** {vegas_line} ({total_edge:.2f})")
                    else:
                        st.caption("No Total Edge")

                # --- MONEYLINE LOGIC ---
                with c3:
                    st.caption("💵 Moneyline")
                    
                    # User Input for Lines (pre-filled with scraped data)
                    col_h, col_a = st.columns(2)
                    with col_h:
                        v_h_ml = st.number_input(f"{home} ML", value=int(auto_h_ml), step=10, key=f"ml_h_{gid}")
                    with col_a:
                        v_a_ml = st.number_input(f"{away} ML", value=int(auto_a_ml), step=10, key=f"ml_a_{gid}")

                    if h_stats is not None:
                        # Calculate Probability
                        proj_h_prob = calculate_moneyline_projection(h_stats, a_stats, h_gsax, a_gsax)
                        proj_a_prob = 1 - proj_h_prob
                        
                        # Convert to American Odds for display
                        proj_h_ml = probability_to_american(proj_h_prob)
                        
                        st.write(f"**Proj:** {home} {proj_h_ml} ({proj_h_prob*100:.1f}%)")

                        # Calculate Edges (Diff in Probability)
                        implied_h = implied_probability(v_h_ml)
                        implied_a = implied_probability(v_a_ml)
                        
                        edge_h = (proj_h_prob - implied_h) * 100
                        edge_a = (proj_a_prob - implied_a) * 100
                        
                        if edge_h >= ml_edge_threshold:
                            st.success(f"**BET {home}** (+{edge_h:.1f}% Edge)")
                        elif edge_a >= ml_edge_threshold:
                            st.success(f"**BET {away}** (+{edge_a:.1f}% Edge)")
                        else:
                            st.caption("No ML Value")
                    else:
                        st.write("Need Stats")

if __name__ == "__main__":
    main()
