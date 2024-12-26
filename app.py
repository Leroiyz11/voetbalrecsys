import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Load datasets
results = pd.read_csv("results.csv")
shootouts = pd.read_csv("shootouts.csv")
goalscorers = pd.read_csv("goalscorers.csv")

# Preprocessing
def preprocess_data():
    results['date'] = pd.to_datetime(results['date'])
    results['year'] = results['date'].dt.year
    return results

results = preprocess_data()

# Dashboard layout
st.set_page_config(page_title="Football Results Dashboard", layout="wide", page_icon="🏆")
st.title("🏆 International Football Recommender System")
st.image("https://cdn.ibbox.id/cdn/img/qvbyl.jpg", caption="Football Moments", use_container_width=True)

# Sidebar
with st.sidebar:
    st.image("https://mf-chan.com/tel-u-logo/lib/FIF/standar-utama.png", width=500)
    st.header("📊 Fun Facts / FYI")

    # Fun fact 1: Team with the most wins
    most_wins_team = results['home_team'].value_counts().idxmax()
    most_wins_count = results['home_team'].value_counts().max()

    # Fun fact 2: Highest scoring match
    highest_scoring_match = results.loc[results[['home_score', 'away_score']].sum(axis=1).idxmax()]
    highest_scoring_total = highest_scoring_match['home_score'] + highest_scoring_match['away_score']

    # Fun fact 3: Most common tournament
    most_common_tournament = results['tournament'].value_counts().idxmax()
    most_common_tournament_count = results['tournament'].value_counts().max()

    # Display Fun Facts
    st.markdown(f"**🏆 Most Matches as Home Team:**")
    st.markdown(f"{most_wins_team} ({most_wins_count} matches)")

    st.markdown(f"**⚽ Highest Scoring Match:**")
    st.markdown(
        f"{highest_scoring_match['date']}: {highest_scoring_match['home_team']} "
        f"{highest_scoring_match['home_score']} - {highest_scoring_match['away_score']} "
        f"{highest_scoring_match['away_team']} ({highest_scoring_total} goals)"
    )

    st.markdown(f"**🌍 Most Common Tournament:**")
    st.markdown(f"{most_common_tournament} ({most_common_tournament_count} matches)")

    st.header("Filters")
    selected_year = st.slider("Select Year Range Of Total Goals", min_value=1872, max_value=2024, value=(2000, 2024))
    selected_team = st.selectbox("Select Team For Match Recommendation", options=sorted(results['home_team'].unique()))

# Filter data based on sidebar inputs
filtered_results = results[(results['year'] >= selected_year[0]) & (results['year'] <= selected_year[1])]

# Feature 1: Interactive Map
st.header("🌍 Match Locations")
map_data = filtered_results[['date', 'home_team', 'away_team', 'city', 'country', 'neutral']].dropna()
map_fig = px.scatter_mapbox(
    map_data,
    lat="neutral",  # Replace with actual latitude if available
    lon="neutral",  # Replace with actual longitude if available
    hover_name="city",
    hover_data={"date": True, "home_team": True, "away_team": True},
    title="Match Locations",
    height=500,
    color_discrete_sequence=["blue"],
)
map_fig.update_layout(mapbox_style="carto-positron", margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(map_fig, use_container_width=True)

# Feature 2: Goals Analysis
st.header("⚽ Goals Analysis")
goals_per_team = (
    filtered_results.groupby("home_team")["home_score"].sum() +
    filtered_results.groupby("away_team")["away_score"].sum()
).sort_values(ascending=False)
goals_fig = px.bar(
    goals_per_team,
    x=goals_per_team.index,
    y=goals_per_team.values,
    labels={"x": "Team", "y": "Total Goals"},
    title="Total Goals by Team",
    color_discrete_sequence=["green"],
)
st.plotly_chart(goals_fig, use_container_width=True)

# Goal Trends Over Time
st.sidebar.header("Goal Trends")
st.sidebar.markdown("View the evolution of goals over time.")
goals_per_year = results.groupby('year')[['home_score', 'away_score']].sum()
goals_per_year['total_goals'] = goals_per_year['home_score'] + goals_per_year['away_score']
fig = px.line(goals_per_year, x=goals_per_year.index, y='total_goals', title="Goals Scored Over Time")
st.plotly_chart(fig)

# Feature 3: Head-to-Head Analysis
st.header("🏃🏆 Head-to-Head Analysis")
h2h_team1 = st.selectbox("Select Team 1", options=sorted(results['home_team'].unique()), key="team1")
h2h_team2 = st.selectbox("Select Team 2", options=sorted(results['home_team'].unique()), key="team2")
h2h_data = results[
    (results['home_team'].isin([h2h_team1, h2h_team2])) &
    (results['away_team'].isin([h2h_team1, h2h_team2]))
]
st.write(h2h_data[['date', 'home_team', 'home_score', 'away_team', 'away_score', 'tournament']])

# Feature 4: Match Recommendations
st.header("🔄 Match Recommendations")
def recommend_matches(team, num_matches=5):
    team_matches = results[(results['home_team'] == team) | (results['away_team'] == team)]
    return team_matches.sort_values(by="date", ascending=False).head(num_matches)

recommended_matches = recommend_matches(selected_team)
st.write(recommended_matches[['date', 'home_team', 'away_team', 'tournament', 'home_score', 'away_score']])

# Feature 5: Dynamic Filters
st.header("🔍 Dynamic Filters")
selected_tournament = st.multiselect("Select Tournament(s)", options=results['tournament'].unique())
if selected_tournament:
    tournament_data = results[results['tournament'].isin(selected_tournament)]
    st.write(tournament_data[['date', 'home_team', 'away_team', 'tournament', 'home_score', 'away_score']])
else:
    st.write("Please select a tournament to view data.")

# Sidebar - Top Scorers
st.sidebar.header("Top Scorers")
num_players = st.sidebar.slider("Number of players to display", 5, 20, 10)
top_scorers = goalscorers['scorer'].value_counts().head(num_players)
st.subheader("Top Scorers")
st.bar_chart(top_scorers)

# Feature 6: Top Scorers from a Country
st.header("🌟 Top Scorers from a Country")
selected_team = st.selectbox(
    "Select a Team (Country) to View Top Scorers",
    options=sorted(goalscorers['team'].unique())
)
team_top_scorers = goalscorers[goalscorers['team'] == selected_team]
team_top_scorers_count = team_top_scorers['scorer'].value_counts()
if not team_top_scorers_count.empty:
    st.subheader(f"Top Scorers from {selected_team}")
    st.bar_chart(team_top_scorers_count.head(10))
else:
    st.write(f"No top scorers data available for {selected_team}.")

# Feature 7: Match Search by Year and Team
st.sidebar.header("Search Matches")
year = st.sidebar.selectbox("Select Year", sorted(results['year'].unique()))
team = st.sidebar.selectbox("Select Team", results['home_team'].unique())
search_results = results[
    (results['year'] == year) & 
    ((results['home_team'] == team) | (results['away_team'] == team))
]
st.subheader(f"Matches in {year} involving {team}")
st.dataframe(search_results[['date', 'home_team', 'home_score', 'away_team', 'away_score', 'tournament']])

# Feature 8: Tournament Recommendations
st.header("🏆 Tournament Recommendations")
team_tournament = st.selectbox("Select a Team for Tournament Insights", sorted(results['home_team'].unique()))
tournament_recommendations = results[
    (results['home_team'] == team_tournament) | (results['away_team'] == team_tournament)
]['tournament'].value_counts().head(5)
st.write(tournament_recommendations)

# Feature 9: Best Times to Watch
st.header("⏱️ Best Times to Watch")
goals_by_minute = goalscorers['minute'].value_counts().sort_index()
st.line_chart(goals_by_minute)

# Feature 10: Players goal by Situations
st.header("🥅 Players by Situations")
situation = st.selectbox("Select Goal Situation", ["Penalty", "Own Goal"])
if situation == "Penalty":
    penalty_scorers = goalscorers[goalscorers['penalty'] == 1]['scorer'].value_counts().head(10)
    st.bar_chart(penalty_scorers)
elif situation == "Own Goal":
    own_goal_scorers = goalscorers[goalscorers['own_goal'] == 1]['scorer'].value_counts().head(10)
    st.bar_chart(own_goal_scorers)

# Feature 11: Rival Recommendations
st.header("🤜🤛 Rival Recommendations")
selected_team = st.selectbox("Select a Team to View Rivals", sorted(results['home_team'].unique()))
rival_teams = results[
    (results['home_team'] == selected_team) | (results['away_team'] == selected_team)
]
rivals = pd.concat([rival_teams['home_team'], rival_teams['away_team']])
rivals = rivals[rivals != selected_team].value_counts().head(5)
st.write(f"Top rivals for {selected_team}:")
st.write(rivals)
