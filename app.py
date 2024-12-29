# KELOMPOK 9 - Content Based Recommender System
# Bayu Seno Nugroho - 1301213270
# Abidzar Ahmad Haikal - 1301213288
# Satya Rayyis Baruna - 1301213316
# Muhammad Rayhan Saniputra - 1301213262

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


scorers = pd.read_csv("goalscorers.csv")


# Preprocessing
# Melakukan preprocessing data dimana melakukan pengubahan kolom “date” menjadi format datetime, dan menambahkan kolom baru “year” untuk mengekstrak tahun dari tanggal permainan
def preprocess_data():
    results['date'] = pd.to_datetime(results['date'])
    results['year'] = results['date'].dt.year
    return results


# Load datasets
# Memuat dataset yang berada di folder seperti result.csv, shootouts.cs, dan goalscorers.csv sebagai dataframe dimana nantinya akan diolah lebih lanjut
results = pd.read_csv("results.csv")
shootouts = pd.read_csv("shootouts.csv")
goal


# Fungsi untuk mengekstrak berbagai fitur yang menggambarkan performa tim sepak bola berdasarkan hasil pertandingan yang ada di dalam results_df (DataFrame yang berisi data hasil pertandingan).
def extract_team_features(results_df):
    """
    Mengekstrak fitur-fitur dari setiap tim untuk content-based recommendation
    """
    # Membuat DataFrame untuk menyimpan fitur tim
    team_features = pd.DataFrame()
   
    # Rata-rata gol yang dicetak sebagai tuan rumah
    home_goals = results_df.groupby('home_team')['home_score'].mean()
   
    # Rata-rata gol yang dicetak sebagai tamu
    away_goals = results_df.groupby('away_team')['away_score'].mean()
   
    # Win rate sebagai tuan rumah
    home_wins = results_df.groupby('home_team').apply(
        lambda x: (x['home_score'] > x['away_score']).mean()
    )
   
    # Win rate sebagai tamu
    away_wins = results_df.groupby('away_team').apply(
        lambda x: (x['away_score'] > x['home_score']).mean()
    )
   
    # Rata-rata gol yang kebobolan sebagai tuan rumah
    home_conceded = results_df.groupby('home_team')['away_score'].mean()
   
    # Rata-rata gol yang kebobolan sebagai tamu
    away_conceded = results_df.groupby('away_team')['home_score'].mean()
   
    # Total pertandingan sebagai tuan rumah
    home_matches = results_df['home_team'].value_counts()
   
    # Total pertandingan sebagai tamu
    away_matches = results_df['away_team'].value_counts()
   
    # Menggabungkan semua fitur
    all_teams = pd.Index(set(results_df['home_team']).union(set(results_df['away_team'])))
   
    team_features['avg_home_goals'] = home_goals.reindex(all_teams).fillna(0)
    team_features['avg_away_goals'] = away_goals.reindex(all_teams).fillna(0)
    team_features['home_win_rate'] = home_wins.reindex(all_teams).fillna(0)
    team_features['away_win_rate'] = away_wins.reindex(all_teams).fillna(0)
    team_features['avg_home_conceded'] = home_conceded.reindex(all_teams).fillna(0)
    team_features['avg_away_conceded'] = away_conceded.reindex(all_teams).fillna(0)
    team_features['total_home_matches'] = home_matches.reindex(all_teams).fillna(0)
    team_features['total_away_matches'] = away_matches.reindex(all_teams).fillna(0)
   
    return team_features


# menghitung kesamaan antar tim menggunakan cosine similarity berdasarkan fitur-fitur yang dimiliki oleh setiap tim.
def calculate_similarity(team_features):
    """
    Menghitung cosine similarity antara semua tim berdasarkan fitur mereka
    """
    # Normalisasi fitur menggunakan Min-Max scaling
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(team_features)
   
    # Menghitung cosine similarity
    similarity_matrix = cosine_similarity(features_normalized)
   
    # Membuat DataFrame similarity dengan nama tim sebagai index dan columns
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=team_features.index,
        columns=team_features.index
    )
   
    return similarity_df


# Merekomendasikan pertandingan sepak bola dengan mencari tim-tim yang paling mirip berdasarkan fitur tim dan cosine similarity, kemudian menampilkan pertandingan terbaru yang melibatkan tim-tim tersebut.
def recommend_matches_content_based(team, results_df, team_features, similarity_df, num_matches=5):
    """
    Merekomendasikan pertandingan berdasarkan content-based filtering dengan cosine similarity
    """
    # Mendapatkan 5 tim yang paling mirip
    similar_teams = similarity_df[team].sort_values(ascending=False)[1:6].index.tolist()
   
    # Mengambil pertandingan terbaru yang melibatkan tim-tim yang mirip
    similar_team_matches = results_df[
        (results_df['home_team'].isin(similar_teams)) |
        (results_df['away_team'].isin(similar_teams))
    ]
   
    # Mengurutkan berdasarkan tanggal terbaru
    recommended_matches = similar_team_matches.sort_values('date', ascending=False).head(num_matches)
   
    return recommended_matches, similar_teams


# fungsi evaluate_recommendation bertujuan untuk mengevaluasi performa sistem rekomendasi menggunakan metrik MSE dan RMSE.
def evaluate_recommendations(results_df):
    """
    Evaluasi sistem rekomendasi menggunakan MSE
    """
# Membuat dataframe kosong untuk menyimpan fitur fitur yang sudah dihitung
    features = pd.DataFrame()
 
# Menghitung rata-rata gol yang dicetak tim sebagai tuan rumah  
    features['avg_home_goals'] = results_df.groupby('home_team')['home_score'].transform('mean')


# Menghitung rata-rata gol yang dicetak tim sebagai tamu
    features['avg_away_goals'] = results_df.groupby('away_team')['away_score'].transform('mean')


# Menghitung tingkat kemenangan tim sebagai tuan rumah   
    features['home_win_rate'] = results_df.groupby('home_team')['home_score'].transform(
        lambda x: (x > results_df.loc[x.index, 'away_score']).mean()
    )


# Menghitung tingkat kemenangan tim sebagai tamu
    features['away_win_rate'] = results_df.groupby('away_team')['away_score'].transform(
        lambda x: (x > results_df.loc[x.index, 'home_score']).mean()
    )
 
# Menentukan target nilai `y` sebagai jumlah total gol dalam pertandingan  
    y = results_df['home_score'] + results_df['away_score']
 
# Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)  
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )


# Membuat prediksi jumlah gol dengan rata-rata fitur gol sebagai tuan rumah dan tamu   
    y_pred = (X_test['avg_home_goals'] + X_test['avg_away_goals']) / 2


# Menghitung Mean Squared Error (MSE) antara data uji dan prediksi  
    mse = mean_squared_error(y_test, y_pred)
# Menghitung Root Mean Squared Error (RMSE) dengan mengambil akar kuadrat dari MSE
    rmse = np.sqrt(mse)
 
# Mengembalikan hasil evaluasi berupa MSE, RMSE, dan ukuran data uji  
    return {
        'mse': mse,
        'rmse': rmse,
        'test_size': len(y_test)
    }
# Menyimpan data yang telah diproses ke dalam variabel result
results = preprocess_data()


# Dashboard layout
# Mengatur konfigurasi halaman dashboard
st.set_page_config(page_title="Football Results Dashboard", layout="wide", page_icon="🏆")
# Menampilkan judul utama pada dashboard
st.title("🏆 International Football Recommender System")
# Menampilkan gambar
st.image("https://cdn.ibbox.id/cdn/img/qvbyl.jpg", caption="Football Moments", use_container_width=True)


# Evaluasi keseluruhan sistem
overall_metrics = evaluate_recommendations(results) #mengukur kinerja dari sistem rekomendasi menggunakan MSE, RMSE, Dan Test Set Size
st.header("📊 Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)#menampilkan tiga kolom
with col1:
    st.metric("Overall MSE", f"{overall_metrics['mse']:.4f}")
with col2:
    st.metric("Overall RMSE", f"{overall_metrics['rmse']:.4f}")
with col3:
    st.metric("Test Set Size", overall_metrics['test_size'])


# Sidebar
with st.sidebar:
    st.image("https://mf-chan.com/tel-u-logo/lib/FIF/standar-utama.png", width=500)
    st.header("📊 Fun Facts / FYI")


    most_wins_team = results['home_team'].value_counts().idxmax()
    most_wins_count = results['home_team'].value_counts().max()


    highest_scoring_match = results.loc[results[['home_score', 'away_score']].sum(axis=1).idxmax()]
    highest_scoring_total = highest_scoring_match['home_score'] + highest_scoring_match['away_score']


    most_common_tournament = results['tournament'].value_counts().idxmax()
    most_common_tournament_count = results['tournament'].value_counts().max()


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
    lat="neutral",
    lon="neutral",
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


# Feature 4: Content-Based Match Recommendations
st.header("🔄 Content-Based Match Recommendations")


# Ekstrak fitur dan hitung similarity
team_features = extract_team_features(results)
similarity_df = calculate_similarity(team_features)


# Dapatkan rekomendasi dan tim yang mirip
recommended_matches, similar_teams = recommend_matches_content_based(
    selected_team,
    results,
    team_features,
    similarity_df
)


# Tampilkan hasil rekomendasi
st.subheader(f"Recommended matches based on teams similar to {selected_team}:")
st.write(recommended_matches[['date', 'home_team', 'away_team', 'tournament', 'home_score', 'away_score']])


# Tampilkan tim yang mirip dan similarity score
st.subheader("Most similar teams:")
similar_teams_scores = similarity_df[selected_team].sort_values(ascending=False)[1:6]
similar_teams_df = pd.DataFrame({
    'Team': similar_teams_scores.index,
    'Similarity Score': similar_teams_scores.values.round(4)
})
st.write(similar_teams_df)


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
