import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("🎬 Movie Recommendation System")
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    movies = df[['title', 'overview']].copy()
    movies['overview'] = movies['overview'].fillna("")
    return movies
movies = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]].title for i in sim_scores[1:6]]
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)
if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
