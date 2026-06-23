import streamlit as st
import pickle
import requests
import os
from dotenv import load_dotenv
import numpy as np
import streamlit as st
import pickle
import requests
import os

import re

POSTER_DIR = "posters"

# create folder if not exists
if not os.path.exists(POSTER_DIR):
    os.makedirs(POSTER_DIR)
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Movie Recommender")

API_KEY = os.getenv("TMDB_API_KEY")

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pickle.load(open('movies.pkl', 'rb'))
movies['genres_str'] = movies['genres'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else str(x)
)

embeddings = pickle.load(open('embeddings.pkl', 'rb'))
movies['title_lower'] = movies['title'].str.lower()

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "home_movies" not in st.session_state:
    st.session_state.home_movies = movies.sample(10).reset_index(drop=True)


# -----------------------------
# POSTER FUNCTION (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    try:
        # ✅ safe filename
        filename = re.sub(r'[^a-zA-Z0-9]', '_', movie_name.lower()) + ".jpg"
        filepath = os.path.join(POSTER_DIR, filename)

        # ✅ 1. check if already saved
        if os.path.exists(filepath):
            return filepath

        # ✅ 2. fetch from TMDB
        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": API_KEY,
            "query": movie_name
        }

        response = requests.get(url, params=params, timeout=5).json()
        results = response.get("results", [])

        if not results:
            return None

        # ✅ exact match first
        poster_path = None
        for movie in results:
            if movie['title'].lower() == movie_name.lower():
                poster_path = movie.get("poster_path")
                break

        # fallback
        if not poster_path:
            poster_path = results[0].get("poster_path")

        if not poster_path:
            return None

        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"

        # ✅ 3. download image
        img_data = requests.get(image_url, timeout=5).content

        with open(filepath, "wb") as f:
            f.write(img_data)

        return filepath

    except:
        return None

# -----------------------------
# RECOMMENDATION FUNCTION (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def recommend(movie):
    movie = movie.lower()

    if movie not in movies['title_lower'].values:
        return []

    index = movies[movies['title_lower'] == movie].index[0]

    # Compute cosine similarity using embeddings
    scores = np.dot(embeddings, embeddings[index]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings[index])
    )

    similar_movies = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    return [(movies.iloc[i[0]].title, i[1]) for i in similar_movies]


def explain_recommendation(selected_movie, recommended_movie):
    
    selected = movies[movies['title'] == selected_movie].iloc[0]
    recommended = movies[movies['title'] == recommended_movie].iloc[0]

    selected_genres = set(selected['genres_str'].lower().split())
    recommended_genres = set(recommended['genres_str'].lower().split())

    common_genres = selected_genres.intersection(recommended_genres)

    if common_genres:
        return f"Shared genres: {', '.join(list(common_genres)[:3])}"
    
    return "Similar overall theme"

# -----------------------------
# HEADER
# -----------------------------
st.title("🎬 Movie Recommendation System")


# -----------------------------
# BACK BUTTON
# -----------------------------
if st.session_state.page == "details":
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.session_state.selected_movie = None
        st.rerun()


# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":

    st.subheader("🔥 Trending Movies")

    movie_list = st.session_state.home_movies
    movies_per_row = 5

    for i in range(0, len(movie_list), movies_per_row):

        row = movie_list.iloc[i:i + movies_per_row]
        cols = st.columns(len(row))

        for j in range(len(row)):

            with cols[j]:

                title = row.iloc[j]['title']
                poster = fetch_poster(title)

                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/200x300?text=No+Image")

                # ✅ FIXED: stable key using movie title
                if st.button(title, key=title):

                    st.session_state.selected_movie = title
                    st.session_state.page = "details"
                    st.rerun()


# -----------------------------
# DETAILS PAGE
# -----------------------------
elif st.session_state.page == "details":

    movie = st.session_state.selected_movie

    if movie:

        st.markdown("---")
        st.subheader("🎯 Selected Movie")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:

            poster = fetch_poster(movie)

            if poster:
                st.image(poster, width=300)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Image")

            st.markdown(f"### {movie}")

        # -----------------------------
        # RECOMMENDATIONS
        # -----------------------------
        st.subheader("🍿 Recommended Movies")
        st.markdown("### 🤔 Why these recommendations?")

        st.info("""
        These movies are recommended based on similarity in:
        - Genre
        - Keywords
        - Movie overview

        We use cosine similarity on movie feature vectors.
        """)

        names = recommend(movie)

        if not names:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(5)

            for i, (name, score) in enumerate(names):

                with cols[i % 5]:
                    st.text(name)
                    st.caption(f"Match: {int(score * 100)}%")

                    # 🔥 ADD THIS LINE
                    reason = explain_recommendation(movie, name)
                    st.caption(reason)

                    rec_poster = fetch_poster(name)

                    if rec_poster:
                        st.image(rec_poster, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Image")