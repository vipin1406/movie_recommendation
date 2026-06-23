import streamlit as st
import pickle
import requests
import os
from dotenv import load_dotenv

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
similarity = pickle.load(open('similarity.pkl', 'rb'))

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
        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": API_KEY,
            "query": movie_name
        }

        response = requests.get(url, params=params, timeout=5).json()

        results = response.get("results", [])

        if not results:
            return None

        poster_path = results[0].get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None

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
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    return [(movies.iloc[i[0]].title, i[1]) for i in movie_list]


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
                    st.caption(f"Similarity: {round(score, 2)}")
                    poster = fetch_poster(name)

                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Image")