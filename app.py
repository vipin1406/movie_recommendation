import streamlit as st
import pickle
import requests
import os
import numpy as np
import re
import ast
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Movie Recommender")
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

POSTER_DIR = "posters"
os.makedirs(POSTER_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pickle.load(open('movies.pkl', 'rb'))

movies['title_lower'] = movies['title'].str.lower()
movies['cast'] = movies['cast'].apply(lambda x: x if isinstance(x, list) else [])

# -----------------------------
# DIRECTOR FROM CREW
# -----------------------------
def get_director(obj):
    try:
        if isinstance(obj, list):
            for i in obj:
                if i.get("job") == "Director":
                    return [i.get("name")]
        return []
    except:
        return []

movies['director'] = movies['crew'].apply(get_director)

# -----------------------------
# GENRES CLEANING
# -----------------------------
def convert(obj):
    try:
        return ast.literal_eval(obj)
    except:
        return []

def get_names(obj):
    return [i['name'] for i in obj if isinstance(i, dict) and 'name' in i]

movies['genres'] = movies['genres'].apply(convert)
movies['genres'] = movies['genres'].apply(get_names)
movies['genres_str'] = movies['genres'].apply(lambda x: " ".join(x))

# -----------------------------
# NORMALIZATION
# -----------------------------
movies['vote_average_norm'] = movies['vote_average'] / 10
movies['vote_count_norm'] = movies['vote_count'] / movies['vote_count'].max()
movies['popularity_norm'] = movies['popularity'] / movies['popularity'].max()

# -----------------------------
# LOOKUP MAP
# -----------------------------
movie_index = {title: i for i, title in enumerate(movies['title_lower'])}

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = pickle.load(open('embeddings.pkl', 'rb'))

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
# POSTER FUNCTION
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    try:
        filename = re.sub(r'[^a-zA-Z0-9]', '_', movie_name.lower()) + ".jpg"
        filepath = os.path.join(POSTER_DIR, filename)

        if os.path.exists(filepath):
            return filepath

        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": API_KEY, "query": movie_name}

        response = requests.get(url, params=params, timeout=5).json()
        results = response.get("results", [])

        if not results:
            return None

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None

        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        img_data = requests.get(image_url, timeout=5).content

        with open(filepath, "wb") as f:
            f.write(img_data)

        return filepath

    except:
        return None

# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------
@st.cache_data(show_spinner=False)
def recommend(movie):
    movie = movie.lower()

    if movie not in movie_index:
        return []

    index = movie_index[movie]

    norms = np.linalg.norm(embeddings, axis=1)
    target_norm = np.linalg.norm(embeddings[index])

    scores = np.dot(embeddings, embeddings[index]) / (norms * target_norm + 1e-10)

    results = []
    selected_genres = set(movies.iloc[index]['genres'])

    for i, sim_score in enumerate(scores):

        if i == index:
            continue    

        if sim_score < 0.5:
            continue

        rec_genres = set(movies.iloc[i]['genres'])

        genre_match = len(selected_genres.intersection(rec_genres))

        final_score = sim_score + (0.1 * genre_match)
        results.append((i, final_score, sim_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:10]

    return [(movies.iloc[i[0]].title, i[2]) for i in results]

# -----------------------------
# EXPLAINABILITY
# -----------------------------
def explain_recommendation(selected_movie, recommended_movie):

    selected = movies[movies['title'] == selected_movie].iloc[0]
    recommended = movies[movies['title'] == recommended_movie].iloc[0]

    selected_genres = set(selected['genres_str'].lower().split())
    recommended_genres = set(recommended['genres_str'].lower().split())

    common_genres = selected_genres.intersection(recommended_genres)

    selected_cast = set(selected.get('cast', []))
    recommended_cast = set(recommended.get('cast', []))
    common_cast = selected_cast.intersection(recommended_cast)

    selected_director = set(selected.get('director', []))
    recommended_director = set(recommended.get('director', []))
    common_director = selected_director.intersection(recommended_director)

    if common_director:
        return f"Same director: {list(common_director)[0]}"

    if common_cast:
        return f"Shared cast: {', '.join(list(common_cast)[:2])}"

    if common_genres:
        return f"Shared genres: {', '.join(list(common_genres)[:3])}"

    return "Similar storyline/theme"

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

# BACK BUTTON
if st.session_state.page == "details":
    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.session_state.selected_movie = None
        st.rerun()

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":

    search_query = st.text_input("🔍 Search movies...")

    st.subheader("🔥 Trending Movies")

    movie_list = st.session_state.home_movies

    if search_query:
        movie_list = movies[movies['title'].str.lower().str.contains(search_query.lower())]

        if movie_list.empty:
            st.warning("⚠️ No movies found")

    cols_per_row = 5

    for i in range(0, len(movie_list), cols_per_row):

        row = movie_list.iloc[i:i + cols_per_row]
        cols = st.columns(len(row))

        for j in range(len(row)):

            with cols[j]:

                title = row.iloc[j]['title']
                poster = fetch_poster(title)

                st.image(
                    poster if poster else "https://via.placeholder.com/200x300",
                    width=200
                )

                if st.button(title, key=f"home_{title}"):
                    st.session_state.selected_movie = title
                    st.session_state.page = "details"
                    st.rerun()

# -----------------------------
# DETAILS PAGE
# -----------------------------
elif st.session_state.page == "details":

    movie = st.session_state.selected_movie

    st.subheader("🎯 Selected Movie")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        poster = fetch_poster(movie)
        st.image(
            poster if poster else "https://via.placeholder.com/300x450",
            width=300
        )

    st.markdown("---")
    st.markdown(f"## 🎬 {movie}")

    rec_movie = movies[movies['title'] == movie].iloc[0]

    st.caption(f"⭐ Rating: {rec_movie['vote_average']}")

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("🍿 Recommended Movies")

    names = recommend(movie)

    # 🔥 HANDLE EMPTY CASE
    if not names:
        st.warning("⚠️ No similar movies found")
    else:
        cols = st.columns(5)

        for i, (name, score) in enumerate(names):

            with cols[i % 5]:

                poster = fetch_poster(name)
                st.image(
                    poster if poster else "https://via.placeholder.com/200x300",
                    width=200
                )

                if st.button(name, key=f"rec_{name}_{i}"):
                    st.session_state.selected_movie = name
                    st.rerun()

                st.caption(f"Match: {int(score*100)}%")
                st.success(explain_recommendation(movie, name))