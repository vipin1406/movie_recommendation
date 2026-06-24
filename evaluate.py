import pickle
import pandas as pd
import numpy as np

movies = pickle.load(open('movies.pkl', 'rb'))

movies['title_lower'] = movies['title'].str.lower()

movie_index = {title: i for i, title in enumerate(movies['title_lower'])}

def precision_at_k(movie_idx, recommended_indices, k=10):

    true_genres = set(movies.iloc[movie_idx]['genres'])

    hits = 0

    for i in recommended_indices[:k]:
        rec_genres = set(movies.iloc[i]['genres'])

        if len(true_genres.intersection(rec_genres)) > 0:
            hits += 1

    return hits / k


def evaluate_movie(movie_name):

    movie_name = movie_name.lower()

    movie_idx = movie_index[movie_name]   # correct lookup

    recs = recommend(movie_name)

    recommended_indices = []

    for name, _ in recs:
        idx = movies[movies['title'] == name].index[0]
        recommended_indices.append(idx)

    prec = precision_at_k(movie_idx, recommended_indices, k=10)

    print(f"Movie: {movie_name}")
    print(f"Precision@10: {prec:.2f}")