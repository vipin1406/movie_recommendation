from sentence_transformers import SentenceTransformer
import pickle
import ast
import re

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SentenceTransformer('all-mpnet-base-v2')

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pickle.load(open('movies.pkl', 'rb'))

# -----------------------------
# SAFE JSON CONVERSION
# -----------------------------
def convert(obj):
    try:
        return ast.literal_eval(obj)
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(convert)

# -----------------------------
# EXTRACT FEATURES
# -----------------------------
def get_names(obj):
    names = []
    for i in obj:
        if isinstance(i, dict) and 'name' in i:
            names.append(i['name'])
    return names

# Top 3 cast
movies['cast'] = movies['cast'].apply(lambda x: get_names(x)[:3])

# Director
def get_director(obj):
    for i in obj:
        if isinstance(i, dict) and i.get('job') == 'Director':
            return [i.get('name', '')]
    return []

movies['director'] = movies['crew'].apply(get_director)

# Keywords + Genres
movies['keywords'] = movies['keywords'].apply(get_names)
movies['genres'] = movies['genres'].apply(get_names)

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_list(text_list):
    return [i.replace(" ", "").lower() for i in text_list]

movies['genres'] = movies['genres'].apply(clean_list)
movies['keywords'] = movies['keywords'].apply(clean_list)
movies['cast'] = movies['cast'].apply(clean_list)
movies['director'] = movies['director'].apply(clean_list)

# Clean overview
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

movies['overview'] = movies['overview'].fillna('').apply(clean_text)

# -----------------------------
# CREATE TAGS (WEIGHTED FEATURES 🔥)
# -----------------------------
movies['tags'] = (
    (movies['overview'] + " ") * 3 +
    (movies['genres'].apply(lambda x: " ".join(x)) + " ") * 3 +
    (movies['keywords'].apply(lambda x: " ".join(x)) + " ") * 2 +
    (movies['cast'].apply(lambda x: " ".join(x)) + " ") * 1 +
    (movies['director'].apply(lambda x: " ".join(x)) + " ") * 1
)

movies['tags'] = movies['tags'].str.strip()

# -----------------------------
# DEBUG CHECK
# -----------------------------
print(movies[['title', 'tags']].head(3))

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
texts = movies['tags'].tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

# -----------------------------
# SAVE
# -----------------------------
pickle.dump(embeddings, open('embeddings.pkl', 'wb'))

print("✅ Embeddings created and saved successfully!")