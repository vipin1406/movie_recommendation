from sentence_transformers import SentenceTransformer
import pickle

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your data
movies = pickle.load(open('movies.pkl', 'rb'))

# Combine text features (IMPORTANT)
movies['tags'] = (
    movies['overview'].fillna('').astype(str) + " " +
    movies['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
)

# Convert to list
texts = movies['tags'].fillna('').tolist()

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings
pickle.dump(embeddings, open('embeddings.pkl', 'wb'))