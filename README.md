source env/bin/activate
pip install -r requirements.txt
python embeddings.py
python -m streamlit run app.py