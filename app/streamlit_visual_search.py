"""Simple Streamlit app to upload an image and view similar images from FAISS index.

Run:
    streamlit run app/streamlit_visual_search.py
"""
import streamlit as st
from pathlib import Path
import numpy as np
import os
import subprocess
from src.config import FAISS_DIR
from src.query_similar_images import load_model, extract_feature, load_faiss
import faiss

st.set_page_config(page_title='Image Fault Visual Search', layout='wide')
st.title('Enterprise Image Fault Visual Search')

uploaded = st.file_uploader('Upload an image of a fault', type=['jpg','jpeg','png'])
top_k = st.slider('Number of similar images', 1, 20, 5)

INDEX_PATH = Path('data/faiss_index/index.faiss')
NAMES_PATH = Path('data/faiss_index/image_names.npy')

if not INDEX_PATH.exists():
    st.warning('FAISS index not found. Run src/build_faiss_index.py first and ensure embeddings exist.')
else:
    index = load_faiss(INDEX_PATH)

if uploaded is not None:
    # Save temp
    temp_path = Path('data') / 'temp_query.jpg'
    with open(temp_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    model = load_model()
    qvec = extract_feature(model, str(temp_path))
    faiss.normalize_L2(qvec)
    dists, inds = index.search(qvec, top_k + 1)
    names = np.load(NAMES_PATH)
    results = []
    for idx in inds[0]:
        if idx < 0 or idx >= len(names):
            continue
        results.append(names[idx])
    # Remove possible duplicate/self match
    unique = []
    for n in results:
        if n not in unique:
            unique.append(n)
    st.header('Query')
    st.image(str(temp_path), width=300)
    st.header('Similar images')
    cols = st.columns(5)
    for i, name in enumerate(unique[:top_k]):
        col = cols[i % 5]
        img_path = Path('data/raw') / name
        if img_path.exists():
            col.image(str(img_path), caption=name, use_column_width=True)
        else:
            col.write(name)
