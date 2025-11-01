"""Build a FAISS index from saved embeddings.

Usage:
    python src/build_faiss_index.py --embeddings data/faiss_index/embeddings.npy --out_dir data/faiss_index
"""
import argparse
import numpy as np
import faiss
from pathlib import Path
from src.config import FAISS_INDEX_FILE, EMBEDDINGS_FILE, NAMES_FILE
import os

def build_flat_index(embeddings, normalize=True):
    # embeddings: numpy array (n, dim) float32
    n, dim = embeddings.shape
    if normalize:
        faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ~ cosine similarity
    index.add(embeddings)
    return index

def save_index(index, out_file):
    faiss.write_index(index, str(out_file))

def load_embeddings(emb_file):
    return np.load(emb_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default=str(EMBEDDINGS_FILE))
    parser.add_argument('--out_dir', type=str, default=str(Path('data/faiss_index')))
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings).astype('float32')
    if embeddings.size == 0:
        raise SystemExit('No embeddings found. Run extract_embeddings.py first.')

    print('Building FAISS index for embeddings shape:', embeddings.shape)
    faiss.normalize_L2(embeddings)
    index = build_flat_index(embeddings, normalize=False)  # already normalized
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / 'index.faiss'
    save_index(index, out_path)
    print('Saved FAISS index to', out_path)
