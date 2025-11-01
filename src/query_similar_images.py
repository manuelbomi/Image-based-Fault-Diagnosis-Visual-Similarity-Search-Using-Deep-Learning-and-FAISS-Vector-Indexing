"""Query a FAISS index for images similar to a provided query image.

Usage:
    python src/query_similar_images.py --query path/to/image.jpg --top_k 5
"""
import argparse
import numpy as np
import faiss
from pathlib import Path
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from src.config import FAISS_INDEX_FILE, EMBEDDINGS_FILE, NAMES_FILE, IMG_SIZE
import os

def load_model():
    # VGG16 without top, pooling average
    from tensorflow.keras.applications.vgg16 import VGG16
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

def preprocess(path):
    img = image.load_img(path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_feature(model, path):
    x = preprocess(path)
    feat = model.predict(x, verbose=0)
    return feat.astype('float32').reshape(1, -1)

def load_faiss(index_path):
    return faiss.read_index(str(index_path))

def search(index, query_vec, top_k=5):
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    return distances, indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Path to query image')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--index', type=str, default=str(Path('data/faiss_index/index.faiss')))
    parser.add_argument('--names', type=str, default=str(Path('data/faiss_index/image_names.npy')))
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise SystemExit('FAISS index not found. Run build_faiss_index.py first.')

    model = load_model()
    qvec = extract_feature(model, args.query)
    index = load_faiss(args.index)
    dists, inds = search(index, qvec, args.top_k + 1)  # +1 to account for self match

    names = np.load(args.names)
    results = []
    for dist, idx in zip(dists[0], inds[0]):
        if idx < 0 or idx >= len(names):
            continue
        results.append((names[idx], float(dist)))
    # Filter out the query itself if it's in the index (first result often self)
    # Present top_k unique results
    seen = set()
    out = []
    for name, score in results:
        if name not in seen:
            out.append((name, score))
            seen.add(name)
        if len(out) >= args.top_k:
            break

    print('Top similar images:')
    for n, s in out:
        print(f"- {n} (score: {s:.6f})")
