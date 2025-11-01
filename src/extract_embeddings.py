"""Extract image embeddings using VGG16 and save them to disk.

Usage:
    python src/extract_embeddings.py --image_dir data/raw --out_dir data/faiss_index
"""
import argparse
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm
from pathlib import Path
import cv2

from src.config import IMG_SIZE, EMBEDDING_DTYPE, EMBEDDINGS_FILE, NAMES_FILE

def load_model():
    # Load pre-trained VGG16, excluding top layers, with global avg pooling to get fixed-size embeddings
    base = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return base

def load_and_preprocess(img_path, target_size=IMG_SIZE):
    # Read and preprocess an image so it can be fed into the VGG16 model
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_embeddings(image_dir, out_dir):
    model = load_model()
    image_paths = [p for p in sorted(Path(image_dir).glob('*')) if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    embeddings = []
    names = []

    for p in tqdm(image_paths, desc='Extracting embeddings'):
        try:
            x = load_and_preprocess(str(p))
            feat = model.predict(x, verbose=0)
            embeddings.append(feat.flatten())
            names.append(p.name)
        except Exception as e:
            print(f"Failed to process {p}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    emb_arr = np.array(embeddings).astype(EMBEDDING_DTYPE)
    np.save(EMBEDDINGS_FILE, emb_arr)
    np.save(NAMES_FILE, np.array(names))

    print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")
    print(f"Saved names to {NAMES_FILE}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=str(Path('data/raw')), help='Directory with input images')
    parser.add_argument('--out_dir', type=str, default=str(Path('data/faiss_index')), help='Directory to save embeddings')
    args = parser.parse_args()
    extract_embeddings(args.image_dir, args.out_dir)
