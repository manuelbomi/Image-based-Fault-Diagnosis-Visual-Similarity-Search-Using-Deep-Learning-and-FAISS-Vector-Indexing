"""Train a simple classifier on saved embeddings to predict fault categories.

Assumes a CSV at data/processed/labels.csv with columns: filename,label
Embeddings must be present in data/faiss_index/embeddings.npy in the same order as image_names.npy
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load
from pathlib import Path
from src.config import EMBEDDINGS_FILE, NAMES_FILE, LABELS_FILE, CLASSIFIER_FILE
import os

def load_data():
    emb = np.load(EMBEDDINGS_FILE)
    names = np.load(NAMES_FILE)
    # Load CSV
    if not Path(LABELS_FILE).exists():
        raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}. Create a CSV with columns: filename,label")
    df = pd.read_csv(LABELS_FILE)
    # Map embeddings to labels by filename
    name_to_label = dict(zip(df['filename'], df['label']))
    y = []
    X = []
    for name, vec in zip(names, emb):
        if name in name_to_label:
            X.append(vec)
            y.append(name_to_label[name])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=str(Path('data/faiss_index/classifier.joblib')))
    args = parser.parse_args()

    X, y = load_data()
    print('Loaded', X.shape, 'with labels', np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    dump(clf, args.out)
    print('Saved classifier to', args.out)
