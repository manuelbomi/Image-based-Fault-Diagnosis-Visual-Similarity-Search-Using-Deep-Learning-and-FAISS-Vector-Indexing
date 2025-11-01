"""Configuration constants for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_DIR = DATA_DIR / "faiss_index"

# Embedding / model settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EMBEDDING_DTYPE = "float32"

# Files
EMBEDDINGS_FILE = FAISS_DIR / "embeddings.npy"
NAMES_FILE = FAISS_DIR / "image_names.npy"
LABELS_FILE = PROCESSED_DIR / "labels.csv"  # optional CSV with columns: filename,label
FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
CLASSIFIER_FILE = FAISS_DIR / "classifier.joblib"
