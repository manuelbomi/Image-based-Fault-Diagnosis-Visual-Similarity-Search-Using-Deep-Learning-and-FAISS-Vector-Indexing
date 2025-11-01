# Enterprise Image Fault Diagnosis & Visual Similarity Search

## Overview
This repository demonstrates an enterprise-ready pipeline to:
- extract image embeddings using a pre-trained VGG16 model,
- index embeddings with FAISS for fast similarity search,
- train a simple classifier on the extracted embeddings to predict fault types,
- provide a Streamlit UI to query similar images and view results.

The intended workflow:
1. Place customer-returned fault images in `data/raw/` and a `labels.csv` mapping (optional).
2. Run `src/extract_embeddings.py` to extract and save embeddings.
3. Run `src/build_faiss_index.py` to create and persist a FAISS index.
4. Run `src/train_classifier.py` to train a classifier on the embeddings + labels.
5. Use `src/query_similar_images.py` or `app/streamlit_visual_search.py` to query the index.

## Repo structure
```
enterprise-image-fault-diagnosis/
│
├─ data/
│   ├─ raw/                     # customer-returned images (fault samples)
│   ├─ processed/               # preprocessed + augmented images (optional)
│   └─ faiss_index/             # saved FAISS index + label mappings
│
├─ notebooks/
│   ├─ 01_explore_dataset.ipynb
│   ├─ 02_train_vgg16_classifier.ipynb
│   └─ 03_build_faiss_index.ipynb
│
├─ src/
│   ├─ config.py
│   ├─ extract_embeddings.py
│   ├─ build_faiss_index.py
│   ├─ query_similar_images.py
│   └─ train_classifier.py
│
├─ app/
│   └─ streamlit_visual_search.py
│
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Notes
- This project uses `VGG16` from Keras as a feature extractor. You can swap it with modern backbones (EfficientNet, ConvNeXt, ViT).
- `faiss` is used for high-performance nearest-neighbor search. Use `faiss-gpu` for GPU acceleration.
- Embeddings and indices are saved under `data/faiss_index/`.
- The code includes argument parsing so you can adapt to your environment.

See the `src/` scripts for usage examples.

