# AI Elephant Identification

A two-stage computer-vision project for identifying individual elephants: YOLO detects an elephant, then a ResNet-50 embedding model supports identity matching. The project received the Team Spirit Special Award at the Bangkok University AI Hackathon 2025.

## Technical approach

The identification implementation uses a ResNet-50 backbone, a 128-dimensional embedding layer, normalized feature vectors, and cosine similarity against stored reference embeddings. The project training approach uses triplet loss.

## Repository guide

- `python/obj_detection.py`: detection component.
- `python/identification.py`: embedding model and reference matching.
- `python/main.py`: application integration.
- `python/train/`: training work.
- `models/`: model assets.
- `requirements.txt` and `guide.txt`: dependency and run references.

## Getting started

Review `requirements.txt` in an isolated Python environment. Check model, reference-embedding, and image paths before running: `python/identification.py` currently contains local Windows-style example paths and loads checkpoints at module scope. Those paths must match your local files. This documentation update does not change code or claim a verified clean installation.

## Contribution, data, and evaluation

Kawin Chamnanchak designed and trained the detection-and-embedding pipeline as part of a team. Preserve the project’s contributor attribution. Use only elephant images and model assets you have permission to use and redistribute. Evaluate with held-out individuals/images and report the split, matching threshold, and metrics explicitly; no unverified accuracy claim is made here.
