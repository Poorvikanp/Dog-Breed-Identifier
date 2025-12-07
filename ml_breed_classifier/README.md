# Dog Breed Classifier (Two Datasets)

A clean starter for training and serving a dog breed classifier on two datasets (Kaggle dog-breed-identification and Stanford Dogs) with FastAPI and TensorFlow.

## Features
- Train two separate models (`dataset1`, `dataset2`) and compare
- REST API: predict per dataset, compare, and view metrics
- Simple web UI for uploading an image and viewing results
- Dockerfile for easy deployment

## Setup
1. Python 3.10 recommended. Install deps:
```
pip install -r ml_breed_classifier/requirements.txt
```
2. Configure Kaggle API: place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

## Data
```
python -m ml_breed_classifier.scripts.download_data --dataset both --out data
python -m ml_breed_classifier.scripts.prepare_dataset --dataset both --inbase data --outbase data --val_ratio 0.15
```

## Train
```
python -m ml_breed_classifier.scripts.train --dataset dataset1 --data_root data --models_root models --epochs 3
python -m ml_breed_classifier.scripts.train --dataset dataset2 --data_root data --models_root models --epochs 3
```

## Run API + UI
```
uvicorn ml_breed_classifier.backend.app:app --reload
```
Open http://localhost:8000/

## Docker
```
docker build -t dog-breed-api .
docker run --rm -p 8000:8000 dog-breed-api
```
