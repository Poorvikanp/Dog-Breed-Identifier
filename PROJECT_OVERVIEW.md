# Project Overview: Dog Breed Classifier (Two Datasets)

## Summary
- **Goal**: Train and serve a dog‑breed image classifier on two datasets, and compare predictions/metrics between them.
- **Behavior**: Uses custom trained models if present. If not, falls back to ImageNet MobileNetV2 for generic predictions.

## Datasets
- **dataset1**: Kaggle Dog Breed Identification (competitions/dog-breed-identification)
- **dataset2**: Stanford Dogs Dataset (datasets/jessicali9530/stanford-dogs-dataset)

## Tech Stack
- **Backend**: FastAPI, Uvicorn, Starlette
- **ML**: TensorFlow 2.20, Keras, NumPy, Pillow
- **Data tooling**: Kaggle API, pandas, scikit-learn
- **UI**: Static HTML/CSS/JS (served by FastAPI)
- **Container/Deploy**: Dockerfile (Python 3.10-slim), Render (Docker)

## Repository Structure
- `ml_breed_classifier/`
  - `backend/`
    - `app.py` — API endpoints, static UI mount
    - `registry.py` — Model loading, preprocessing, prediction, metrics I/O
    - `static/index.html` — Minimal UI to upload image, predict, compare, and view metrics
  - `scripts/`
    - `download_data.py` — Download datasets via Kaggle API
    - `prepare_dataset.py` — Split/prepare datasets
    - `train.py` — Train per dataset, export artifacts
  - `requirements.txt`, `Dockerfile`, `render.yaml`, `README.md`
- `models/`
  - `dataset1/`, `dataset2/` (optional artifacts)
    - `breed_classifier.keras` — Saved model
    - `class_names.txt` — Label mapping (one per line)
    - `metrics.json` — Validation metrics for UI/API
- `data/` — Raw and prepared data (large; local)
- `run_server.bat` — Windows helper (creates venv, installs deps, runs server on port 9000)

## Running Locally
- **Windows quick start**
  - Double‑click or run `run_server.bat` → opens `http://127.0.0.1:9000/`
- **Manual**
  - `pip install -r ml_breed_classifier/requirements.txt`
  - `uvicorn ml_breed_classifier.backend.app:app --reload`
  - Open `http://localhost:8000/`
- Note: README mentions Python 3.10; batch script uses Python 3.11. TensorFlow 2.20 supports both.

## Data & Training Workflow
- **Configure Kaggle (Windows)**
  - Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`
- **Download datasets**
  - `python -m ml_breed_classifier.scripts.download_data --dataset both --out data`
- **Prepare datasets**
  - `python -m ml_breed_classifier.scripts.prepare_dataset --dataset both --inbase data --outbase data --val_ratio 0.15`
- **Train models**
  - `python -m ml_breed_classifier.scripts.train --dataset dataset1 --data_root data --models_root models --epochs 3`
  - `python -m ml_breed_classifier.scripts.train --dataset dataset2 --data_root data --models_root models --epochs 3`
- **Artifacts produced**
  - `models/<dataset>/breed_classifier.keras`
  - `models/<dataset>/class_names.txt`
  - `models/<dataset>/metrics.json`

## API Endpoints
- `GET /` — Serves the UI
- `GET /health` — Health check `{ "status": "ok" }`
- `POST /predict?dataset=dataset1|dataset2`
  - Body: multipart `file` (image)
  - Response: `{ prediction: string, confidence: number }`
- `POST /predict_compare`
  - Body: multipart `file` → predictions from both datasets
- `GET /metrics?dataset=dataset1|dataset2`
  - Returns contents of `metrics.json` if present; 404 otherwise

### Example: Predict via curl (Windows PowerShell)
```powershell
curl -F "file=@.\dog.jpg" "http://localhost:9000/predict?dataset=dataset1"
```

## Docker & Deployment
- **Build**
  - `docker build -t dog-breed-api -f ml_breed_classifier/Dockerfile .`
- **Run**
  - `docker run --rm -p 8000:8000 dog-breed-api`
- **Render**
  - `ml_breed_classifier/render.yaml` defines a Docker web service (health check: `/health`)
  - The app binds to `$PORT` if provided; defaults to `8000`
  - Ensure trained models are available (mount volume or bake into image) for custom predictions

## Operational Notes
- If `models/<dataset>/breed_classifier.keras` is absent, the app falls back to ImageNet predictions (not dog‑breed specific).
- Place trained artifacts under `models/dataset1` and `models/dataset2` for correct dog‑breed outputs.
- `data/` is large and local; exclude from version control.
- CPU Docker image by default; for GPU, switch to a TF CUDA‑compatible base image.
