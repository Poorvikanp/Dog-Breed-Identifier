from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
from io import BytesIO

from .registry import predict as predict_image, load_metrics

app = FastAPI(title="Dog Breed Classifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(dataset: str = Query(..., description="Dataset key, e.g., dataset1 or dataset2"), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = predict_image(image, dataset_key=dataset)
    return JSONResponse(result)


@app.get("/metrics")
def metrics(dataset: str = Query(..., description="Dataset key, e.g., dataset1 or dataset2")):
    data = load_metrics(dataset)
    if data is None:
        raise HTTPException(status_code=404, detail="Metrics not found for dataset")
    return JSONResponse(data)


@app.post("/predict_compare")
async def predict_compare(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    out = {ds: predict_image(image, dataset_key=ds) for ds in ("dataset1", "dataset2")}
    return JSONResponse(out)
