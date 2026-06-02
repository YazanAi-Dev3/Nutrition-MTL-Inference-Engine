import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch

from app.config import load_settings
from app.model import load_inference_model
from app.data import load_metadata_database, build_visual_knowledge_base
from app.pipeline import process_input_image, run_hybrid_pipeline

settings = load_settings()

DATASET_EGYPTIAN_CSV = settings["paths"]["dataset_egyptian_csv"]
MODEL_WEIGHTS_PATH = settings["paths"]["model_weights_path"]
REFERENCE_IMAGES_DIR = settings["paths"]["reference_images_dir"]
KNOWLEDGE_BASE_PATH = settings["paths"]["knowledge_base_path"]

SERVER_TITLE = settings["server"]["title"]
SERVER_HOST = settings["server"]["host"]
SERVER_PORT = settings["server"]["port"]

MATCH_THRESHOLD = settings["pipeline"]["match_threshold"]
TOP_K = settings["pipeline"]["k"]

NUM_INGREDIENTS = settings["model"]["num_ingredients"]
USE_CUDA_IF_AVAILABLE = settings["device"]["use_cuda_if_available"]

IMAGE_SIZE = tuple(settings["image"]["size"])
NORMALIZE_MEAN = settings["image"]["normalize_mean"]
NORMALIZE_STD = settings["image"]["normalize_std"]

device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

# Global Initialization
metadata_df = load_metadata_database(DATASET_EGYPTIAN_CSV)
inference_model = load_inference_model(MODEL_WEIGHTS_PATH, device, num_ingredients=NUM_INGREDIENTS)
kb_embeddings = build_visual_knowledge_base(
    inference_model,
    REFERENCE_IMAGES_DIR,
    KNOWLEDGE_BASE_PATH,
    device,
    image_size=IMAGE_SIZE,
    normalize_mean=NORMALIZE_MEAN,
    normalize_std=NORMALIZE_STD
)

app = FastAPI(title=SERVER_TITLE)


@app.post("/analyze-dish")
async def analyze_dish(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_tensor = process_input_image(
            contents,
            image_size=IMAGE_SIZE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD
        )

        if img_tensor is None:
            return JSONResponse(status_code=400, content={"Status": "Error", "Message": "Invalid image format."})

        result = run_hybrid_pipeline(
            img_tensor=img_tensor,
            model=inference_model,
            knowledge_base=kb_embeddings,
            metadata_df=metadata_df,
            device=device,
            match_threshold=MATCH_THRESHOLD,
            k=TOP_K
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"Status": "Error", "Message": str(e)})


nest_asyncio.apply()


def run():
    print(f"Starting API Server on http://{SERVER_HOST}:{SERVER_PORT} ...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
