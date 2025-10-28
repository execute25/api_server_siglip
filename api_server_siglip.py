from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import base64
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logging
import time

# ------------------------------
# Logging and setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(CPU_COUNT, 16)

app = FastAPI(title="SigLIP Classification API v10 - GPU Ready")

# ------------------------------
# Load model
# ------------------------------
print("🔄 Loading SigLIP model...")
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
model.eval()
print(f"✅ Model loaded in {time.time() - start:.2f}s")

# ------------------------------
# Pydantic models
# ------------------------------
class BatchItem(BaseModel):
    keywords: List[str]
    text: str
    base64_image: str

class BatchRequest(BaseModel):
    items: List[BatchItem]

class StyleBatchItem(BaseModel):
    categories_config: dict
    main_category: str
    text: str
    base64_image: str

class StyleBatchRequest(BaseModel):
    items: List[StyleBatchItem]

# ------------------------------
# Utility functions
# ------------------------------
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[1])
    return Image.open(BytesIO(image_data)).convert('RGB')

def process_single_classification(keywords, text, base64_image):
    try:
        image = base64_to_image(base64_image)
        inputs = processor(
            text=keywords, images=[image], padding="max_length", return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "logits_per_image"):
                logits_per_image = outputs.logits_per_image
            else:
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                logits_per_image = image_embeds @ text_embeds.T

            probs = logits_per_image.softmax(dim=1).cpu()
        results = [[kw, float(probs[0][i])] for i, kw in enumerate(keywords)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        raise

def process_style_classification(categories_config, main_category, text, base64_image):
    try:
        image = base64_to_image(base64_image)
        prompts = [f"{main_category} - {subcat}" for subcat in categories_config.get(main_category, [])]

        if not prompts:
            return {"main_category": main_category, "candidates": []}

        inputs = processor(
            text=prompts, images=[image], padding="max_length", return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "logits_per_image"):
                logits_per_image = outputs.logits_per_image
            else:
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                logits_per_image = image_embeds @ text_embeds.T

            probs = logits_per_image.softmax(dim=1).cpu()

        results = [
            {"subcategory": subcat, "score": float(probs[0][i])}
            for i, subcat in enumerate(categories_config.get(main_category, []))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"main_category": main_category, "candidates": results}

    except Exception as e:
        logger.error(f"Error in style classification: {e}")
        raise

# ------------------------------
# Async batch processing
# ------------------------------
async def process_batch_parallel(items, fn):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [loop.run_in_executor(executor, fn, *args) for args in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "v10-gpu-full",
        "device": str(device),
        "gpu_enabled": torch.cuda.is_available(),
        "workers": MAX_WORKERS
    }

@app.post("/classify-batch")
async def classify_batch(request: BatchRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")
    logger.info(f"🚀 classify-batch on {device}, {len(request.items)} items")
    start = time.time()
    formatted_items = [(i.keywords, i.text, i.base64_image) for i in request.items]
    results = await process_batch_parallel(formatted_items, process_single_classification)
    return {
        "device": str(device),
        "results": results,
        "time": time.time() - start
    }

@app.post("/classify-style-batch")
async def classify_style_batch(request: StyleBatchRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")
    logger.info(f"🎨 classify-style-batch on {device}, {len(request.items)} items")
    start = time.time()
    formatted_items = [
        (i.categories_config, i.main_category, i.text, i.base64_image)
        for i in request.items
    ]
    results = await process_batch_parallel(formatted_items, process_style_classification)
    return {
        "device": str(device),
        "results": results,
        "time": time.time() - start
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting API on {device}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
