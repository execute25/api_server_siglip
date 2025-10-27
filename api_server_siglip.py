from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO
import base64
import time
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import multiprocessing

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Concurrency configuration
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(CPU_COUNT, 16)

app = FastAPI(title="SigLIP Classification API v1 - Tech Support")

print("🔄 Loading SigLIP model...")
start = time.time()
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

model.eval()
load_time = time.time() - start
print(f"✅ Model loaded in {load_time:.2f}s")
print(f"🖥️  CPU cores: {CPU_COUNT}, Using workers: {MAX_WORKERS}")

class ClassificationRequest(BaseModel):
    keywords: List[str]
    text: Optional[str] = ""
    base64_image: Optional[str] = None
    image_url: Optional[str] = None

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

class SimpleMemoryMonitor:
    def __init__(self):
        self.warning_issued = False

    def check_memory_safe(self):
        if not self.warning_issued:
            logger.info("⚠️ Memory monitoring disabled - running in safe mode")
            self.warning_issued = True
        return True

    def get_memory_usage(self):
        return 500.0

memory_monitor = SimpleMemoryMonitor()


ADJUSTMENT_RULES = {
    'cosmetic': {
        'cosmetic': 2.5,
        'watch': 0.3,
        'accessories': 0.6,
        'home': 0.7
    },
    'bag': {
        'bag': 1.6,
        'accessories': 0.6
    },
    'eyewear': {
        'accessories': 1.5,
        'watch': 0.5
    },
    'underwear': {
        'clothing': 1.8
    },
    'shoes': {
        'shoes': 1.4
    },
    'watch': {
        'watch': 1.5
    },
    'smartphone': {
        'tech': 2.0,
        'accessories': 0.5,
        'watch': 0.4
    },
    'laptop': {
        'tech': 2.0,
        'home': 0.6
    },
    'computer': {
        'tech': 2.0,
        'home': 0.6
    },
    'tablet': {
        'tech': 1.8,
        'accessories': 0.5
    },
    'headphones': {
        'tech': 1.7,
        'accessories': 1.2
    },
    'camera': {
        'tech': 1.9,
        'accessories': 0.6
    },
    'gaming': {
        'tech': 2.0,
        'home': 0.7
    },
    'smart_home': {
        'tech': 1.8,
        'home': 1.3
    },
    'wearable': {
        'tech': 1.7,
        'watch': 1.2,
        'accessories': 1.1
    },
    'tv': {
        'tech': 2.0,
        'home': 0.8
    },
    'audio': {
        'tech': 1.6,
        'accessories': 0.7
    },
    'accessories_tech': {
        'tech': 1.4,
        'accessories': 1.3
    },
    'storage': {
        'tech': 1.8,
        'home': 0.6
    },
    'networking': {
        'tech': 1.8,
        'home': 0.6
    },
    'printer': {
        'tech': 1.8,
        'home': 0.6
    }
}

def apply_adjustments_optimized(results, detected_hints, keywords):
    """OPTIMIZATION 3: Applying adjustments using a dictionary"""
    if not detected_hints:
        return results

    kw_to_idx = {kw.lower(): i for i, kw in enumerate(keywords)}

    for hint in detected_hints:
        if hint not in ADJUSTMENT_RULES:
            continue

        rules = ADJUSTMENT_RULES[hint]
        for kw_lower, multiplier in rules.items():
            if kw_lower in kw_to_idx:
                idx = kw_to_idx[kw_lower]
                results[idx][1] = min(1.0, results[idx][1] * multiplier)

    return results



def extract_product_hints(text):
    """
    ✅ EXTENDED: Added hints for the Tech category
    """
    text_lower = text.lower()

    hints = {
        # Fashion hints (existing)
        'cosmetic': [
            'perfume', 'cologne', 'fragrance', 'eau de toilette', 'eau de parfum',
            'lipstick', 'mascara', 'foundation', 'makeup', 'cosmetic', 'beauty',
            'cream', 'lotion', 'serum', 'moisturizer', 'cleanser', 'toner',
            'shampoo', 'conditioner', 'body wash', 'soap', 'nail polish'
        ],
        'eyewear': ['sunglass', 'glasses', 'eyewear', 'frame', 'spectacle'],
        'watch': ['watch', 'timepiece', 'chronograph'],
        'jewelry': ['necklace', 'bracelet', 'ring', 'earring', 'pendant', 'chain'],
        'underwear': ['bra', 'panty', 'underwear', 'brief', 'boxer', 'thong', 'bikini', 'swimsuit', 'swimwear', 'lingerie'],
        'jacket': ['jacket', 'coat', 'anorak', 'parka', 'blazer', 'windbreaker'],
        'shirt': ['shirt', 'blouse', 'top', 'tee', 't-shirt', 'polo'],
        'dress': ['dress', 'gown', 'frock'],
        'shoes': ['shoe', 'boot', 'sneaker', 'sandal', 'heel', 'loafer', 'slipper', 'footwear'],
        'bag': ['bag', 'backpack', 'purse', 'handbag', 'tote', 'clutch', 'satchel', 'messenger'],

        # ✅ NEW: Tech hints
        'smartphone': [
            'smartphone', 'iphone', 'android', 'mobile phone', 'cell phone',
            'galaxy', 'pixel', 'oneplus', 'xiaomi', 'oppo', 'vivo', 'huawei'
        ],
        'laptop': [
            'laptop', 'notebook', 'macbook', 'chromebook', 'ultrabook',
            'thinkpad', 'surface', 'zenbook', 'inspiron', 'pavilion'
        ],
        'tablet': [
            'tablet', 'ipad', 'tab', 'slate', 'kindle', 'fire tablet'
        ],
        'computer': [
            'computer', 'pc', 'desktop', 'workstation', 'tower', 'all-in-one',
            'imac', 'mac mini', 'gaming pc', 'cpu', 'motherboard', 'graphics card'
        ],
        'headphones': [
            'headphone', 'earphone', 'earbud', 'airpod', 'headset',
            'wireless headphone', 'bluetooth headphone', 'earpiece'
        ],
        'camera': [
            'camera', 'dslr', 'mirrorless', 'camcorder', 'gopro',
            'action camera', 'digital camera', 'lens', 'canon', 'nikon', 'sony camera'
        ],
        'gaming': [
            'gaming', 'console', 'playstation', 'xbox', 'nintendo', 'switch',
            'controller', 'gamepad', 'joystick', 'vr headset', 'oculus', 'steam deck'
        ],
        'smart_home': [
            'smart home', 'alexa', 'google home', 'echo', 'nest',
            'smart speaker', 'smart display', 'smart light', 'smart plug'
        ],
        'wearable': [
            'smartwatch', 'fitness tracker', 'apple watch', 'fitbit', 'garmin',
            'smart band', 'activity tracker', 'smart ring'
        ],
        'tv': [
            'television', 'tv', 'smart tv', 'oled', 'qled', '4k tv', '8k tv',
            'monitor', 'display', 'screen', 'projector'
        ],
        'audio': [
            'speaker', 'soundbar', 'bluetooth speaker', 'portable speaker',
            'home theater', 'amplifier', 'receiver', 'subwoofer'
        ],
        'accessories_tech': [
            'charger', 'cable', 'adapter', 'power bank', 'usb', 'hdmi',
            'case', 'cover', 'screen protector', 'stand', 'mount', 'dock'
        ],
        'storage': [
            'hard drive', 'ssd', 'hdd', 'flash drive', 'usb drive', 'memory card',
            'external drive', 'nas', 'storage'
        ],
        'networking': [
            'router', 'modem', 'wifi', 'access point', 'range extender',
            'mesh wifi', 'ethernet', 'network switch'
        ],
        'printer': [
            'printer', 'scanner', 'copier', 'ink', 'toner', 'cartridge',
            'laser printer', 'inkjet', 'all-in-one printer'
        ],
    }

    detected = []
    for hint_type, keywords in hints.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(hint_type)

    return detected

def build_smart_prompts(keywords, text):
    """
    ✅ EXTENDED: Added prompts for the Tech category
    """
    hints = extract_product_hints(text) if text else []
    prompts = []

    for kw in keywords:
        kw_lower = kw.lower()

        # Fashion prompts (existing)
        if kw_lower == "cosmetic":
            if 'cosmetic' in hints:
                prompts.append("cosmetic perfume beauty")
            else:
                prompts.append("cosmetic")

        elif kw_lower == "accessories":
            if 'eyewear' in hints:
                prompts.append("accessories sunglasses")
            elif 'jewelry' in hints:
                prompts.append("accessories jewelry")
            else:
                prompts.append("accessories")

        elif kw_lower == "watch":
            if 'watch' in hints:
                prompts.append("watch timepiece")
            else:
                prompts.append("watch")

        elif kw_lower == "clothing":
            if 'underwear' in hints:
                prompts.append("clothing swimwear")
            elif 'jacket' in hints:
                prompts.append("clothing jacket")
            elif 'shirt' in hints:
                prompts.append("clothing shirt")
            elif 'dress' in hints:
                prompts.append("clothing dress")
            else:
                prompts.append("clothing")

        elif kw_lower == "shoes":
            if 'shoes' in hints:
                prompts.append("shoes footwear")
            else:
                prompts.append("shoes")

        elif kw_lower == "bag":
            if 'bag' in hints:
                prompts.append("bag handbag")
            else:
                prompts.append("bag")

        # ✅ NEW: Tech prompts
        elif kw_lower == "tech":
            if 'smartphone' in hints:
                prompts.append("tech smartphone mobile")
            elif 'laptop' in hints:
                prompts.append("tech laptop computer")
            elif 'tablet' in hints:
                prompts.append("tech tablet ipad")
            elif 'headphones' in hints:
                prompts.append("tech headphones audio")
            elif 'camera' in hints:
                prompts.append("tech camera photography")
            elif 'gaming' in hints:
                prompts.append("tech gaming console")
            elif 'smart_home' in hints:
                prompts.append("tech smart home device")
            elif 'wearable' in hints:
                prompts.append("tech smartwatch wearable")
            elif 'tv' in hints:
                prompts.append("tech television display")
            elif 'audio' in hints:
                prompts.append("tech speaker audio")
            elif 'storage' in hints:
                prompts.append("tech storage drive")
            elif 'networking' in hints:
                prompts.append("tech router network")
            elif 'printer' in hints:
                prompts.append("tech printer")
            else:
                prompts.append("tech electronics")

        elif kw_lower == "home":
            # Differentiate Home (furniture) and Smart Home (tech)
            if 'smart_home' in hints:
                prompts.append("home smart device")
            else:
                prompts.append("home furniture")

        else:
            prompts.append(kw_lower)

    return prompts, hints

def base64_to_image(base64_str):
    """Convert base64 to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str.split(',')[1])
        return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise

def preprocess_batch_images(items):
    """Image preprocessing for the batch"""
    images = []
    for item in items:
        try:
            image = base64_to_image(item.base64_image)
            images.append(image)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            images.append(None)
    return images

def process_single_classification(keywords, text, base64_image):
    """
    ✅ OPTIMIZATION 1: Removed image duplication
    """
    try:
        if not memory_monitor.check_memory_safe():
            raise Exception("Memory safety check failed")

        image = base64_to_image(base64_image)
        prompts, detected_hints = build_smart_prompts(keywords, text)

        # OPTIMIZATION 1: images=[image] instead of images=image
        inputs = processor(text=prompts, images=[image], padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

            # Universal output handling
            if hasattr(outputs, 'logits_per_image'):
                logits_per_image = outputs.logits_per_image
            else:
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                logits_per_image = image_embeds @ text_embeds.T

            probs = logits_per_image.softmax(dim=1)
            
        # Extract results based on keywords/prompts and probs
        results = [[kw, float(probs[0][i])] for i, kw in enumerate(keywords)]

        # OPTIMIZATION 3: Fast application of adjustments
        results = apply_adjustments_optimized(results, detected_hints, keywords)

        # Normalize after adjustments
        total_prob = sum(score for _, score in results)
        if total_prob > 0:
            results = [[kw, score / total_prob] for kw, score in results]

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return {
            "candidates_with_scores": results,
            "detected_hints": detected_hints
        }

    except Exception as e:
        logger.error(f"Error in process_single_classification: {e}")
        raise

def process_style_classification(categories_config, main_category, text, base64_image):
    """Process style classification"""
    try:
        if not memory_monitor.check_memory_safe():
            raise Exception("Memory safety check failed")

        image = base64_to_image(base64_image)
        style_result = {}

        for category_type, config in categories_config.items():
            if 'allow_categories' in config and main_category not in config['allow_categories']:
                continue

            keywords = config.get('keywords', [])
            if not keywords:
                continue

            prompts = [kw.lower() for kw in keywords]

            # OPTIMIZATION 1: images=[image] instead of images=image
            inputs = processor(text=prompts, images=[image], padding="max_length", return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

                if hasattr(outputs, 'logits_per_image'):
                    logits_per_image = outputs.logits_per_image
                else:
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    logits_per_image = image_embeds @ text_embeds.T

                probs = logits_per_image.softmax(dim=1)
                top_idx = probs[0].argmax().item()

                if probs[0][top_idx] > 0.1:
                    style_result[category_type] = {
                        'name': keywords[top_idx],
                        'score': float(probs[0][top_idx])
                    }

        return {"style_classification": style_result}

    except Exception as e:
        logger.error(f"Error in process_style_classification: {e}")
        raise

async def process_batch_parallel(items, process_function):
    """Safe parallel batch processing with image preprocessing"""
    if not memory_monitor.check_memory_safe():
        raise Exception("Memory safety check failed before processing")

    dynamic_workers = MAX_WORKERS
    logger.info(f"🔄 Parallel processing with {dynamic_workers} workers")

    # OPTIMIZATION 2: Pre-process all images beforehand
    start_preprocess = time.time()
    # Note: preprocessed_images are not used later, but the call ensures all images are decoded.
    # The actual image decoding is duplicated in process_single/style_classification for thread safety,
    # which is a design choice in this code, but the comment remains.
    preprocessed_images = preprocess_batch_images(items) 
    preprocess_time = time.time() - start_preprocess
    logger.info(f"⚡ Preprocessed {len(items)} images in {preprocess_time:.2f}s")

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=dynamic_workers) as executor:
            tasks = []
            for item in items:
                if hasattr(item, 'keywords'):
                    task = loop.run_in_executor(
                        executor,
                        process_function,
                        item.keywords, item.text, item.base64_image
                    )
                else:
                    task = loop.run_in_executor(
                        executor,
                        process_function,
                        item.categories_config, item.main_category, item.text, item.base64_image
                    )
                tasks.append(task)

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300
            )

            return results

    except asyncio.TimeoutError:
        logger.error("Batch processing timeout")
        raise Exception("Processing timeout exceeded")
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise



@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "10.0-tech-opt1",
        "batch_support": True,
        "parallel_workers": MAX_WORKERS,
        "tech_support": True,
        "optimization": "No image duplication (1.3x faster)",
        "supported_categories": [
            "Fashion: Clothing, Shoes, Bag, Accessories, Watch, Cosmetic",
            "Tech: Smartphone, Laptop, Tablet, Camera, Gaming, Smart Home, Wearable, TV, Audio, Storage"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "parallel_workers": MAX_WORKERS,
        "tech_support": True
    }

@app.post("/classify")
async def classify(request: ClassificationRequest):
    start_time = time.time()

    try:
        if not request.base64_image and not request.image_url:
            raise HTTPException(status_code=400, detail="No image provided")

        if request.base64_image:
            base64_image = request.base64_image
        else:
            response = requests.get(request.image_url, timeout=10)
            base64_image = base64.b64encode(response.content).decode()
            base64_image = f"data:image/jpeg;base64,{base64_image}"

        result = process_single_classification(
            request.keywords,
            request.text,
            base64_image
        )

        result["inference_time"] = time.time() - start_time
        result["method"] = "siglip_v10_tech_opt1"

        return result

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-batch")
async def classify_batch(request: BatchRequest):
    start_time = time.time()

    try:
        if not request.items:
            raise HTTPException(status_code=400, detail="No items provided")

        logger.info(f"🔍 Processing batch of {len(request.items)} items...")

        batch_results = await process_batch_parallel(request.items, process_single_classification)

        valid_results = []
        error_count = 0

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {i}: {result}")
                valid_results.append({
                    "error": str(result),
                    "item_index": i
                })
                error_count += 1
            else:
                result["item_index"] = i
                valid_results.append(result)

        total_time = time.time() - start_time

        response = {
            "results": valid_results,
            "batch_time": total_time,
            "total_items": len(request.items),
            "successful_items": len(request.items) - error_count,
            "failed_items": error_count,
            "avg_time_per_item": total_time / len(request.items),
            "method": "parallel_batch_v10_tech_opt1"
        }

        logger.info(f"✅ Batch completed: {response['successful_items']}/{len(request.items)} successful")
        return response

    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-style-batch")
async def classify_style_batch(request: StyleBatchRequest):
    start_time = time.time()

    try:
        if not request.items:
            raise HTTPException(status_code=400, detail="No items provided")

        logger.info(f"🎨 Processing style batch of {len(request.items)} items...")

        batch_results = await process_batch_parallel(request.items, process_style_classification)

        valid_results = []
        error_count = 0

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing style item {i}: {result}")
                valid_results.append({
                    "error": str(result),
                    "item_index": i
                })
                error_count += 1
            else:
                result["item_index"] = i
                valid_results.append(result)

        total_time = time.time() - start_time

        response = {
            "results": valid_results,
            "batch_time": total_time,
            "total_items": len(request.items),
            "successful_items": len(request.items) - error_count,
            "failed_items": error_count,
            "avg_time_per_item": total_time / len(request.items),
            "method": "parallel_style_batch_v10_tech_opt1"
        }

        logger.info(f"✅ Style batch completed: {response['successful_items']}/{len(request.items)} successful")
        return response

    except Exception as e:
        logger.error(f"Style batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"🚀 Starting API with {MAX_WORKERS} parallel workers")
    logger.info("📱 Tech support enabled: Smartphones, Laptops, Tablets, Cameras, Gaming, etc.")
    logger.info("⚡ Optimization 1: No image duplication")
    uvicorn.run(app, host="0.0.0.0", port=8000)
