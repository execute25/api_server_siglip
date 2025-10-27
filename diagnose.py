import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO

print("Loading model...")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
model.eval()

# Problem image
url = "https://cdn.realry.co/W-wKF_5eTdszEAmwtC1-8vN6esQ=/300x300/products/o/fdf30e34bc999c1ffc81dbd50a87cb72b700f0e3.jpeg"
response = requests.get(url, timeout=10)
image = Image.open(BytesIO(response.content)).convert('RGB')

print("\n" + "="*70)
print("🔍 DIAGNOSTIC ANALYSIS")
print("="*70)
print(f"Image: {url}")
print(f"Product: Lyle & Scott Green Anorak Jacket")
print("="*70)

# Test 1: Different prompt styles
print("\n📊 TEST 1: Different Prompt Styles")
print("-"*70)

categories = ["Clothing", "Shoes", "Bag", "Home", "Tech", "Accessories"]

prompt_styles = {
    "A. Single words": categories,
    
    "B. Generic phrases": [
        "clothing",
        "shoes", 
        "bag",
        "home goods",
        "technology",
        "accessories"
    ],
    
    "C. 'This is' format": [
        "this is clothing",
        "this is shoes",
        "this is a bag",
        "this is home goods",
        "this is technology",
        "this is accessories"
    ],
    
    "D. 'Photo of' format": [
        "a photo of clothing",
        "a photo of shoes",
        "a photo of a bag",
        "a photo of home goods",
        "a photo of technology",
        "a photo of accessories"
    ],
    
    "E. Specific items": [
        "a jacket coat outerwear",
        "footwear shoes boots",
        "a handbag purse",
        "furniture decor",
        "electronics devices",
        "jewelry watches"
    ],
    
    "F. Product category": [
        "product category: apparel",
        "product category: footwear",
        "product category: bags",
        "product category: home",
        "product category: electronics",
        "product category: accessories"
    ]
}

for style_name, prompts in prompt_styles.items():
    inputs = processor(text=prompts, images=image, padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    print(f"\n{style_name}:")
    for i, cat in enumerate(categories):
        score = probs[i] * 100
        bar = "█" * int(score / 5)
        print(f"  {cat:<15} {score:5.1f}% {bar}")
    
    best_idx = probs.argmax()
    print(f"  → Winner: {categories[best_idx]} ({probs[best_idx]*100:.1f}%)")

# Test 2: What does the model "see"?
print("\n" + "="*70)
print("📊 TEST 2: Free-form descriptions")
print("-"*70)

descriptions = [
    "a green jacket",
    "green clothing",
    "outerwear",
    "a coat",
    "sportswear",
    "athletic wear",
    "green shoes",
    "sneakers",
    "winter jacket",
    "anorak",
]

inputs = processor(text=descriptions, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

sorted_results = sorted(zip(descriptions, probs), key=lambda x: x[1], reverse=True)

print("\nTop matches:")
for desc, score in sorted_results[:10]:
    print(f"  {desc:<20} {score*100:5.1f}%")

# Test 3: Check if it's the color causing confusion
print("\n" + "="*70)
print("📊 TEST 3: Color analysis")
print("-"*70)

color_test = [
    "green clothing",
    "green shoes",
    "green bag",
    "green jacket",
    "green footwear",
]

inputs = processor(text=color_test, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

print("\nColor-specific prompts:")
for i, desc in enumerate(color_test):
    print(f"  {desc:<20} {probs[i]*100:5.1f}%")

# Test 4: Hierarchical approach
print("\n" + "="*70)
print("📊 TEST 4: Hierarchical Classification")
print("-"*70)

# Step 1: Is it fashion or not?
step1_prompts = [
    "fashion apparel clothing",
    "technology electronics",
    "home furniture decor"
]

inputs = processor(text=step1_prompts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

print("\nStep 1 - Domain:")
for i, prompt in enumerate(step1_prompts):
    print(f"  {prompt:<30} {probs[i]*100:5.1f}%")

# Step 2: What type of fashion?
if probs[0] > 0.5:  # If it's fashion
    step2_prompts = [
        "jacket coat outerwear",
        "shirt top blouse",
        "pants trousers jeans",
        "dress skirt",
        "shoes boots footwear"
    ]
    
    inputs = processor(text=step2_prompts, images=image, padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    print("\nStep 2 - Fashion type:")
    for i, prompt in enumerate(step2_prompts):
        print(f"  {prompt:<30} {probs[i]*100:5.1f}%")

print("\n" + "="*70)
