import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, Dinov2Model
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

def get_text_embeds(clip_model, text_inputs):
    text_outputs = clip_model.text_model(**text_inputs)
    pooled_output = text_outputs[1]
    text_features = clip_model.text_projection(pooled_output)
    return text_features

def get_clip_image_embeds(clip_model, image_inputs):
    vision_outputs = clip_model.vision_model(**image_inputs)
    pooled_output = vision_outputs[1]
    image_features = clip_model.visual_projection(pooled_output)
    return image_features

def get_dinov2_image_embeds(dinov2_model, image_inputs):
    outputs = dinov2_model(**image_inputs)
    return outputs.pooler_output

def prepare_dataset():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load CLIP model (for category filtering)
    clip_model_id = "openai/clip-vit-base-patch32"
    print(f"Loading {clip_model_id}...")
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    
    # 2. Load DinoV2 model (for training feature extraction)
    dinov2_model_id = "facebook/dinov2-base"
    print(f"Loading {dinov2_model_id}...")
    dino_processor = AutoImageProcessor.from_pretrained(dinov2_model_id)
    dino_model = Dinov2Model.from_pretrained(dinov2_model_id).to(device)
    
    # Define prompts for CLIP
    cat_prompts = [
        "a photo of a hand or fingers",
        "a photo of a human face",
        "an image with text, writing, or letters on a screen or sign"
    ]
    
    hands_clean = "natural hand with normal fingers and joints"
    hands_art   = "deformed, mutated, melted, or malformed hand with extra fingers or joints"
    faces_clean = "a normal, correctly proportioned human face"
    faces_art   = "a distorted, warped, or deformed human face"
    text_clean  = "clear, legible, readable text"
    text_art    = "garbled, distorted, illegible, or gibberish text"
    obj_clean   = "a normal, realistic, well-formed object or scene"
    obj_art     = "a warped, melted, glitchy, or physically impossible shape or object"
    
    all_prompts = cat_prompts + [
        hands_clean, hands_art,
        faces_clean, faces_art,
        text_clean, text_art,
        obj_clean, obj_art
    ]
    
    # 3. Stream dataset
    print("Streaming dataset from Hugging Face...")
    ds = load_dataset('Parveshiiii/AI-vs-Real', split='train', streaming=True)
    iterator = iter(ds)
    
    bins = {
        'hands': [],
        'faces': [],
        'text': [],
        'general': []
    }
    
    # We will scan images, categorize them using CLIP, and store their PIL Images.
    # After scanning, we will extract DinoV2 features only for the selected subset of images.
    max_scan = 1500
    print(f"Scanning up to {max_scan} AI images to select candidates...")
    pbar = tqdm(total=max_scan, desc="Scanning AI images")
    
    text_inputs = clip_processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = get_text_embeds(clip_model, text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
    num_artifact_prompts = 4 # hands, faces, text, obj
    scan_count = 0
    
    while scan_count < max_scan:
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        img = item['image']
        label = item['binary_label']
        
        if label != 1:
            continue
            
        scan_count += 1
        pbar.update(1)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize for CLIP
        img_resized_clip = img.resize((224, 224))
        
        inputs = clip_processor(images=img_resized_clip, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = get_clip_image_embeds(clip_model, inputs)
            image_features_norm = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            similarities = torch.matmul(image_features_norm, text_embeds.t()) * 100
            sims = similarities[0].cpu().numpy()
            
        has_hand = sims[0]
        has_face = sims[1]
        has_text = sims[2]
        
        cat = 'general'
        max_cat_score = 20.0
        
        if has_hand > max_cat_score:
            cat = 'hands'
            max_cat_score = has_hand
        if has_face > max_cat_score:
            if has_face > max_cat_score:
                cat = 'faces'
                max_cat_score = has_face
        if has_text > max_cat_score:
            if has_text > max_cat_score:
                cat = 'text'
                max_cat_score = has_text
                
        if cat == 'hands':
            score_diff = sims[4] - sims[3]
        elif cat == 'faces':
            score_diff = sims[6] - sims[5]
        elif cat == 'text':
            score_diff = sims[8] - sims[7]
        else:
            score_diff = sims[10] - sims[9]
            
        bins[cat].append({
            'image': img, # keep original image for DinoV2 feature extraction later
            'score_diff': score_diff
        })
        
    pbar.close()
    
    # Select balanced samples
    selected_clean_images = []
    selected_art_images = []
    target_per_bin = 75
    
    print("\nCategory Bin Sizes:")
    for cat, items in bins.items():
        print(f"  {cat}: {len(items)} candidates")
        
    for cat, items in bins.items():
        if len(items) < target_per_bin * 2:
            num_take = len(items) // 2
            print(f"  Warning: Bin {cat} has only {len(items)} items. Splitting {num_take}/{num_take}.")
        else:
            num_take = target_per_bin
            
        sorted_items = sorted(items, key=lambda x: x['score_diff'])
        
        clean_items = sorted_items[:num_take]
        art_items = sorted_items[-num_take:]
        
        selected_clean_images.extend([x['image'] for x in clean_items])
        selected_art_images.extend([x['image'] for x in art_items])
        
    print(f"\nExtracting DinoV2 features for {len(selected_clean_images)} clean and {len(selected_art_images)} artifact images...")
    
    X_clean_list = []
    X_art_list = []
    
    # Extract DinoV2 features
    for img in tqdm(selected_clean_images, desc="Extracting Clean DinoV2"):
        inputs = dino_processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
        with torch.no_grad():
            emb = get_dinov2_image_embeds(dino_model, inputs)
            X_clean_list.append(emb[0].cpu().numpy())
            
    for img in tqdm(selected_art_images, desc="Extracting Artifact DinoV2"):
        inputs = dino_processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
        with torch.no_grad():
            emb = get_dinov2_image_embeds(dino_model, inputs)
            X_art_list.append(emb[0].cpu().numpy())
            
    X_clean = np.stack(X_clean_list)
    y_clean = np.zeros(len(X_clean))
    
    X_artifact = np.stack(X_art_list)
    y_artifact = np.ones(len(X_artifact))
    
    X = np.vstack([X_clean, X_artifact])
    y = np.concatenate([y_clean, y_artifact])
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    np.savez('data/train_features.npz', features=X, labels=y)
    print("DinoV2 balanced dataset prepared and saved to data/train_features.npz!")

if __name__ == '__main__':
    prepare_dataset()
