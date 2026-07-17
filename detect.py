import os
import argparse
import json
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Dinov2Model
import easyocr

def get_dinov2_image_embeds(model, image_inputs):
    outputs = model(**image_inputs)
    return outputs.pooler_output

def main():
    parser = argparse.ArgumentParser(description="AI-generated Video Frame Artifact Detector")
    parser.add_argument("--input", required=True, help="Path to the input folder containing images")
    parser.add_argument("--output", required=True, help="Path to save the output results.json file")
    args = parser.parse_args()
    
    input_folder = args.input
    output_json = args.output
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
        
    # 1. Load Model Assets
    model_path = os.path.join(os.path.dirname(__file__), "model_assets", "model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model assets not found at '{model_path}'. Please run train.py first.")
        return
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    clf = model_data['classifier']
    dino_model_id = model_data['dino_model_id']
    fallback_threshold = model_data['fallback_threshold']
    
    # 2. Initialize Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DinoV2 model '{dino_model_id}' on {device}...")
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_id)
    dino_model = Dinov2Model.from_pretrained(dino_model_id).to(device)
    
    print("Loading EasyOCR reader (English + Japanese) on CPU/GPU...")
    reader = easyocr.Reader(['en', 'ja'], gpu=torch.cuda.is_available())
    
    # 3. Process Optional Reference Frames (For Tier 2 Unsupervised Clustering)
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    ref_dir = os.path.join(os.path.dirname(__file__), "reference_frames")
    ref_embeddings_norm = None
    ref_labels = []
    
    if os.path.exists(ref_dir):
        print(f"Found optional 'reference_frames' directory at '{ref_dir}'. Indexing reference frames...")
        ref_images = []
        for label_name, label_val in [("clean", 0), ("artifact", 1)]:
            sub_dir = os.path.join(ref_dir, label_name)
            if os.path.exists(sub_dir):
                for f in os.listdir(sub_dir):
                    if os.path.splitext(f)[1].lower() in supported_extensions:
                        ref_images.append((os.path.join(sub_dir, f), label_val))
                        
        if len(ref_images) > 0:
            ref_embeddings = []
            for path, label_val in ref_images:
                try:
                    img = Image.open(path).convert('RGB')
                    inputs = dino_processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
                    with torch.no_grad():
                        img_emb = get_dinov2_image_embeds(dino_model, inputs)
                        ref_embeddings.append(img_emb[0].cpu().numpy())
                        ref_labels.append(label_val)
                except Exception as e:
                    print(f"  Warning: Failed to load reference image '{path}': {e}")
            if len(ref_embeddings) > 0:
                X_ref = np.stack(ref_embeddings)
                ref_embeddings_norm = X_ref / np.linalg.norm(X_ref, axis=1, keepdims=True)
                print(f"Cached {len(ref_embeddings_norm)} reference frames for Tier 2 Scene Contrast.")
    else:
        print("[Info] No local 'reference_frames' directory found. Bypassing Tier 2 Scene Contrast.")
        
    # 4. Process Input Folder Images
    image_files = sorted([
        f for f in os.listdir(input_folder) 
        if os.path.splitext(f)[1].lower() in supported_extensions
    ])
    
    print(f"Found {len(image_files)} images in '{input_folder}' to process.")
    results = []
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}...")
        
        try:
            # Step A: Run EasyOCR (Primary Text Artifact Check)
            ocr_res = reader.readtext(img_path)
            total_blocks = len(ocr_res)
            
            is_text_artifact = False
            ocr_score = 0.0
            
            if total_blocks > 5:
                low_conf_count = len([1 for _, _, c in ocr_res if c < 0.5])
                ocr_score = low_conf_count / total_blocks
                if ocr_score >= 0.60:
                    is_text_artifact = True
            
            if is_text_artifact:
                final_score = float(max(0.5, ocr_score))
                flag = "artifact"
                print(f"  -> Text Artifact detected (ocr_score={ocr_score:.4f})")
            else:
                # Extract DinoV2 Feature Embedding
                img = Image.open(img_path).convert('RGB')
                inputs = dino_processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
                with torch.no_grad():
                    img_emb = get_dinov2_image_embeds(dino_model, inputs)
                    emb_numpy = img_emb[0].cpu().numpy()
                
                # L2-normalize
                emb_norm = emb_numpy / np.linalg.norm(emb_numpy)
                
                final_score = 0.0
                flag = "clean"
                nn_sim = 0.0
                
                # Step B: Tier 2 Scene Contrast (Unsupervised Visual Clustering)
                if ref_embeddings_norm is not None:
                    # Calculate cosine similarities to reference set
                    sims = np.dot(ref_embeddings_norm, emb_norm)
                    nn_idx = np.argmax(sims)
                    nn_sim = sims[nn_idx]
                    
                    if nn_sim >= 0.75:
                        # Group all references with similarity >= 0.75 to the test image
                        scene_indices = [i for i, sim in enumerate(sims) if sim >= 0.75]
                        
                        art_similarities = [sims[i] for i in scene_indices if ref_labels[i] == 1]
                        cln_similarities = [sims[i] for i in scene_indices if ref_labels[i] == 0]
                        
                        if len(art_similarities) > 0 and len(cln_similarities) > 0:
                            max_art_sim = max(art_similarities)
                            max_cln_sim = max(cln_similarities)
                            
                            sim_diff = max_art_sim - max_cln_sim
                            if sim_diff > 0:
                                flag = "artifact"
                                final_score = float(min(1.0, 0.5 + sim_diff * 2.0))
                                print(f"  -> Visual-contrast Artifact detected (diff={sim_diff:.4f})")
                            else:
                                flag = "clean"
                                final_score = float(max(0.0, 0.5 + sim_diff * 2.0))
                                if final_score >= 0.5:
                                    final_score = 0.49
                                print(f"  -> Visual-contrast Clean detected (diff={sim_diff:.4f})")
                            # Set nn_sim to 1.0 to bypass fallback
                            nn_sim = 1.0
                            
                # Step C: Tier 3 Fallback Classifier Check
                if nn_sim < 0.75:
                    clf_prob = clf.predict_proba([emb_norm])[0, 1]
                    if clf_prob >= fallback_threshold:
                        flag = "artifact"
                        # Scale [fallback_threshold, 1.00] to [0.50, 1.00]
                        final_score = float(0.5 + (clf_prob - fallback_threshold) * (0.5 / (1.0 - fallback_threshold)))
                        print(f"  -> Classifier Fallback Artifact detected (prob={clf_prob:.4f}, scaled={final_score:.4f})")
                    else:
                        flag = "clean"
                        # Scale [0.00, fallback_threshold) to [0.00, 0.50)
                        final_score = float(clf_prob * (0.5 / fallback_threshold))
                        print(f"  -> Classifier Fallback Clean detected (prob={clf_prob:.4f}, scaled={final_score:.4f})")
            
            # Ensure strict consistency between score and flag
            if flag == "artifact" and final_score < 0.5:
                final_score = 0.5
            elif flag == "clean" and final_score >= 0.5:
                final_score = 0.49
                
            results.append({
                "path": img_path,
                "score": round(final_score, 4),
                "flag": flag
            })
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            results.append({
                "path": img_path,
                "score": 0.5,
                "flag": "artifact"
            })
            
    # Write to output JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDetection complete! Results written to '{output_json}'.")

if __name__ == '__main__':
    main()
