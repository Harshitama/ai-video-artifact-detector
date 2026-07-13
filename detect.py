import os
import argparse
import json
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Dinov2Model
import easyocr

def get_scene_key(filename):
    # Remove prefix "artifact_XX_" or "clean_XX_"
    name = filename.replace("artifact_", "").replace("clean_", "")
    # Remove leading digits and underscores
    while name and (name[0].isdigit() or name[0] == '_'):
        name = name[1:]
    # Remove suffix starting with common separators
    for suffix in ["_f", "_final", "_demo", "_hero", "_payoff", "_hook", "_hero"]:
        if suffix in name:
            name = name.split(suffix)[0]
    return name

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
        print(f"Error: Model assets not found at '{model_path}'. Please run train_hybrid.py first.")
        return
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    clf = model_data['classifier']
    dino_model_id = model_data['dino_model_id']
    ref_embeddings_norm = model_data['ref_embeddings_norm']
    ref_names = model_data['ref_names']
    ref_labels = model_data['ref_labels']
    
    # Pre-calculate reference scene keys
    ref_scene_keys = [get_scene_key(name) for name in ref_names]
    
    # 2. Initialize Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DinoV2 model '{dino_model_id}' on {device}...")
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_id)
    dino_model = Dinov2Model.from_pretrained(dino_model_id).to(device)
    
    print("Loading EasyOCR reader (English + Japanese) on CPU/GPU...")
    reader = easyocr.Reader(['en', 'ja'], gpu=torch.cuda.is_available())
    
    # 3. Process Images
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
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
                # Map score to >= 0.5 for artifact
                final_score = float(max(0.5, ocr_score))
                flag = "artifact"
                print(f"  -> Text Artifact detected (ocr_score={ocr_score:.4f})")
            else:
                # Step B: Run DinoV2 Feature Extraction
                img = Image.open(img_path).convert('RGB')
                inputs = dino_processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
                with torch.no_grad():
                    img_emb = get_dinov2_image_embeds(dino_model, inputs)
                    emb_numpy = img_emb[0].cpu().numpy()
                
                # L2-normalize
                emb_norm = emb_numpy / np.linalg.norm(emb_numpy)
                
                # Calculate cosine similarities to reference set
                sims = np.dot(ref_embeddings_norm, emb_norm)
                nn_idx = np.argmax(sims)
                nn_sim = sims[nn_idx]
                
                final_score = 0.0
                flag = "clean"
                
                # Scene-specific relative contrastive check (for known scenes)
                if nn_sim >= 0.65:
                    target_scene = ref_scene_keys[nn_idx]
                    # Find all references matching this scene key
                    scene_indices = [i for i, sk in enumerate(ref_scene_keys) if sk == target_scene]
                    
                    art_similarities = [sims[i] for i in scene_indices if ref_labels[i] == 1]
                    cln_similarities = [sims[i] for i in scene_indices if ref_labels[i] == 0]
                    
                    if len(art_similarities) > 0 and len(cln_similarities) > 0:
                        max_art_sim = max(art_similarities)
                        max_cln_sim = max(cln_similarities)
                        
                        # Relative score difference
                        sim_diff = max_art_sim - max_cln_sim
                        if sim_diff > 0:
                            # Closer to the artifact version
                            flag = "artifact"
                            # Map positive diff to a score >= 0.5
                            final_score = float(min(1.0, 0.5 + sim_diff * 2.0))
                            print(f"  -> Scene-contrast Artifact detected (scene={target_scene}, diff={sim_diff:.4f})")
                        else:
                            # Closer to the clean version
                            flag = "clean"
                            final_score = float(max(0.0, 0.5 + sim_diff * 2.0))
                            if final_score >= 0.5:
                                final_score = 0.49 # Keep consistent
                            print(f"  -> Scene-contrast Clean detected (scene={target_scene}, diff={sim_diff:.4f})")
                    else:
                        # Fallback if the scene doesn't have both types of references
                        nn_sim = -1.0 # Force fallback
                        
                # Step C: Fallback Classifier Check (for new/unknown scenes)
                if nn_sim < 0.65:
                    clf_prob = clf.predict_proba([emb_norm])[0, 1]
                    if clf_prob >= 0.60:
                        flag = "artifact"
                        # Scale [0.60, 1.00] to [0.50, 1.00]
                        final_score = float(0.5 + (clf_prob - 0.60) * (0.5 / 0.4))
                        print(f"  -> Classifier Fallback Artifact detected (prob={clf_prob:.4f}, scaled={final_score:.4f})")
                    else:
                        flag = "clean"
                        # Scale [0.00, 0.60) to [0.00, 0.50)
                        final_score = float(clf_prob * (0.5 / 0.60))
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
            # Safe default fallback on error
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
