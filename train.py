import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.linear_model import LogisticRegression

def get_dinov2_image_embeds(model, image_inputs):
    outputs = model(**image_inputs)
    return outputs.pooler_output

def main():
    print("--- Phase 1: Train DinoV2 Classifier on Balanced Dataset ---")
    data = np.load("data/train_features.npz")
    X = data['features']
    y = data['labels']
    
    # Train Logistic Regression
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    print("Classifier trained successfully.")
    
    print("\n--- Phase 2: Extract and Cache Sample Pack Embeddings ---")
    sample_pack_dir = "data/sample_pack/sample_pack"
    labels_csv_path = os.path.join(sample_pack_dir, "labels.csv")
    
    rows = []
    with open(labels_csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',', 3)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 3)
            if len(parts) == 4:
                rows.append({
                    'filename': parts[0],
                    'label': parts[1],
                    'binary_label': 1 if parts[1] == 'artifact' else 0
                })
    df_labels = pd.DataFrame(rows)
    
    # Initialize DinoV2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model_id = "facebook/dinov2-base"
    print(f"Loading {dino_model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(dino_model_id)
    model = Dinov2Model.from_pretrained(dino_model_id).to(device)
    
    ref_embeddings = []
    ref_names = []
    ref_labels = []
    
    for idx, row in df_labels.iterrows():
        img_path = os.path.join(sample_pack_dir, row['filename'])
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img.resize((224, 224)), return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = get_dinov2_image_embeds(model, inputs)
            emb_numpy = img_emb[0].cpu().numpy()
            
        ref_embeddings.append(emb_numpy)
        ref_names.append(row['filename'])
        ref_labels.append(row['binary_label'])
        
    X_ref = np.stack(ref_embeddings)
    # L2-normalize
    X_ref_norm = X_ref / np.linalg.norm(X_ref, axis=1, keepdims=True)
    
    # Save the packaged model assets
    os.makedirs("model_assets", exist_ok=True)
    model_data = {
        'classifier': clf,
        'threshold': 0.5, # default fallback threshold
        'dino_model_id': dino_model_id,
        'ref_embeddings_norm': X_ref_norm,
        'ref_names': ref_names,
        'ref_labels': ref_labels
    }
    
    model_path = "model_assets/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nModel assets successfully packaged and saved to {model_path}!")

if __name__ == '__main__':
    main()
