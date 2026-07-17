import os
import pickle
import numpy as np
from sklearn.svm import SVC

def main():
    print("--- Phase 1: Train DinoV2 RBF SVM Classifier on Balanced Dataset ---")
    # Load training features
    data = np.load("data/train_features.npz")
    X = data['features']
    y = data['labels']
    
    # L2-normalize training features for SVM stability
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Train RBF SVM
    # Using probability=True to enable predict_proba output
    clf = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
    clf.fit(X_norm, y)
    print("Classifier trained successfully.")
    
    # Save the packaged model assets (completely excluding sample pack data)
    os.makedirs("model_assets", exist_ok=True)
    model_data = {
        'classifier': clf,
        'dino_model_id': "facebook/dinov2-base",
        'fallback_threshold': 0.60
    }
    
    model_path = "model_assets/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nModel assets successfully packaged and saved to {model_path}!")

if __name__ == '__main__':
    main()
