import os
import json
import subprocess
import pandas as pd
import sys

def main():
    print("=== Running Evaluation Script ===")
    
    sample_pack_dir = os.path.join("data", "sample_pack", "sample_pack")
    labels_csv_path = os.path.join(sample_pack_dir, "labels.csv")
    output_json = "results.json"
    
    if not os.path.exists(sample_pack_dir):
        print(f"Error: Sample pack directory '{sample_pack_dir}' not found.")
        print("Please run download_sample_pack.py first.")
        return
        
    if not os.path.exists(labels_csv_path):
        print(f"Error: Ground-truth file '{labels_csv_path}' not found.")
        return
        
    # Check if reference_frames directory exists, which would cause leakage
    ref_dir = "reference_frames"
    if os.path.exists(ref_dir):
        print(f"Warning: Found local '{ref_dir}' folder. Temporarily renaming it during evaluation to ensure zero data leakage...")
        os.rename(ref_dir, "reference_frames_temp")
        
    try:
        # Run detect.py CLI via subprocess using virtual environment python
        cmd = [
            sys.executable,
            "detect.py",
            "--input", sample_pack_dir,
            "--output", output_json
        ]
        print(f"Executing: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error running detect.py:")
            print(result.stderr)
            return
            
        print("detect.py execution complete.")
        
    finally:
        # Restore reference_frames folder if it was renamed
        if os.path.exists("reference_frames_temp"):
            os.rename("reference_frames_temp", ref_dir)
            print("Restored 'reference_frames' directory.")
            
    # Load predictions
    if not os.path.exists(output_json):
        print(f"Error: Output predictions file '{output_json}' was not generated.")
        return
        
    with open(output_json, 'r') as f:
        predictions = json.load(f)
        
    # Load ground-truth labels
    gt_labels = {}
    with open(labels_csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',', 3)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 3)
            if len(parts) == 4:
                filename = parts[0]
                label_str = parts[1]
                binary_val = 1 if label_str == 'artifact' else 0
                gt_labels[filename] = binary_val
                
    # Align and evaluate
    tp, fp, tn, fn = 0, 0, 0, 0
    misclassified = []
    
    for pred in predictions:
        pred_path = pred['path']
        pred_flag = pred['flag']
        pred_score = pred['score']
        
        filename = os.path.basename(pred_path)
        if filename not in gt_labels:
            print(f"Warning: Filename '{filename}' not found in ground truth. Skipping.")
            continue
            
        gt_val = gt_labels[filename]
        pred_val = 1 if pred_flag == 'artifact' else 0
        
        if gt_val == 1 and pred_val == 1:
            tp += 1
        elif gt_val == 0 and pred_val == 1:
            fp += 1
            misclassified.append((filename, "clean", "artifact", pred_score))
        elif gt_val == 0 and pred_val == 0:
            tn += 1
        elif gt_val == 1 and pred_val == 0:
            fn += 1
            misclassified.append((filename, "artifact", "clean", pred_score))
            
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Target Domain: Client's Sample Pack (Unseen Test Set)")
    print(f"Total Evaluated Frames: {total}")
    print("-" * 52)
    print(f"Accuracy:  {accuracy:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall:    {recall:.4%}")
    print(f"F1-Score:  {f1:.4%}")
    print("-" * 52)
    print("Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}  (Artifacts correctly flagged)")
    print(f"  False Positives (FP): {fp}  (Clean frames wrongly flagged)")
    print(f"  True Negatives (TN):  {tn}  (Clean frames correctly passed)")
    print(f"  False Negatives (FN): {fn}  (Artifacts wrongly passed)")
    print("-" * 52)
    
    if len(misclassified) > 0:
        print("Misclassified Details:")
        for name, gt_lbl, pred_lbl, score in misclassified:
            print(f"  - {name}: True={gt_lbl:<8} Pred={pred_lbl:<8} (score={score:.4f})")
    else:
        print("All frames correctly classified!")
    print("====================================================")

if __name__ == '__main__':
    main()
