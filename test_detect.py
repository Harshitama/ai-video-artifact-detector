import os
import json
import subprocess
import pytest
from PIL import Image

def test_cli_execution(tmp_path):
    # 1. Create a temporary folder with a dummy image
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    
    # Create a dummy image
    img_path = test_dir / "dummy.png"
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img.save(img_path)
    
    output_json = tmp_path / "results.json"
    
    import sys
    # 2. Run detect.py CLI using subprocess
    cmd = [
        sys.executable,
        "detect.py",
        "--input", str(test_dir),
        "--output", str(output_json)
    ]
    
    # Run with UTF-8 encoding environment variable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Verify execution was successful
    assert result.returncode == 0, f"detect.py failed with: {result.stderr}"
    
    # 3. Verify output JSON file exists
    assert output_json.exists(), "results.json was not created"
    
    # 4. Verify JSON content structure
    with open(output_json, 'r') as f:
        data = json.load(f)
        
    assert isinstance(data, list), "Output should be a list"
    assert len(data) == 1, "Should contain exactly 1 entry"
    
    entry = data[0]
    assert "path" in entry, "Entry should contain 'path'"
    assert "score" in entry, "Entry should contain 'score'"
    assert "flag" in entry, "Entry should contain 'flag'"
    
    # Check data types and values
    assert isinstance(entry["path"], str)
    assert isinstance(entry["score"], float)
    assert 0.0 <= entry["score"] <= 1.0
    assert entry["flag"] in ["artifact", "clean"]
    
    # Check consistency: score >= 0.5 <-> flag == 'artifact'
    if entry["flag"] == "artifact":
        assert entry["score"] >= 0.5
    else:
        assert entry["score"] < 0.5

def test_missing_input_folder(tmp_path):
    import sys
    # Run CLI with a non-existent input folder
    output_json = tmp_path / "results.json"
    cmd = [
        sys.executable,
        "detect.py",
        "--input", "non_existent_folder_xyz_123",
        "--output", str(output_json)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The script should print error and exit cleanly (or with error)
    assert "Error:" in result.stdout or "Error:" in result.stderr
    assert not output_json.exists(), "results.json should not be created on error"
