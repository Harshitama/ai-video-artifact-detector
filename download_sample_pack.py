import os
import zipfile
import gdown

def download_and_extract():
    url = 'https://drive.google.com/uc?id=1yxq5cQuaB25Cp44-_k8ROYjzSvMlFclG'
    output = 'sample_pack.zip'
    
    if not os.path.exists(output):
        print("Downloading sample pack from Google Drive...")
        gdown.download(url, output, quiet=False)
    else:
        print("sample_pack.zip already exists.")
        
    extract_dir = 'data/sample_pack'
    if not os.path.exists(extract_dir):
        print(f"Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
    else:
        print(f"Extraction directory {extract_dir} already exists.")

if __name__ == '__main__':
    download_and_extract()
