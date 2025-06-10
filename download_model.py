import gdown
import os

file_id = '1kTXyF9O4gCGHIVsM9HfRr7X3eopc4zj4'
output = 'best.pt'

if not os.path.exists(output):
    print("Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id=1kTXyF9O4gCGHIVsM9HfRr7X3eopc4zj4', output, quiet=False)
else:
    print("Model already exists.")
