import os, re
import time
from datetime import datetime

# Path to the directory containing the model.pt files
TARGET_DIR = "/home/nand-ml/decoder-only-transformer/log_ZAPjKM"

FILE_PATTERN = re.compile(r"model_14\d+\.pt")

def delete_model_files():
    """Delete all 'model.pt' files in the target directory."""
    if not os.path.exists(TARGET_DIR):
        print(f"{datetime.now()}: Directory {TARGET_DIR} does not exist.")
        return
    
    # Find and delete all model.pt files
    files_deleted = 0
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if FILE_PATTERN.match(file):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                files_deleted += 1
                print(f"{datetime.now()}: Deleted {file_path}")
    
    if files_deleted == 0:
        print(f"{datetime.now()}: No model.pt files found.")
    else:
        print(f"{datetime.now()}: Deleted {files_deleted} model.pt file(s).")

if __name__ == "__main__":
    while True:
        delete_model_files()
        # Wait for 15 minutes
        time.sleep(5 * 60)
