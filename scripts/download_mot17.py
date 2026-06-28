r"""
download_mot17.py — MOT17 Dataset Downloader
============================================
Downloads and extracts the MOT17 tracking dataset (~5.5GB).
Saves to D:\YOLO next\datasets\MOT17 to preserve Drive C space.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

# Configuration
DATASET_URL = "https://motchallenge.net/data/MOT17.zip"
TARGET_DIR = Path("D:/YOLO next/datasets")
ZIP_PATH = TARGET_DIR / "MOT17.zip"
EXTRACT_DIR = TARGET_DIR / "MOT17"

def download_dataset():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    if EXTRACT_DIR.exists() and (EXTRACT_DIR / "train").exists():
        print(f"✅ MOT17 already exists at {EXTRACT_DIR}")
        return

    if not ZIP_PATH.exists():
        print(f"📥 Downloading MOT17 from {DATASET_URL}...")
        print(f"💾 Target path: {ZIP_PATH} (Drive D)")
        
        try:
            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
            ) as progress:
                
                with urllib.request.urlopen(DATASET_URL) as response:
                    total_size = int(response.headers.get("Content-Length", 0))
                    task_id = progress.add_task("download", filename="MOT17.zip", total=total_size)
                    
                    with open(ZIP_PATH, "wb") as file:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            file.write(chunk)
                            progress.update(task_id, advance=len(chunk))
        except Exception as e:
            print(f"❌ Failed to download MOT17: {e}")
            print("Note: motchallenge.net sometimes requires manual download from browser.")
            print("If this fails, download manually from https://motchallenge.net/data/MOT17.zip")
            print(f"and extract it into {TARGET_DIR}")
            return
            
    print(f"\n📦 Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Get list of files for progress tracking
            file_list = zip_ref.namelist()
            
            with Progress() as progress:
                task = progress.add_task("[green]Extracting...", total=len(file_list))
                for file in file_list:
                    zip_ref.extract(file, TARGET_DIR)
                    progress.update(task, advance=1)
                    
        print(f"✅ Extraction complete!")
        # Clean up the 5.5GB zip file to save space
        print(f"🗑️ Deleting zip file to free up space...")
        os.remove(ZIP_PATH)
        print("🎉 MOT17 Dataset is ready!")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    download_dataset()
