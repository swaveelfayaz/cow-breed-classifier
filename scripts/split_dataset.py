import os
import shutil
from pathlib import Path
import random

random.seed(42)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

for split in ["train", "val", "test"]:
    split_dir = PROCESSED_DIR / split
    split_dir.mkdir(exist_ok=True)
    for breed_dir in RAW_DIR.iterdir():
        (split_dir / breed_dir.name).mkdir(exist_ok=True)

for breed_dir in RAW_DIR.iterdir():
    breed_name = breed_dir.name
    images = list(breed_dir.glob("*.[jJ][pP][gG]")) + \
             list(breed_dir.glob("*.[jJ][pP][eE][gG]")) + \
             list(breed_dir.glob("*.[pP][nN][gG]"))
    
    random.shuffle(images)
    n = len(images)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    for i, img in enumerate(images):
        if i < n_train:
            dest = PROCESSED_DIR / "train" / breed_name / img.name
        elif i < n_train + n_val:
            dest = PROCESSED_DIR / "val" / breed_name / img.name
        else:
            dest = PROCESSED_DIR / "test" / breed_name / img.name
        shutil.copy2(img, dest)
    
    print(f"{breed_name}: {n} total → {n_train} train, {n_val} val, {n-n_train-n_val} test")

print("✅ Dataset split complete!")
