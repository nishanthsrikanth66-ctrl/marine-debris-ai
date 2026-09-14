from pathlib import Path
import shutil
import random

SOURCE_IMAGES = Path("dataset/final/images/train")
SOURCE_LABELS = Path("dataset/final/labels/train")

VAL_IMAGES = Path("dataset/final/images/val")
VAL_LABELS = Path("dataset/final/labels/val")

TEST_IMAGES = Path("dataset/final/images/test")
TEST_LABELS = Path("dataset/final/labels/test")

VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)
TEST_IMAGES.mkdir(parents=True, exist_ok=True)
TEST_LABELS.mkdir(parents=True, exist_ok=True)

random.seed(42)

tire_files = []

for label_file in SOURCE_LABELS.glob("*.txt"):
    content = label_file.read_text().strip()

    if content.startswith("0 "):
        for ext in [".jpg", ".jpeg", ".png"]:
            image_file = SOURCE_IMAGES / (label_file.stem + ext)

            if image_file.exists():
                tire_files.append((image_file, label_file))
                break

random.shuffle(tire_files)

total = len(tire_files)

val_count = 40
test_count = 40

val_files = tire_files[:val_count]
test_files = tire_files[val_count:val_count + test_count]

for image_file, label_file in val_files:
    shutil.move(image_file, VAL_IMAGES / image_file.name)
    shutil.move(label_file, VAL_LABELS / label_file.name)

for image_file, label_file in test_files:
    shutil.move(image_file, TEST_IMAGES / image_file.name)
    shutil.move(label_file, TEST_LABELS / label_file.name)

print()
print("TIRE SPLIT COMPLETE")
print("Total tire images:", total)
print("Moved to validation:", len(val_files))
print("Moved to test:", len(test_files))
print("Remaining in train:", total - len(val_files) - len(test_files))
