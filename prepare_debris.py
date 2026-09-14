from pathlib import Path
import shutil

SOURCE_ROOT = Path("dataset/fls-debris")

SOURCE_IMAGES = SOURCE_ROOT / "images"
SOURCE_LABELS = SOURCE_ROOT / "labels"

DEST_ROOT = Path("dataset/final")

# FLS classes that we will combine into "debris"
DEBRIS_CLASSES = {
    "1",   # can
    "2",   # bottle
    "3",   # drink-carton
    "9",   # shampoo-bottle
    "10"   # standing-bottle
}

# Final class ID
DEBRIS_ID = "5"

total_images = 0
total_objects = 0

for split in ["train", "val", "test"]:

    source_images = SOURCE_IMAGES / split
    source_labels = SOURCE_LABELS / split

    dest_images = DEST_ROOT / "images" / split
    dest_labels = DEST_ROOT / "labels" / split

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for label_file in source_labels.glob("*.txt"):

        lines = label_file.read_text().splitlines()
        debris_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) >= 5 and parts[0] in DEBRIS_CLASSES:
                parts[0] = DEBRIS_ID
                debris_lines.append(" ".join(parts))
                total_objects += 1

        if not debris_lines:
            continue

        image_file = None

        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = source_images / (label_file.stem + ext)

            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            continue

        shutil.copy2(
            image_file,
            dest_images / image_file.name
        )

        output_label = dest_labels / label_file.name
        output_label.write_text(
            "\n".join(debris_lines) + "\n"
        )

        total_images += 1

print()
print("DEBRIS DATA PREPARED")
print("--------------------")
print("Images copied:", total_images)
print("Debris objects:", total_objects)
print("FLS classes combined:")
print("  can")
print("  bottle")
print("  drink-carton")
print("  shampoo-bottle")
print("  standing-bottle")
print()
print("Final class ID: 5 = debris")