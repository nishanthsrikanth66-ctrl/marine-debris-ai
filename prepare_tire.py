from pathlib import Path
import shutil

SOURCE_IMAGES = Path("dataset/fls-debris/images/train")
SOURCE_LABELS = Path("dataset/fls-debris/labels/train")

DEST_IMAGES = Path("dataset/final/images/train")
DEST_LABELS = Path("dataset/final/labels/train")

DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_LABELS.mkdir(parents=True, exist_ok=True)

count = 0

for label_file in SOURCE_LABELS.glob("*.txt"):

    lines = label_file.read_text().splitlines()
    tire_lines = []

    for line in lines:
        parts = line.split()

        if len(parts) >= 5 and parts[0] == "6":
            parts[0] = "0"
            tire_lines.append(" ".join(parts))

    if not tire_lines:
        continue

    image_found = False

    for ext in [".jpg", ".jpeg", ".png"]:
        image_file = SOURCE_IMAGES / (label_file.stem + ext)

        if image_file.exists():
            shutil.copy2(
                image_file,
                DEST_IMAGES / image_file.name
            )
            image_found = True
            break

    if image_found:
        output_label = DEST_LABELS / label_file.name
        output_label.write_text("\n".join(tire_lines) + "\n")
        count += 1

print()
print("TIRE DATA PREPARED")
print("Images copied:", count)
print("Class mapping: 6 -> 0")
print("Final class 0 = tire")