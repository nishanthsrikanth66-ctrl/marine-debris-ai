from pathlib import Path
import shutil

SOURCE = Path("dataset/drishti-sss")
DEST = Path("dataset/final")

# DRISHTI → FINAL class mapping
CLASS_MAP = {
    0: 1,  # crab_pot
    1: 2,  # submarine_pipeline
    2: 3,  # shipwreck
    3: 4,  # ghost_net
    4: 5,  # mine_cylinder
}

for split in ["train", "val", "test"]:

    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    dest_images = DEST / "images" / split
    dest_labels = DEST / "labels" / split

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    count = 0

    for label_file in source_labels.glob("*.txt"):

        new_lines = []

        for line in label_file.read_text().splitlines():

            parts = line.split()

            if len(parts) < 5:
                continue

            old_class = int(parts[0])

            if old_class not in CLASS_MAP:
                continue

            parts[0] = str(CLASS_MAP[old_class])
            new_lines.append(" ".join(parts))

        if not new_lines:
            continue

        # Find matching image
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
        output_label.write_text("\n".join(new_lines) + "\n")

        count += 1

    print(f"{split}: {count} images copied")

print()
print("DRISHTI SSS DATA ADDED")
print("0 = tire")
print("1 = crab_pot")
print("2 = submarine_pipeline")
print("3 = shipwreck")
print("4 = ghost_net")
print("5 = mine_cylinder")