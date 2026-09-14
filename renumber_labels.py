from pathlib import Path

folders = [
    Path("dataset/final/labels/train"),
    Path("dataset/final/labels/val"),
    Path("dataset/final/labels/test")
]

mapping = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}

for folder in folders:
    for label_file in folder.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) < 5:
                continue

            old_class = int(parts[0])

            if old_class in mapping:
                parts[0] = str(mapping[old_class])
                new_lines.append(" ".join(parts))

        label_file.write_text("\n".join(new_lines) + "\n")

print()
print("LABEL RENUMBERING COMPLETE")
print("0 = tire")
print("1 = submarine_pipeline")
print("2 = shipwreck")
print("3 = ghost_net")
print("4 = mine_cylinder")
