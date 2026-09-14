from pathlib import Path
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Drum dataset location
SOURCE_ROOT = Path(
    r"C:\Users\srika\Downloads\Drum\IOES-Lab-Small-Underwater-Objects-3D-Point-Cloud-Dataset-07bcaed\SUOP_dataset\drum"
)

# Output location
OUTPUT_ROOT = Path(
    r"C:\Users\srika\OneDrive\Desktop\marine-debris-ai\dataset\final\images\barrel_source"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def rotation_matrix(axis, theta):
    c, s = np.cos(theta), np.sin(theta)

    if axis == "x":
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    if axis == "y":
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    if axis == "z":
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])


def create_image(point_cloud, output_path):

    points = np.asarray(point_cloud.points)

    if points.size == 0:
        return False

    # Same rotations as the SUOP script
    Rx = rotation_matrix("x", np.deg2rad(-5))
    Ry = rotation_matrix("y", np.deg2rad(-170))
    Rz = rotation_matrix("z", np.deg2rad(180))

    R = Rz @ Ry @ Rx

    points = points @ R.T

    z = points[:, 2]

    z_min = z.min()
    z_max = z.max()

    norm = (z - z_min) / (z_max - z_min + 1e-8)

    clipped = np.clip(
        (norm - 0.3) / 0.8,
        0.0,
        1.0
    )

    colors = plt.get_cmap("jet")(clipped)[:, :3]

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.1,
        depthshade=False
    )

    ax.view_init(
        elev=30,
        azim=45
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return True


count = 0

for xyz_file in SOURCE_ROOT.rglob("point_cloud.xyz"):

    # Example:
    # drum_range_10m/case_001/point_cloud.xyz

    case_folder = xyz_file.parent
    case_name = case_folder.name

    range_name = None

    for part in xyz_file.parts:

        if part == "drum_range_3m":
            range_name = "3m"

        elif part == "drum_range_6m":
            range_name = "6m"

        elif part == "drum_range_10m":
            range_name = "10m"

    if range_name is None:
        continue

    output_dir = OUTPUT_ROOT / range_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{case_name}.png"

    if output_file.exists():
        print(f"[SKIP] {output_file}")
        continue

    print(f"[PROCESS] {range_name} / {case_name}")

    pcd = o3d.io.read_point_cloud(str(xyz_file))

    success = create_image(
        pcd,
        output_file
    )

    if success:
        count += 1
        print(f"[OK] {output_file}")


print()
print("DRUM IMAGE GENERATION COMPLETE")
print("Images generated:", count)