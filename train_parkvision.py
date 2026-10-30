# ParkVision AI - Model training (YOLOv8 slot-level detection on PKLot)
#
# Dataset: "PKLot - v2 640" (Roboflow export, COCO annotations, 12,416 images,
# folders: train/ and test/, each with an _annotations.coco.json file).
#
# Run in Google Colab (Runtime > Change runtime type > T4 GPU) or locally.
# You can paste each "# %%" block into its own notebook cell and save it as .ipynb.

# %% 1. Install dependencies (Colab)
# !pip install -q ultralytics pyyaml

# %% 2. Configuration
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

# Folder that contains the dataset's "train" and "test" folders.
# Colab tip: upload the zip to Google Drive, then:
#   from google.colab import drive; drive.mount('/content/drive')
#   !unzip -q "/content/drive/MyDrive/pklot.zip" -d /content/pklot
DATASET_DIR = Path("pklot")

SEED = 42
SUBSET_SIZE = 600          # images used (brief needs >= 100 images per class)
EPOCHS = 25                # brief suggests 15-30
BATCH = 16
IMGSZ = 640                # dataset images are already 640x640
OUT_DIR = Path("parkvision_data")
MODEL_OUT = Path("model")

random.seed(SEED)

# Final YOLO class ids
NAMES = ["space-empty", "space-occupied"]   # 0 = empty, 1 = occupied

# %% 3. Read the COCO annotations and convert them to YOLO format (in memory)
ann_files = []
for p in DATASET_DIR.rglob("*.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        ann_files.append(p)

assert ann_files, "No COCO annotation json found - check DATASET_DIR."
print("Annotation files:", [str(a) for a in ann_files])


def load_coco(json_path: Path):
    """Return list of (image_path, [(cls, xc, yc, w, h), ...]) with normalised YOLO boxes."""
    data = json.loads(json_path.read_text())
    cats = {c["id"]: c["name"].lower() for c in data["categories"]}
    print(f"{json_path.parent.name}: categories = {cats}")
    images = {im["id"]: im for im in data["images"]}
    boxes = defaultdict(list)

    for a in data["annotations"]:
        name = cats.get(a["category_id"], "")
        if "empty" in name:
            cls = 0
        elif "occupied" in name:
            cls = 1
        else:
            continue  # ignore the generic parent category (e.g. "spaces")
        im = images[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]                       # COCO: top-left x, y, width, height (pixels)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(W), x + w), min(float(H), y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes[a["image_id"]].append(
            (cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        )

    out = []
    for img_id, im in images.items():
        path = json_path.parent / im["file_name"]
        if path.exists() and boxes[img_id]:
            out.append((path, boxes[img_id]))
    return out


all_items = []
for jf in ann_files:
    all_items += load_coco(jf)
print("Usable labelled images:", len(all_items))

# %% 4. Select a subset and check class balance
random.shuffle(all_items)
subset = all_items[:SUBSET_SIZE]

img_counts = {0: 0, 1: 0}   # images that contain each class
box_counts = {0: 0, 1: 0}   # total slots of each class
for _, bxs in subset:
    present = set()
    for b in bxs:
        box_counts[b[0]] += 1
        present.add(b[0])
    for c in present:
        img_counts[c] += 1

for c in (0, 1):
    print(f"{NAMES[c]}: in {img_counts[c]} images, {box_counts[c]} slots")
    assert img_counts[c] >= 100, f"Need >= 100 images with '{NAMES[c]}' - increase SUBSET_SIZE"

# %% 5. Split 70 / 15 / 15 and write the YOLO folder structure
n = len(subset)
splits = {
    "train": subset[: int(0.70 * n)],
    "val": subset[int(0.70 * n): int(0.85 * n)],
    "test": subset[int(0.85 * n):],
}

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split, items in splits.items():
    (OUT_DIR / "images" / split).mkdir(parents=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True)
    for img_path, bxs in items:
        stem = f"{img_path.parent.name}_{img_path.stem}"      # avoids name clashes
        shutil.copy(img_path, OUT_DIR / "images" / split / f"{stem}{img_path.suffix}")
        with open(OUT_DIR / "labels" / split / f"{stem}.txt", "w") as f:
            for cls, xc, yc, w, h in bxs:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    print(f"{split}: {len(items)} images")

with open(OUT_DIR / "data.yaml", "w") as f:
    yaml.safe_dump(
        {
            "path": str(OUT_DIR.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: nm for i, nm in enumerate(NAMES)},
        },
        f,
    )

# %% 6. Train YOLOv8 (augmentation is applied on the fly)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small + fast; use yolov8s.pt for higher accuracy
model.train(
    data=str(OUT_DIR / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=SEED,
    project="runs",
    name="parkvision",
    exist_ok=True,
    # augmentation: rotation, flips, brightness
    degrees=10,
    fliplr=0.5,
    flipud=0.0,
    hsv_v=0.4,      # brightness
    hsv_s=0.5,
    hsv_h=0.015,
)

# %% 7. Evaluate on the held-out TEST split (precision, recall, mAP, confusion matrix)
best = Path("runs/parkvision/weights/best.pt")
model = YOLO(str(best))
metrics = model.val(data=str(OUT_DIR / "data.yaml"), split="test", imgsz=IMGSZ,
                    plots=True, project="runs", name="parkvision_test", exist_ok=True)

print(f"Precision : {metrics.box.mp:.3f}")
print(f"Recall    : {metrics.box.mr:.3f}")
print(f"mAP@0.5   : {metrics.box.map50:.3f}")
print(f"mAP@0.5:95: {metrics.box.map:.3f}")
print("Confusion matrix image: runs/parkvision_test/confusion_matrix.png")
print("Training curves       : runs/parkvision/results.png")

# %% 8. Save the model for the Streamlit app
MODEL_OUT.mkdir(exist_ok=True)
shutil.copy(best, MODEL_OUT / "best.pt")
print("Saved ->", MODEL_OUT / "best.pt")
# Commit model/best.pt to your GitHub repo (only ~6 MB for yolov8n).

# %% 9. Pick clean TEST images for sample_images/ (used by the app's "sample photo" buttons)
# PKLot labels are straight boxes around tilted slots, so some images have many overlapping boxes.
# We choose images with many labelled slots, both classes present and little overlap, from different
# days. We also keep ONE messy image on purpose, to show the known limit of the data.
import re

SAMPLE_OUT = Path("sample_images")
N_CLEAN = 5           # clean samples (+1 hard sample = 6 buttons in the app)
MIN_SLOTS = 15        # a useful demo image has at least this many labelled slots
MAX_OVERLAP = 0.10    # share of boxes allowed to overlap another box (IoU > 0.3)


def read_label_boxes(label_path):
    boxes = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            c, xc, yc, w, h = int(p[0]), *map(float, p[1:])
            boxes.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def overlap_share(boxes, thr=0.3):
    """Share of boxes that overlap at least one other box with IoU above thr."""
    bad = 0
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
            inter = ix * iy
            union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
            if union > 0 and inter / union > thr:
                bad += 1
                break
    return bad / len(boxes) if boxes else 1.0


stats = []
for p in sorted((OUT_DIR / "images" / "test").glob("*")):
    lab = OUT_DIR / "labels" / "test" / f"{p.stem}.txt"
    if not lab.exists():
        continue
    bxs = read_label_boxes(lab)
    n_empty = sum(1 for b in bxs if b[0] == 0)
    n_occ = len(bxs) - n_empty
    day = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    stats.append(dict(path=p, n=len(bxs), empty=n_empty, occ=n_occ,
                      overlap=overlap_share(bxs), day=day.group(0) if day else p.stem))


def pick(cands, k, used_days):
    """Take the cleanest images first, from different days where possible."""
    chosen = []
    for c in sorted(cands, key=lambda c: (c["overlap"], -c["n"])):
        if len(chosen) == k:
            break
        if c["day"] not in used_days:
            chosen.append(c)
            used_days.add(c["day"])
    for c in sorted(cands, key=lambda c: (c["overlap"], -c["n"])):   # not enough days: allow repeats
        if len(chosen) == k:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


good = [c for c in stats if c["n"] >= MIN_SLOTS and c["empty"] >= 3 and c["occ"] >= 3]
clean = pick([c for c in good if c["overlap"] <= MAX_OVERLAP], N_CLEAN, set())
if len(clean) < N_CLEAN:                       # relax the overlap limit if the subset is small
    rest = [c for c in good if c not in clean]
    clean += pick(rest, N_CLEAN - len(clean), {c["day"] for c in clean})
hard = sorted((c for c in good if c not in clean), key=lambda c: -c["overlap"])[:1]

if SAMPLE_OUT.exists():
    shutil.rmtree(SAMPLE_OUT)
SAMPLE_OUT.mkdir()
print(f"{'file':60s} {'slots':>5s} {'free':>5s} {'taken':>5s} {'overlap':>8s}  kind")
for kind, group in (("clean", clean), ("hard", hard)):
    for c in group:
        shutil.copy(c["path"], SAMPLE_OUT / c["path"].name)
        print(f"{c['path'].name[:60]:60s} {c['n']:5d} {c['empty']:5d} {c['occ']:5d} {c['overlap']:8.0%}  {kind}")
print("Sample images saved to", SAMPLE_OUT, "- commit this folder to GitHub.")