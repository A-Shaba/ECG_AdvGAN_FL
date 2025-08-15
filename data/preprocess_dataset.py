# data/preprocess_dataset.py
import argparse, os, random, math, csv, json
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedShuffleSplit

def _find_images(root):
    exts = {".png", ".jpg", ".jpeg"}
    items = []
    for cls_dir in sorted(Path(root).glob("*")):
        if cls_dir.is_dir():
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    items.append((str(p), cls_dir.name))
    return items

def _write_manifest(rows, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def _make_synthetic(root, classes=("normal","sveb","veb"), n_per_cls=200, w=256, h=64):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (root/c).mkdir(exist_ok=True)
        for i in range(n_per_cls):
            img = Image.new("L", (w,h), 255)
            d = ImageDraw.Draw(img)
            # draw sinusoid with class-specific frequency/perturbation
            for x in range(w):
                freq = {"normal":3, "sveb":5, "veb":2}[c]
                y = int(h/2 + (h/3)*math.sin(2*math.pi*(x/w)*freq))
                d.point((x,y), fill=0)
            img.save(root/c/f"{c}_{i:04d}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", default=None, help="optional CSV with filepath,label,...")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--make-synthetic", action="store_true")
    args = ap.parse_args()

    raw = Path(args.input)
    if args.make-synthetic:
        _make_synthetic(raw)

    if args.csv:
        rows = [r for r in csv.DictReader(open(args.csv))]
        items = [(r["filepath"], r["label"]) for r in rows]
        # keep metadata columns if any
        meta_cols = [c for c in rows[0].keys() if c not in ("filepath","label")]
    else:
        items = _find_images(raw)
        meta_cols = []

    X = [p for p,_ in items]
    y = [c for _,c in items]

    # stratified split: train/val/test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(sss1.split(X,y))
    X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
    X_test, y_test = [X[i] for i in test_idx], [y[i] for i in test_idx]

    val_ratio = args.val_size/(1-args.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    tr_idx, val_idx = next(sss2.split(X_train,y_train))

    def rows_from(idx_list):
        rows = []
        for i in idx_list:
            row = {"filepath":X[i], "label":y[i]}
            # attach extra metadata if csv was given
            # (simple example: store everything as JSON string)
            rows.append(row)
        return rows

    out = Path(args.output)
    _write_manifest(rows_from(tr_idx), out/"train.csv")
    _write_manifest(rows_from(val_idx), out/"val.csv")
    _write_manifest(rows_from(test_idx), out/"test.csv")
    print("Wrote manifests to", out)

if __name__ == "__main__":
    main()
