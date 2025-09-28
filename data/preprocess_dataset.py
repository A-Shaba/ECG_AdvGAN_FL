import argparse
import os
import math
import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

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
    if not rows:
        # write header only
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filepath","label"])
            w.writeheader()
        return
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _make_synthetic(root, classes=("normal","sveb","veb"), n_per_cls=200, w=256, h=64):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (root/c).mkdir(exist_ok=True)
        for i in range(n_per_cls):
            img = Image.new("L", (w,h), 255)
            d = ImageDraw.Draw(img)
            for x in range(w):
                freq = {"normal":3, "sveb":5, "veb":2}.get(c, 3)
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
    ap.add_argument("--balance", action="store_true",
                    help="Limit each class to the same number of samples (min class count) before splitting")
    ap.add_argument("--per-class", type=int, default=None,
                    help="If set, limit each class to this many samples (overrides --balance)")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.input)
    if args.make_synthetic:
        _make_synthetic(raw)

    # Load items and optional metadata
    if args.csv:
        rows_in = [r for r in csv.DictReader(open(args.csv))]
        items = [(r["filepath"], r["label"], {k:v for k,v in r.items() if k not in ("filepath","label")}) for r in rows_in]
        meta_cols = [c for c in rows_in[0].keys() if c not in ("filepath","label")]
    else:
        items = [(p, c, {}) for p,c in _find_images(raw)]
        meta_cols = []

    # Build class -> list of items (each item is (filepath,label,metadict))
    class_map = {}
    for fp, lbl, meta in items:
        class_map.setdefault(lbl, []).append((fp, lbl, meta))

    classes = sorted(class_map.keys())
    class_counts = {c: len(class_map[c]) for c in classes}
    print("Found classes and counts:", class_counts)

    # Determine per-class limit
    min_count = min(class_counts.values()) if class_counts else 0
    per_class_limit = None
    if args.per_class is not None:
        per_class_limit = args.per_class
    elif args.balance:
        per_class_limit = min_count

    # Seed randomness
    rnd = random.Random(args.random_state)

    # Prepare rows for each split
    train_rows = []
    val_rows = []
    test_rows = []

    for cls in classes:
        items_cls = class_map[cls].copy()
        rnd.shuffle(items_cls)

        # Apply per-class limit if requested
        if per_class_limit is not None:
            items_cls = items_cls[:per_class_limit]

        n_total = len(items_cls)
        if n_total == 0:
            continue

        # Compute counts for test and val, using proportions but ensuring integers
        # We follow the convention: test, then from remaining split validation.
        n_test = int(round(args.test_size * n_total))
        remaining = n_total - n_test
        # val_size is interpreted as fraction of the original remaining (consistent with prior code)
        # Convert val_size relative to remaining fraction: val_ratio = val_size / (1 - test_size)
        if (1.0 - args.test_size) <= 0:
            n_val = 0
        else:
            val_ratio = args.val_size / (1.0 - args.test_size)
            n_val = int(round(val_ratio * remaining))
        n_train = n_total - n_test - n_val

        # Edge-case handling to ensure sum equals n_total
        if n_train < 0:
            n_train = max(0, n_total - n_test)
            n_val = n_total - n_test - n_train

        # Slice
        start = 0
        train_slice = items_cls[start:start+n_train]; start += n_train
        val_slice = items_cls[start:start+n_val]; start += n_val
        test_slice = items_cls[start:start+n_test]; start += n_test

        assert len(train_slice) + len(val_slice) + len(test_slice) == n_total

        # Convert to row dicts (include meta columns if present)
        def make_rows(slist):
            rows = []
            for fp, lbl, meta in slist:
                row = {"filepath": fp, "label": lbl}
                if meta_cols:
                    for c in meta_cols:
                        row[c] = meta.get(c, "")
                rows.append(row)
            return rows

        train_rows.extend(make_rows(train_slice))
        val_rows.extend(make_rows(val_slice))
        test_rows.extend(make_rows(test_slice))

        print(f"Class {cls}: total={n_total}, train={len(train_slice)}, val={len(val_slice)}, test={len(test_slice)}")

    # Shuffle final combined lists so classes are mixed
    rnd.shuffle(train_rows)
    rnd.shuffle(val_rows)
    rnd.shuffle(test_rows)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(train_rows, out/"train.csv")
    _write_manifest(val_rows, out/"val.csv")
    _write_manifest(test_rows, out/"test.csv")

    print("Wrote manifests to", out)
    print(f"Train/Val/Test sizes: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

if __name__ == "__main__":
    main()