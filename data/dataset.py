# data/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch, json

class ECGImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, meta_cols=None, to_rgb=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.meta_cols = meta_cols or [c for c in self.df.columns if c not in ("filepath","label")]
        self.to_rgb = to_rgb

        # build class index
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.filepath).convert("L")  # ECG images are usually single-channel
        if self.to_rgb:  # for ResNet pretrained
            img = img.convert("RGB")
        if self.transform: img = self.transform(img)
        y = torch.tensor(self.class_to_idx[row.label]).long()
        meta = None
        if self.meta_cols:
            meta = torch.tensor([row[c] for c in self.meta_cols], dtype=torch.float32)
        return {"image": img, "label": y, "meta": meta}
