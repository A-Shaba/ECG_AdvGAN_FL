# data/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import numpy as np

class ECGImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, meta_cols=None, to_rgb=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.to_rgb = to_rgb

        # If meta_cols not specified, take all columns except filepath and label
        self.meta_cols = meta_cols or [c for c in self.df.columns if c not in ("filepath", "label")]

        # If there are meta columns, coerce them to numeric and fill NaNs
        if self.meta_cols:
            for col in self.meta_cols:
                # If object/categorical, factorize; else try converts to numeric
                if self.df[col].dtype == 'object':
                    self.df[col] = pd.factorize(self.df[col])[0]
                else:
                    # coerce numeric columns and fill NA with 0
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # build class index
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.filepath).convert("L")  # grayscale ECG images
        if self.to_rgb:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = torch.tensor(self.class_to_idx[row.label]).long()

        meta = None
        if self.meta_cols:
            meta_vals = [row[c] for c in self.meta_cols]
            # ensure floats and convert to tensor
            meta_arr = np.array(meta_vals, dtype=np.float32)
            meta = torch.tensor(meta_arr, dtype=torch.float32)
        if meta is not None:
            return {"image": img, "label": y, "meta": meta}
        else:
            return {"image": img, "label": y}