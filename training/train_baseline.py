# training/train_baseline.py
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import make_model
from utils.seed import seed_everything
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_everything(cfg.get("seed", 42))

    device = cfg["train"]["device"]

    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Carico dataset
    ds_tr = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)
    ds_va = ECGImageDataset(cfg["data"]["val_csv"], transform=tr)

    num_classes = len(ds_tr.classes)
    meta_dim = len(ds_tr.meta_cols) if ds_tr.meta_cols else 0

    # Creo modello
    model = make_model(cfg["model"]["name"], num_classes, meta_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["bs"], shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["bs"], shuffle=False, num_workers=2)

    best = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for b in tqdm(dl_tr, desc=f"epoch {epoch}"):
            x = b["image"].to(device)
            y = b["label"].to(device)
            if meta_dim > 0:
                meta = b["meta"].to(device)
                logits = model(x, meta)
            else:
                logits = model(x)
            opt.zero_grad()
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        sched.step()

        # Valutazione
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b in dl_va:
                x = b["image"].to(device)
                y = b["label"].to(device)
                if meta_dim > 0:
                    meta = b["meta"].to(device)
                    pred = model(x, meta).argmax(1)
                else:
                    pred = model(x).argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = correct / total
        if acc > best:
            best = acc
            Path(cfg["train"]["out_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(cfg["train"]["out_dir"]) / "baseline_best.pt")
        print(f"Epoch {epoch}: val_acc = {acc:.4f}")

if __name__ == "__main__":
    main()
