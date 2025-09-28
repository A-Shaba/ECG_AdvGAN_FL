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
import collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_everything(cfg.get("seed", 42))

    # Safer device handling
    device_str = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) or "cpu" in device_str else "cpu")

    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load datasets
    ds_tr = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)
    ds_va = ECGImageDataset(cfg["data"]["val_csv"], transform=tr)

    # Try test csv if present in config
    ds_te = None
    if cfg["data"].get("test_csv", None):
        try:
            ds_te = ECGImageDataset(cfg["data"]["test_csv"], transform=tr)
        except Exception:
            ds_te = None

    num_classes = len(ds_tr.classes)
    meta_dim = len(ds_tr.meta_cols) if ds_tr.meta_cols else 0

    # Print class distributions to detect imbalance
    print("\n[INFO] Class distributions:")
    def print_counts(name, ds):
        if ds is None:
            return
        try:
            counts = collections.Counter(ds.df['label'])
            total = len(ds)
            print(f"  {name}: total={total}, counts={counts}")
            maj_label, maj_count = counts.most_common(1)[0]
            print(f"   -> majority: {maj_label} {maj_count}/{total} = {maj_count/total:.4f}")
        except Exception as e:
            print(f"  {name}: could not compute counts: {e}")

    print_counts("train", ds_tr)
    print_counts("val", ds_va)
    print_counts("test", ds_te)

    # Create model
    model = make_model(cfg["model"]["name"], num_classes, meta_dim).to(device)

    # Print number of trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model params: total={total_params}, trainable={trainable_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    print(f"[INFO] Optimizer param groups: {len(opt.param_groups)}, params in first group: {len(opt.param_groups[0]['params'])}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["bs"], shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["bs"], shuffle=False, num_workers=2)

    # Optional smoke test single-batch (toggle via config train.smoke_test = true)
    if cfg.get("train", {}).get("smoke_test", False):
        print("\n[INFO] Running single-batch smoke test to verify parameter updates...")
        model.train()
        batch = next(iter(dl_tr))
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        meta = batch.get("meta", None)
        if meta is not None:
            meta = meta.to(device)

        params_before = [p.clone().detach() for p in model.parameters()]

        opt.zero_grad()
        logits = model(x, meta) if meta is not None else model(x)
        loss = criterion(logits, y)
        loss.backward()
        # compute grad norm quick
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"  smoke loss: {loss.item():.6f}, grad_norm: {grad_norm:.6f}")

        opt.step()

        params_after = [p.clone().detach() for p in model.parameters()]
        changes = [ (a - b).norm().item() for a,b in zip(params_after, params_before) ]
        print("  param change norms (first 10):", changes[:10])
        print("[INFO] Smoke test done.\n")

    best = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        num_batches = 0

        # snapshot first parameter for change detection
        first_param_before = None
        for p in model.parameters():
            first_param_before = p.data.clone()
            break

        for i, b in enumerate(tqdm(dl_tr, desc=f"epoch {epoch}")):
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

            # compute a batch-level grad norm (for debugging)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            opt.step()

            running_loss += loss.item()
            num_batches += 1

            # print diagnostic every N batches
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} batch {i+1}: loss={loss.item():.6f}, grad_norm={total_grad_norm:.6f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        # param change detection
        first_param_after = None
        for p in model.parameters():
            first_param_after = p.data.clone()
            break
        param_change_norm = (first_param_after - first_param_before).norm().item() if (first_param_after is not None and first_param_before is not None) else 0.0

        print(f"\n[INFO] Epoch {epoch} summary: avg_train_loss = {avg_loss:.6f}, first_param_change_norm = {param_change_norm:.6f}")

        sched.step()

        # Evaluation
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
        acc = correct / total if total > 0 else 0.0
        # print with more precision to see small changes
        print(f"Epoch {epoch}: val_acc = {acc:.6f}")

        if acc > best:
            best = acc
            Path(cfg["train"]["out_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(cfg["train"]["out_dir"]) / "baseline_best.pt")

    print("\n[INFO] Training finished. Best val acc: {:.6f}".format(best))

if __name__ == "__main__":
    main()