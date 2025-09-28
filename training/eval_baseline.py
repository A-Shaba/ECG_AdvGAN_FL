# eval_baseline.py
import argparse
import yaml
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import make_model
from attacks.gradient_based.fgsm import fgsm_attack
from attacks.gradient_based.pgd import pgd_attack
from attacks.gan_based.advGAN.attack_advgan import AdvGAN_Attack
from utils.metrics import accuracy_with_meta, compute_per_class_asr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config YAML file")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint file")
    ap.add_argument("--attack", default="none", choices=["none", "fgsm", "pgd", "advgan"], help="Attack type")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    cfg = yaml.safe_load(open(cfg_path))
    device_str = cfg.get("eval", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or "cpu" in device_str else "cpu")

    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    ds_te = ECGImageDataset(cfg["data"]["test_csv"], transform=tr)
    dl_te = DataLoader(ds_te, batch_size=cfg.get("eval", {}).get("bs", 64), shuffle=False)

    num_classes = len(ds_te.classes)
    meta_dim = len(ds_te.meta_cols) if ds_te.meta_cols else 0

    model = make_model(cfg["model"]["name"], num_classes, meta_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Setup adversary (should accept meta optional param)
    adversary = None
    if args.attack == "fgsm":
        eps = cfg.get("attack", {}).get("fgsm_eps", 0.03)
        def adversary(m, x, y, meta=None):
            return fgsm_attack(m, x, y, eps=eps, meta=meta)

    elif args.attack == "pgd":
        eps = cfg.get("attack", {}).get("pgd_eps", 0.03)
        alpha = cfg.get("attack", {}).get("pgd_alpha", 0.005)
        steps = cfg.get("attack", {}).get("pgd_steps", 40)
        def adversary(m, x, y, meta=None):
            return pgd_attack(m, x, y, eps=eps, alpha=alpha, steps=steps, meta=meta)

    elif args.attack == "advgan":
        advgan_cfg = cfg.get("attack", {}).get("advgan", {})
        advgan_ckpt = advgan_cfg.get("model_ckpt", None)
        if advgan_ckpt is None:
            raise ValueError("Missing AdvGAN checkpoint in config under attack.advgan.model_ckpt")
        eps = advgan_cfg.get("eps", 0.03)
        advgan_attack = AdvGAN_Attack(model, ckpt_path=advgan_ckpt, eps=eps, device=device)
        # AdvGAN attack might not need meta (check your AdvGAN implementation)
        def adversary(m, x, y, meta=None):
            # Return adversarial examples (ensure device & dtype)
            return advgan_attack(x)

    # Accuracy on clean data
    clean_acc, clean_preds, clean_labels, clean_mask = accuracy_with_meta(
        model, dl_te, device, adversary=None, return_details=True
    )

    robust_acc = None
    per_class_asr = {}

    if args.attack != "none":
        # Only samples that were classified correctly on clean input
        correct_indices = torch.where(clean_mask)[0]
        # Build dataset with only those samples (list of dicts)
        correct_samples = [ds_te[i.item()] for i in correct_indices]

        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, samples): self.samples = samples
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx): return self.samples[idx]

        dl_correct = DataLoader(
            SubsetDataset(correct_samples),
            batch_size=cfg.get("eval", {}).get("bs", 64),
            shuffle=False
        )

        robust_acc, robust_preds, _, _ = accuracy_with_meta(
            model, dl_correct, device, adversary=adversary, return_details=True
        )

        # compute per-class ASR (attack success rate) comparing clean labels (for correct samples)
        per_class_asr = compute_per_class_asr(clean_labels[correct_indices], robust_preds)

    # Save results
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_results.json"

    results = {}
    if eval_path.exists():
        with open(eval_path, "r") as f:
            results = json.load(f)

    results["clean_acc"] = clean_acc
    if args.attack != "none":
        results.setdefault("attacks", {})
        results["attacks"][args.attack] = {
            "robust_acc": robust_acc,
            "per_class_asr": per_class_asr
        }

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Results saved to {eval_path}")
    print(f"=> Clean Accuracy: {clean_acc:.4f}")
    if args.attack != "none":
        print(f"=> Robust Accuracy ({args.attack}): {robust_acc:.4f}")
        print("=> Per-Class ASR:")
        for cls, asr in per_class_asr.items():
            print(f"   Class {cls}: {asr if asr is not None else 'N/A'}")


if __name__ == "__main__":
    main()