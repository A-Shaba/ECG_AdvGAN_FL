import torch
from pathlib import Path
import matplotlib.pyplot as plt
import argparse, yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import SmallECGCNN, resnet18, DeepECGCNN
from attacks.gan_based.advGAN.attack_advgan import AdvGAN_Attack


def make_model(name, num_classes):
    if name == "small_cnn":
        return SmallECGCNN(1, num_classes)
    if name == "resnet18":
        return resnet18(num_classes)
    if name == "deep_cnn":
        return DeepECGCNN(1, num_classes)
    raise ValueError(f"Unknown model {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to baseline yaml config")
    ap.add_argument("--out", default="outputs/advgan_examples", help="Output dir for images")
    ap.add_argument("--max_batches", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))


    # --- dataset ---
    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    ds = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)  # using train_csv since no val_csv in config
    dl = DataLoader(
        ds, batch_size=cfg["train"]["bs"], shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0)
    )

    # --- model ---
    model = make_model(cfg["model"]["name"], num_classes=len(ds.classes))
    model.load_state_dict(torch.load(cfg["model"]["ckpt"], map_location=cfg["train"]["device"]))
    model.to(cfg["train"]["device"])
    model.eval()

    # --- attack ---
    atk_cfg = cfg["attack"]["advgan"]
    atk = AdvGAN_Attack(model, ckpt_path=atk_cfg["model_ckpt"], eps=atk_cfg["eps"])

    # --- save comparisons ---
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            x = batch["image"].to(cfg["train"]["device"])
            y = batch["label"].to(cfg["train"]["device"])
            x_adv = atk(x)

            pred_clean = model(x).argmax(1)
            pred_adv = model(x_adv).argmax(1)

            for j in range(x.size(0)):
                clean_img = x[j].cpu().squeeze().numpy()
                adv_img = x_adv[j].cpu().squeeze().numpy()
                label = int(y[j].item())
                pc = int(pred_clean[j].item())
                pa = int(pred_adv[j].item())

                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(clean_img, cmap="gray")
                ax[0].set_title(f"Clean (pred={pc}, label={label})")
                ax[0].axis("off")

                ax[1].imshow(adv_img, cmap="gray")
                ax[1].set_title(f"Adv (pred={pa})")
                ax[1].axis("off")

                fig.tight_layout()
                fig.savefig(Path(args.out) / f"batch{i}_idx{j}_compare.png")
                plt.close(fig)


if __name__ == "__main__":
    main()
