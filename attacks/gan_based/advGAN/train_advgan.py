# attacks/gan_based/advGAN/train_advgan.py
import argparse, yaml, torch, os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from .generator_2d import AdvGANGenerator
from .discriminator_2d import AdvGANDiscriminator
from .advgan import AdvGANWrapper, advgan_losses
from tqdm import tqdm

def load_state_if_present(m, ckpt_path, device):
    if ckpt_path and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd)
        print(f"[INFO] Loaded checkpoint {ckpt_path}")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device if torch.cuda.is_available() or "cpu" in str(device) else "cpu")
    print("[INFO] Using device:", device)

    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])  # map [0,1] -> [-1,1]
    ])

    ds = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)
    dl = DataLoader(ds, batch_size=cfg["train"]["bs"], shuffle=True, num_workers=cfg["train"].get("num_workers", 0))

    # Load target model (victim)
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    num_classes = len(ds.classes)
    # You need to import your model factory — adapt to your project
    from models.ecg_classifier_cnn import make_model
    target = make_model(model_name, num_classes, meta_dim=0)
    target_ckpt = model_cfg.get("ckpt")
    if target_ckpt is None:
        raise ValueError("Please provide model.ckpt path in config")
    target = load_state_if_present(target, target_ckpt, device)
    target = target.to(device).eval()
    for p in target.parameters():
        p.requires_grad = False

    # Create G and D
    G = AdvGANGenerator(in_ch=1).to(device)
    D = AdvGANDiscriminator(in_ch=1).to(device)

    # Wrap utility
    wrap = AdvGANWrapper(target, G, D, eps=cfg["attack"]["advgan"]["eps"])

    # Optimizers
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["train"]["g_lr"])
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["train"]["d_lr"])

    # Hyperparams
    g_steps = cfg["train"].get("g_steps", 1)
    d_steps = cfg["train"].get("d_steps", 1)
    epochs = cfg["train"]["epochs"]

    adv_cfg = cfg["attack"]["advgan"]
    lambda_adv = adv_cfg.get("lambda_adv", 1.0)
    lambda_gan = adv_cfg.get("lambda_gan", 0.1)
    lambda_pert = adv_cfg.get("lambda_pert", 0.05)
    attack_loss_type = adv_cfg.get("attack_loss_type", "cw")
    kappa = adv_cfg.get("kappa", 0.0)
    targeted = adv_cfg.get("targeted", False)
    target_class = adv_cfg.get("target_class", None)
    if targeted and target_class is None:
        raise ValueError("For targeted attack set attack.advgan.target_class")

    # Ensure output dir
    out_dir = Path(cfg["train"].get("out_dir", cfg.get("advgan", {}).get("out_dir", "outputs/advgan")))
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        G.train(); D.train()
        g_losses, d_losses = [], []
        for b in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
            x = b["image"].to(device)
            y = b["label"].to(device)

            # targeted label if needed
            y_target = torch.full_like(y, fill_value=target_class) if targeted else None

            # --- Update Discriminator ---
            for _ in range(d_steps):
                x_adv, _ = wrap.perturb(x)
                g_loss, d_loss = advgan_losses(
                    D, target, x, y, x_adv,
                    lambda_adv=lambda_adv,
                    lambda_gan=lambda_gan,
                    lambda_pert=lambda_pert,
                    targeted=targeted,
                    y_target=y_target,
                    attack_loss_type=attack_loss_type,
                    kappa=kappa,
                    device=device
                )
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()
                d_losses.append(d_loss.item())

            # --- Update Generator ---
            for _ in range(g_steps):
                x_adv, _ = wrap.perturb(x)
                g_loss, _ = advgan_losses(
                    D, target, x, y, x_adv,
                    lambda_adv=lambda_adv,
                    lambda_gan=lambda_gan,
                    lambda_pert=lambda_pert,
                    targeted=targeted,
                    y_target=y_target,
                    attack_loss_type=attack_loss_type,
                    kappa=kappa,
                    device=device
                )
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_losses.append(g_loss.item())

        # Logging per epoch
        print(f"[Epoch {epoch+1}] g_loss={sum(g_losses)/max(1,len(g_losses)):.4f} | d_loss={sum(d_losses)/max(1,len(d_losses)):.4f}")

        # Monitor perturbations on last batch
        with torch.no_grad():
            x_adv, delta = wrap.perturb(x)
            print("  mean |δ| =", delta.abs().mean().item(), "max |δ| =", delta.abs().max().item())

        # Save models each epoch (or best-criteria)
        torch.save(G.state_dict(), out_dir / f"generator_epoch{epoch+1}.pt")
        torch.save(D.state_dict(), out_dir / f"discriminator_epoch{epoch+1}.pt")

    # Final save (also a plain 'generator.pt' and 'discriminator.pt')
    torch.save(G.state_dict(), out_dir / "generator.pt")
    torch.save(D.state_dict(), out_dir / "discriminator.pt")
    print("[INFO] Finished training and saved generator/discriminator to", out_dir)

if __name__ == "__main__":
    main()