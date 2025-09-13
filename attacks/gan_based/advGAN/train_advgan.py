import argparse, yaml, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import SmallECGCNN, resnet18_gray, DeepECGCNN
from .generator_2d import AdvGANGenerator
from .discriminator_2d import AdvGANDiscriminator
from .advgan import AdvGANWrapper, advgan_losses
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # === Dataset & transforms ===
    tr = transforms.Compose([
        transforms.Resize(tuple(cfg["data"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["bs"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0)
    )

    # === Model ===
    if cfg["model"]["name"] == "resnet18":
        target = resnet18_gray(len(ds.classes))
    elif cfg["model"]["name"] == "small_cnn":
        target = SmallECGCNN(1, len(ds.classes))
    elif cfg["model"]["name"] == "deep_cnn":
        target = DeepECGCNN(1, len(ds.classes))
    else:
        raise ValueError(f"Unknown model name: {cfg['model']['name']}")

    target.load_state_dict(torch.load(cfg["model"]["ckpt"], map_location=cfg["train"]["device"]))
    target.eval()

    # === AdvGAN components ===
    G, D = AdvGANGenerator(1), AdvGANDiscriminator(1)
    wrap = AdvGANWrapper(target, G, D, eps=cfg["attack"]["advgan"]["eps"])
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["train"]["g_lr"])
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["train"]["d_lr"])

    g_steps = cfg["train"].get("g_steps", 1)
    d_steps = cfg["train"].get("d_steps", 1)

    # === Attack mode from config ===
    targeted = cfg["attack"]["advgan"].get("targeted", False)
    target_class = cfg["attack"]["advgan"].get("target_class", None)

    if targeted and target_class is None:
        raise ValueError("For targeted attack you must set attack.advgan.target_class in config")

    for epoch in range(cfg["train"]["epochs"]):
        g_losses, d_losses = [], []
        for i, b in enumerate(tqdm(dl)):
            x, y = b["image"], b["label"]

            # If targeted: replace labels with target_class
            if targeted:
                y_target = torch.full_like(y, fill_value=target_class)
            else:
                y_target = None

            # --- Update Discriminator ---
            for _ in range(d_steps):
                x_adv, _ = wrap.perturb(x)
                g_loss, d_loss = advgan_losses(
                    D, target, x, y, x_adv,
                    lambda_adv=cfg["attack"]["advgan"]["lambda_adv"],
                    targeted=targeted,
                    y_target=y_target
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
                    lambda_adv=cfg["attack"]["advgan"]["lambda_adv"],
                    targeted=targeted,
                    y_target=y_target
                )
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_losses.append(g_loss.item())
  

        # === Epoch summary ===
        print(f"[Epoch {epoch+1}] g_loss={sum(g_losses)/len(g_losses):.4f} | d_loss={sum(d_losses)/len(d_losses):.4f}")

        # Monitor perturbations
        with torch.no_grad():
            x_adv, delta = wrap.perturb(x)
            print("  mean |δ| =", delta.abs().mean().item(),
                  "max |δ| =", delta.abs().max().item())

    out_dir = cfg["advgan"]["out_dir"]
    torch.save(G.state_dict(), f"{out_dir}/generator.pt")
    torch.save(D.state_dict(), f"{out_dir}/discriminator.pt")


if __name__ == "__main__":
    main()
