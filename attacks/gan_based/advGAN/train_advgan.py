import argparse, yaml, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import SmallECGCNN, resnet18_gray
from .generator_2d import AdvGANGenerator
from .discriminator_2d import AdvGANDiscriminator
from .advgan import AdvGANWrapper, advgan_losses
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tr = transforms.Compose([transforms.Resize(tuple(cfg["data"]["resize"])),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5],[0.5])])
    ds = ECGImageDataset(cfg["data"]["train_csv"], transform=tr)
    dl = DataLoader(ds, batch_size=cfg["train"]["bs"], shuffle=True)

    target = resnet18_gray(len(ds.classes)) if cfg["model"]["name"]=="resnet18" else SmallECGCNN(1,len(ds.classes))
    target.load_state_dict(torch.load(cfg["model"]["ckpt"], map_location="cpu"))
    target.eval()

    G = AdvGANGenerator(1); D = AdvGANDiscriminator(1)
    wrap = AdvGANWrapper(target, G, D, eps=cfg["attack"]["eps"])
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["train"]["g_lr"])
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["train"]["d_lr"])

    for epoch in range(cfg["train"]["epochs"]):
        for b in tqdm(dl):
            x, y = b["image"], b["label"]
            x_adv, _ = wrap.perturb(x)
            # update D
            d_opt.zero_grad()
            g_loss, d_loss = advgan_losses(D, target, x, y, x_adv, cfg["attack"]["lambda_adv"])
            d_loss.backward(); d_opt.step()
            # update G
            g_opt.zero_grad()
            x_adv, _ = wrap.perturb(x)
            g_loss, _ = advgan_losses(D, target, x, y, x_adv, cfg["attack"]["lambda_adv"])
            g_loss.backward(); g_opt.step()
    torch.save(G.state_dict(), cfg["train"]["out_g"])
    torch.save(D.state_dict(), cfg["train"]["out_d"])

if __name__=="__main__":
    main()
