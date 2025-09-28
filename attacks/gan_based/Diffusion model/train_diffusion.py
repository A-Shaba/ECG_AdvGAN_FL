import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusion_gen import DiffusionGenerator
from data.dataset import ECGImageDataset

def main():
    import yaml
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    resize = tuple(cfg['data']['resize'])
    bs = cfg['train']['bs']
    device = torch.device(cfg['train'].get('device', 'cuda'))
    tr = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds = ECGImageDataset(cfg['data']['train_csv'], transform=tr)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    model = DiffusionGenerator(in_ch=1, img_size=resize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train'].get('lr', 1e-3))
    for epoch in range(cfg['train']['epochs']):
        model.train()
        losses = []
        for batch in dl:
            x = batch['image'].to(device)
            delta = model(x)
            x_rec = x + delta
            loss = torch.nn.functional.mse_loss(x_rec, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: {sum(losses)/len(losses):.4f}")
        torch.save(model.state_dict(), f"outputs/diffusion/diff_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), "outputs/diffusion/diff.pt")
    print("Saved diffusion generator.")

if __name__ == "__main__":
    main()