# training/eval_baseline.py
import argparse, yaml, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import SmallECGCNN, resnet18_gray
from attacks.gradient_based.fgsm import fgsm_attack
from utils.metrics import accuracy

def make_model(name, num_classes):
    return SmallECGCNN(1,num_classes) if name=="small_cnn" else resnet18_gray(num_classes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attack", default="none", choices=["none","fgsm"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tr = transforms.Compose([transforms.Resize(tuple(cfg["data"]["resize"])),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5],[0.5])])
    ds_te = ECGImageDataset(cfg["data"]["test_csv"], transform=tr)
    dl_te = DataLoader(ds_te, batch_size=cfg["eval"]["bs"], shuffle=False)

    model = make_model(cfg["model"]["name"], len(ds_te.classes))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # clean
    clean_acc = accuracy(model, dl_te)
    # robust
    if args.attack=="fgsm":
        rob_acc = accuracy(model, dl_te, adversary=lambda m,x,y: fgsm_attack(m,x,y,eps=cfg["eval"]["fgsm_eps"]))
    else:
        rob_acc = None
    print({"clean_acc":clean_acc, "robust_acc":rob_acc})

if __name__=="__main__":
    main()
