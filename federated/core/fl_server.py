# federated/core/fl_server.py
import copy, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ECGImageDataset
from models.ecg_classifier_cnn import SmallECGCNN, resnet18, DeepECGCNN

def make_model(name, num_classes):
    if name == "small_cnn": return SmallECGCNN(1, num_classes)
    if name == "resnet18":  return resnet18(num_classes)
    if name == "deep_cnn":  return DeepECGCNN(1, num_classes)
    raise ValueError(name)

def make_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])

def fedavg(state_dicts, weights=None):
    """Simple FedAvg over a list of client state_dicts."""
    avg = copy.deepcopy(state_dicts[0])
    if weights is None:
        weights = [1.0/len(state_dicts)] * len(state_dicts)
    # sum weighted
    for k in avg.keys():
        avg[k].zero_()
        for sd, w in zip(state_dicts, weights):
            avg[k] += sd[k] * w
    return avg

def evaluate(model, dl, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            pred = model(x).argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct/total

class FLServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if (torch.cuda.is_available() and cfg["train"]["device"]=="cuda") else "cpu"

        tr = make_transform(cfg["data"]["resize"][0], cfg["data"]["resize"][1])
        ds_te = ECGImageDataset(cfg["data"]["test_csv"], transform=tr)
        self.num_classes = len(ds_te.classes)
        self.dl_te = DataLoader(ds_te,
                                batch_size=cfg["fl"]["eval_bs"],
                                shuffle=False,
                                num_workers=cfg["fl"].get("num_workers", 2),
                                pin_memory=(self.device=="cuda"))

        self.global_model = make_model(cfg["model"]["name"], self.num_classes).to(self.device)

    def broadcast(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_state_dicts, client_weights=None):
        new_sd = fedavg(client_state_dicts, client_weights)
        self.global_model.load_state_dict(new_sd, strict=True)

    def evaluate_global(self):
        return evaluate(self.global_model, self.dl_te, self.device)
