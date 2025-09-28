# federated/core/fl_client.py
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

class FLClient:
    """
    One client holds:
      - its own dataloader
      - its local model copy (synced from server at round start)
      - an optional attack hook that can modify batches during training
    """
    def __init__(self, cid, cfg, csv_path, attack_hook=None, device=None):
        self.cid = cid
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() and cfg["train"]["device"]=="cuda" else "cpu")

        tr = make_transform(cfg["data"]["resize"][0], cfg["data"]["resize"][1])
        ds = ECGImageDataset(csv_path, transform=tr)
        self.classes = ds.classes
        self.dl = DataLoader(ds,
                             batch_size=cfg["fl"]["client_bs"],
                             shuffle=True,
                             num_workers=cfg["fl"].get("num_workers", 2),
                             pin_memory=(self.device=="cuda"))

        self.model = make_model(cfg["model"]["name"], len(self.classes)).to(self.device)
        self.attack_hook = attack_hook  # callable(model, x, y) -> (x_mod, y_mod) during local training

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def local_train(self):
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["fl"]["client_lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(self.cfg["fl"]["local_epochs"]):
            for batch in self.dl:
                x = batch["image"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)

                if self.attack_hook is not None:
                    x, y = self.attack_hook(self.model, x, y)

                opt.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
