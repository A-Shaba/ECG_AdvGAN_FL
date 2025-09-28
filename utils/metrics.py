# utils/metrics.py
import torch

def accuracy_with_meta(model, dataloader, device, adversary=None, return_details=False):
    model.eval()
    all_preds, all_labels, correct_mask = [], [], []

    correct, total = 0, 0

    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        meta = batch.get("meta", None)
        if meta is not None:
            meta = meta.to(device)

        if adversary is not None:
            # adversary is expected to return a tensor on the same device
            x = adversary(model, x, y, meta=meta)

        with torch.no_grad():
            logits = model(x, meta) if meta is not None else model(x)
            preds = logits.argmax(dim=1)

        correct_preds = (preds == y)
        correct += correct_preds.sum().item()
        total += y.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        correct_mask.append(correct_preds.cpu())

    acc = correct / total if total > 0 else 0.0

    if return_details:
        return acc, torch.cat(all_preds), torch.cat(all_labels), torch.cat(correct_mask)
    return acc


def compute_per_class_asr(clean_labels, robust_preds):
    """
    ASR = Attack Success Rate per class
    For each class, ASR = fraction of samples originally in that class (and correctly classified) that were flipped by the attack.
    Returns dict {class_index: asr}
    """
    per_class_asr = {}
    for cls in torch.unique(clean_labels):
        cls_mask = (clean_labels == cls)
        n_total = cls_mask.sum().item()
        if n_total == 0:
            per_class_asr[int(cls)] = None
            continue

        flipped = (robust_preds[cls_mask] != cls).sum().item()
        per_class_asr[int(cls)] = round(flipped / n_total, 4)

    return per_class_asr