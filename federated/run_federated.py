# federated/run_federated.py
import argparse, yaml, torch
from pathlib import Path
from federated.core.fl_server import FLServer
from federated.core.fl_client import FLClient
from federated.fl_attack_wrapper import PoisonWithAdvGAN
from federated.split_clients import split_clients
from models.ecg_classifier_cnn import SmallECGCNN, resnet18, DeepECGCNN

def make_target(name, num_classes):
    if name == "small_cnn": return SmallECGCNN(1, num_classes)
    if name == "resnet18":  return resnet18(num_classes)
    if name == "deep_cnn":  return DeepECGCNN(1, num_classes)
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # server + eval loader
    server = FLServer(cfg)
    device = server.device

    # Prepare clients (3 clients, client_csvs in cfg)
    client_paths = [Path(p) for p in cfg["fl"]["client_csvs"]]
    if not all(p.exists() for p in client_paths):
        split_clients(cfg["data"]["train_csv"], Path(cfg["fl"]["out_dir"]), n_clients=len(client_paths),
                      label_col=cfg["data"].get("label_col","label"), iid=cfg["fl"].get("iid", True))
        client_paths = [Path(cfg["fl"]["out_dir"]) / f"client{i}.csv" for i in range(len(client_paths))]

    # Build a frozen target model for the attacker hook (same arch as global)
    # It loads the baseline ckpt the AdvGAN generator was trained against.
    target_ref = make_target(cfg["model"]["name"], server.num_classes).to(device).eval()
    target_ref.load_state_dict(torch.load(cfg["model"]["ckpt"], map_location=device))
    for p in target_ref.parameters(): p.requires_grad = False

    # Malicious hook for client 2 (index 1) — uses trained AdvGAN generator
    adv_eps = cfg["attack"]["advgan"]["eps"]
    adv_ckpt = cfg["attack"]["advgan"]["model_ckpt"]
    poison_frac = cfg["fl"].get("poison_frac", 0.5)
    malicious_hook = PoisonWithAdvGAN(target_ref, ckpt_path=adv_ckpt, eps=adv_eps, frac=poison_frac, device=device)

    clients = []
    for i, csv in enumerate(client_paths):
        attack = malicious_hook if i == cfg["fl"].get("malicious_index", 1) else None
        clients.append(FLClient(i, cfg, csv_path=csv, attack_hook=attack, device=device))

    rounds = cfg["fl"]["rounds"]
    out_dir = Path(cfg["fl"].get("out_dir", "outputs/federated"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # FedAvg rounds
    for r in range(rounds):
        # broadcast
        global_sd = server.broadcast()
        for c in clients:
            c.set_weights(global_sd)

        # local training
        for c in clients:
            c.local_train()

        # aggregate
        client_sds = [c.get_weights() for c in clients]
        server.aggregate(client_sds)

        # evaluate
        acc = server.evaluate_global()
        print(f"[Round {r+1}/{rounds}] global clean accuracy = {acc:.4f}")

    # Save final global model
    torch.save(server.global_model.state_dict(), out_dir / "global_model.pt")

if __name__ == "__main__":
    main()
