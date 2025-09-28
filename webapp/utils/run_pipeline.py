# webapp/utils/run_pipeline.py
import subprocess, shlex
from pathlib import Path
import os

# Map names -> commands (run from project root)
# IMPORTANT: adjust paths/config filenames to match your repo setup.
CMD_MAP = {
    "split_diagnosis": "python data/split_diagnosis.py",
    "preprocess": "python data/preprocess_dataset.py --input data/raw --output data/processed --val-size 0.1 --test-size 0.1",
    "preprocess_synthetic": "python data/preprocess_dataset.py --input data/synthetic --output data/processed_synthetic --make-synthetic --val-size 0.1 --test-size 0.1",
    "preprocess_metadata": "python data/preprocess_dataset.py --input data/raw --output data/processed --csv data/raw/dataset.csv --val-size 0.1 --test-size 0.1",
    "split_clients_iid": "python -m data.split_clients --input-csv data/processed/train.csv --out-dir data/processed --n-clients 3 --iid",
    "split_clients_noniid": "python -m data.split_clients --input-csv data/processed/train.csv --out-dir data/processed --n-clients 3",
    "train_baseline": "python -m training.train_baseline --config experiments/configs/baseline.yaml",
    "eval_baseline_clean": "python -m training.eval_baseline --config experiments/configs/baseline.yaml --ckpt outputs/baseline/baseline_best.pt --attack none",
    "eval_baseline_fgsm": "python -m training.eval_baseline --config experiments/configs/baseline.yaml --ckpt outputs/baseline/baseline_best.pt --attack fgsm",
    "eval_baseline_pgd": "python -m training.eval_baseline --config experiments/configs/baseline.yaml --ckpt outputs/baseline/baseline_best.pt --attack pgd",
    "eval_baseline_advgan": "python -m training.eval_baseline --config experiments/configs/baseline.yaml --ckpt outputs/baseline/baseline_best.pt --attack advgan",
    "train_advgan": "python -m attacks.gan_based.advGAN.train_advgan --config experiments/configs/advgan.yaml",
    "run_federated": "python -m federated.run_federated --config experiments/configs/federated.yaml",
    "reset_project": "python -m utils.reset_project"
}

def available_steps():
    return list(CMD_MAP.keys())

def run_command(cmd):
    """
    Run command (string), capture stdout+stderr, and return them as a string.
    Runs from the repository root (parent of webapp).
    """
    repo_root = Path(__file__).resolve().parents[2]  # webapp/utils -> repo root
    # Ensure we run from repo root so module imports work
    shell = True if os.name == "nt" else False
    proc = subprocess.run(cmd if shell else shlex.split(cmd),
                          cwd=str(repo_root),
                          capture_output=True,
                          text=True,
                          shell=shell)
    out = f"$ {cmd}\n\n"
    out += proc.stdout or ""
    if proc.stderr:
        out += "\n===== STDERR =====\n" + proc.stderr
    return out

def run_step(step_name):
    if step_name not in CMD_MAP:
        raise ValueError(f"Unknown step {step_name}")
    cmd = CMD_MAP[step_name]
    return run_command(cmd)
