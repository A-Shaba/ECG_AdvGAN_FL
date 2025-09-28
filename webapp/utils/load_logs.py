# webapp/utils/load_logs.py
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# -----------------------
# Directories
# -----------------------
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
PLOTS_DIR = STATIC_DIR / "plots"
STATIC_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def save_plot(fig, fname):
    """Helper to save a matplotlib figure safely"""
    path = PLOTS_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    try:
        fig.savefig(path)
    except Exception as e:
        print(f"[load_logs] Failed to save plot {path}: {e}")
    plt.close(fig)

def clean_old_attack_plots(prefixes=["asr_", "robust_acc_"]):
    """Delete old attack-related plots to avoid clutter"""
    for fname in os.listdir(PLOTS_DIR):
        if any(fname.startswith(p) for p in prefixes):
            try:
                os.remove(PLOTS_DIR / fname)
            except Exception as e:
                print(f"[load_logs] Could not delete old plot {fname}: {e}")

# -----------------------
# 📊 DATASET
# -----------------------
def get_dataset_stats():
    proc = Path("data/processed")
    train_csv = proc / "train.csv"
    feat_json = proc / "feature_cols.json"
    plots, stats = [], {}

    if not train_csv.exists():
        return {"error": "train.csv not found"}, plots

    df = pd.read_csv(train_csv)
    stats["n_samples"] = len(df)
    if "Diagnosis" in df.columns:
        stats["class_counts"] = df["Diagnosis"].value_counts().to_dict()

    # feature histograms (max 4 features)
    try:
        feature_cols = json.load(open(feat_json)) if feat_json.exists() else [
            c for c in df.columns if c != "Diagnosis"
        ]
    except Exception:
        feature_cols = [c for c in df.columns if c != "Diagnosis"]

    for i, col in enumerate(feature_cols[:4]):
        fig = plt.figure(figsize=(6, 3))
        df[col].hist(bins=40)
        plt.title(f"Histogram: {col}")
        fname = f"hist_{i}_{col}.png".replace(" ", "_")
        plt.tight_layout()
        save_plot(fig, fname)
        plots.append(fname)

    return stats, plots


# -----------------------
# 🧑‍🏫 BASELINE TRAINING
# -----------------------
def get_training_logs():
    out = Path("outputs/baseline")
    log_csv = out / "training_log.csv"
    eval_json = out / "eval_results.json"
    plots, stats = [], {}

    if not log_csv.exists():
        return {"error": "training_log.csv not found"}, plots

    df = pd.read_csv(log_csv)
    stats = df.to_dict(orient="list")
    stats["val_acc"] = df.get("val_acc", []).tolist() if "val_acc" in df.columns else []
    stats["val_loss"] = df.get("val_loss", []).tolist() if "val_loss" in df.columns else []
    stats["epoch"] = df.get("epoch", []).tolist() if "epoch" in df.columns else []

    # Loss plot
    if {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        fig = plt.figure(figsize=(6, 3))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
        plt.legend(); plt.title("Loss")
        save_plot(fig, "baseline_loss.png")
        plots.append("baseline_loss.png")

    # Accuracy plot
    if {"epoch", "val_acc"}.issubset(df.columns):
        fig = plt.figure(figsize=(6, 3))
        plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy")
        plt.legend(); plt.title("Validation Accuracy")
        save_plot(fig, "baseline_acc.png")
        plots.append("baseline_acc.png")

    # Confusion Matrix + ROC if eval exists
    if eval_json.exists():
        results = json.load(open(eval_json))
        y_true = results.get("y_true", [])
        y_pred = results.get("y_pred", [])
        y_score = results.get("y_score", None)

        if y_true and y_pred:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig = disp.plot(cmap="Blues").figure
            plt.title("Confusion Matrix")
            save_plot(fig, "baseline_confusion.png")
            plots.append("baseline_confusion.png")

        if y_true and y_score:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("ROC Curve")
            plt.legend()
            save_plot(fig, "baseline_roc.png")
            plots.append("baseline_roc.png")

    return stats, plots


# -----------------------
# ⚡ ADVGAN TRAINING
# -----------------------
def get_advgan_logs():
    out = Path("outputs/advgan")
    log_csv = out / "training_log.csv"
    plots, stats = [], {}

    if not log_csv.exists():
        return {"error": "advgan training_log.csv not found"}, plots

    df = pd.read_csv(log_csv)
    stats = df.to_dict(orient="list")
    stats["gen_loss"] = df.get("gen_loss", []).tolist()
    stats["disc_loss"] = df.get("disc_loss", []).tolist()
    stats["asr"] = df.get("asr", []).tolist()
    stats["epoch"] = df.get("epoch", []).tolist()

    if "gen_loss" in df.columns:
        fig = plt.figure()
        plt.plot(df["epoch"], df["gen_loss"], label="Generator Loss")
        plt.title("Generator Loss")
        plt.legend()
        save_plot(fig, "advgan_gen_loss.png")
        plots.append("advgan_gen_loss.png")

    if "disc_loss" in df.columns:
        fig = plt.figure()
        plt.plot(df["epoch"], df["disc_loss"], label="Discriminator Loss")
        plt.title("Discriminator Loss")
        plt.legend()
        save_plot(fig, "advgan_disc_loss.png")
        plots.append("advgan_disc_loss.png")

    if "asr" in df.columns:
        fig = plt.figure()
        plt.plot(df["epoch"], df["asr"], label="Attack Success Rate")
        plt.title("ASR")
        plt.legend()
        save_plot(fig, "advgan_asr.png")
        plots.append("advgan_asr.png")

    return stats, plots


# -----------------------
# 🛡 ATTACK EVALUATION
# -----------------------
def get_attack_logs():
    eval_json = Path("outputs") / "baseline" / "eval_results.json"
    if not eval_json.exists():
        return {"error": "No evaluation results found."}, []

    results = json.load(open(eval_json))

    # Prepare stats dict for template
    stats = {}
    stats['clean_acc'] = results.get('clean_acc', "N/A")

    attacks = results.get("attacks", {})

    # Clean old attack plots first to avoid clutter
    clean_old_attack_plots()

    plots = []

    # Generate per-class ASR plots for each attack (if available)
    for attack_name, attack_data in attacks.items():
        per_class_asr = attack_data.get("per_class_asr", {})
        if per_class_asr:
            labels = list(per_class_asr.keys())
            values = list(per_class_asr.values())

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values, color='tomato')
            ax.set_title(f'Per-class ASR ({attack_name.upper()})')
            ax.set_xlabel('Class')
            ax.set_ylabel('Attack Success Rate')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)

            plot_filename = f"asr_{attack_name}.png"
            save_plot(fig, plot_filename)
            plots.append(plot_filename)

    # Generate robust accuracy bar chart for all attacks (skip attacks with null robust_acc)
    robust_accs = {name.upper(): data.get("robust_acc") for name, data in attacks.items() if data.get("robust_acc") is not None}
    if robust_accs:
        names = list(robust_accs.keys())
        acc_values = list(robust_accs.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, acc_values, color='royalblue')
        ax.set_title("Robust Accuracy per Attack")
        ax.set_xlabel("Attack")
        ax.set_ylabel("Robust Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar, val in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha='center', fontsize=9)

        plot_filename = "robust_acc_all_attacks.png"
        save_plot(fig, plot_filename)
        plots.append(plot_filename)

    # Also add raw attack stats to return for display if needed
    stats['attacks'] = attacks

    return stats, plots


# -----------------------
# 🌐 FEDERATED LEARNING
# -----------------------
def get_federated_logs():
    outdir = Path("outputs/federated")
    acc_csv = outdir / "round_accuracies.csv"
    plots, stats = [], {}

    if not acc_csv.exists():
        return {"error": "No federated logs found"}, plots

    df = pd.read_csv(acc_csv)
    stats = df.to_dict(orient="list")
    stats["round"] = df.get("round", []).tolist()
    stats["global_acc"] = df.get("global_acc", []).tolist()
    stats["clients"] = df.get("clients", []).tolist() if "clients" in df.columns else []

    if {"round", "global_acc"}.issubset(df.columns):
        fig = plt.figure()
        plt.plot(df["round"], df["global_acc"], marker="o")
        plt.title("Global Accuracy per Round")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        save_plot(fig, "federated_global_acc.png")
        plots.append("federated_global_acc.png")

    return stats, plots
