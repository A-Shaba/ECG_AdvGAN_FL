# webapp/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from webapp.utils.run_pipeline import run_step
from webapp.utils.load_logs import (
    get_dataset_stats, 
    get_training_logs, 
    get_advgan_logs, 
    get_attack_logs, 
    get_federated_logs
)
from pathlib import Path

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace-with-a-secure-random-key"

PIPELINE_CATEGORIES = {
    "Preprocessing": ["preprocess", "split_clients_iid", "split_clients_noniid"],
    "Training": ["train_baseline", "train_advgan"],
    "Evaluation": ["eval_baseline_clean", "eval_baseline_fgsm", "eval_baseline_pgd", "eval_baseline_advgan"],
    "Federated": ["run_federated"]
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", pipeline=PIPELINE_CATEGORIES)

# Dataset page
@app.route("/dataset")
def dataset():
    stats, plots = get_dataset_stats()
    return render_template("dataset.html", stats=stats, plots=plots)

# Baseline Training page
@app.route("/training")
def training():
    logs, plots = get_training_logs()
    return render_template("training.html", logs=logs, plots=plots)

# AdvGAN + Attacks page
@app.route("/attacks")
def attacks():
    advgan_stats, advgan_plots = get_advgan_logs()
    attack_stats, attack_plots = get_attack_logs()
    return render_template(
        "attacks.html",
        advgan_stats=advgan_stats,
        advgan_plots=advgan_plots,
        attack_stats=attack_stats,
        attack_plots=attack_plots
    )
    

# Federated page
@app.route("/federated")
def federated():
    stats, plots = get_federated_logs()
    return render_template("federated.html", stats=stats, plots=plots)

# Serve plot files
@app.route("/plots/<path:filename>")
def plots(filename):
    return send_from_directory(Path(app.static_folder) / "plots", filename)

# Reset project
@app.route("/reset_project", methods=["POST"])
def reset_project_api():
    try:
        logs = run_step("reset_project")
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"success": False, "logs": str(e)})

# API endpoint to run a pipeline step
@app.route("/run_step", methods=["POST"])
def run_step_api():
    step = request.form.get("step")
    try:
        logs = run_step(step)
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"success": False, "logs": str(e)})

# Logs Dashboard (overview)
@app.route("/logs")
def logs_dashboard():
    dataset_stats, dataset_plots = get_dataset_stats()
    training_stats, training_plots = get_training_logs()
    advgan_stats, advgan_plots = get_advgan_logs()
    attack_stats, attack_plots = get_attack_logs()
    federated_stats, federated_plots = get_federated_logs()

    return render_template(
        "logs.html",
        dataset_stats=dataset_stats, dataset_plots=dataset_plots,
        training_stats=training_stats, training_plots=training_plots,
        advgan_stats=advgan_stats, advgan_plots=advgan_plots,
        attack_stats=attack_stats, attack_plots=attack_plots,
        federated_stats=federated_stats, federated_plots=federated_plots,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
