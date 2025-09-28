import os
import shutil
import pandas as pd

# === CONFIGURAZIONE ===
csv_path =   "data/Ningbo/baseline_Ningbo.csv"  #"data/PTB_xl/baseline_PTB_xl.csv"      "data/SPH/baseline_SPH.csv"                 
images_dir = "data/Ningbo/images"                          
output_dir = "data/raw"                                      # <-- immagini e dataser.csv di output
diagnosis_cols = ["AVBI", "AF", "LBBB", "RBBB", "SNB", "NSR", "SNT"]

# Diagnosi da considerare (quelle tre principali)
target_diags = ["SNB", "NSR", "SNT"]

# === CREAZIONE CARTELLE DI OUTPUT ===
os.makedirs(output_dir, exist_ok=True)
for diag in target_diags + ["abnormal"]:
    os.makedirs(os.path.join(output_dir, diag), exist_ok=True)

# === LETTURA CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Converti Gender in numerico (M=0, F=1, altri=-1)
def gender_to_num(g):
    if g == "M": return 0
    if g == "F": return 1
    return -1

df["Gender"] = df["Gender"].apply(gender_to_num)

# Funzione per decidere la diagnosi finale
def get_label(row):
    active = [d for d in target_diags if row[d] == 1]
    if len(active) == 1:
        return active[0]
    elif len(active) > 1:
        return "abnormal"
    else:
        return None  # scarta

rows_for_csv = []

for idx, row in df.iterrows():
    name = row["Name"]
    image_path = os.path.join(images_dir, f"{name}_0.jpg")

    if not os.path.exists(image_path):
        print(f"[MISSING IMAGE] {image_path} non trovato.")
        continue

    label = get_label(row)
    if label is None:
        print(f"[NO DIAGNOSIS] {name} non rientra nelle diagnosi considerate.")
        continue

    # Copia immagine nella cartella corretta
    target_dir = os.path.join(output_dir, label)
    shutil.copy(image_path, target_dir)

    # Costruisci riga per CSV
    rows_for_csv.append({
        "Age": row["Age"],
        "Gender": row["Gender"],
        "filepath": os.path.join(target_dir, f"{name}_0.jpg"),
        "label": label
    })

# Scrivi CSV finale
import csv
csv_out = os.path.join(output_dir, "dataset.csv")
with open(csv_out, "w", newline="") as f:
    fieldnames = ["Age", "Gender", "filepath", "label"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_for_csv)

print(f"Dataset creato con {len(rows_for_csv)} campioni, CSV salvato in {csv_out}")
