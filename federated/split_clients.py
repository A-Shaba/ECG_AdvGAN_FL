# data/split_clients.py
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np

def split_clients(input_csv, out_dir, n_clients=3, label_col="label", iid=True, random_state=42):
    df = pd.read_csv(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if iid:
        # IID: stratified shuffle, then split equally
        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
        for i, (_, idx) in enumerate(skf.split(df, df[label_col])):
            client_df = df.iloc[idx].reset_index(drop=True)
            client_df.to_csv(out_dir / f"client{i}.csv", index=False)
            print(f"Saved IID client {i} with {len(client_df)} samples")
    else:
        # Non-IID: sort by label, chunk into clients
        df_sorted = df.sort_values(by=label_col).reset_index(drop=True)
        chunks = np.array_split(df_sorted, n_clients)
        for i, chunk in enumerate(chunks):
            chunk.to_csv(out_dir / f"client{i}.csv", index=False)
            print(f"Saved non-IID client {i} with {len(chunk)} samples")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="Path to full dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output directory for client CSVs")
    ap.add_argument("--n-clients", type=int, default=3)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--iid", action="store_true", help="Use IID split (default non-IID)")
    args = ap.parse_args()

    split_clients(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        n_clients=args.n_clients,
        label_col=args.label_col,
        iid=args.iid
    )