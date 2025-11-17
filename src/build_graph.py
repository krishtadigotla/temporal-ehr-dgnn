import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import os

DATA_DIR = "data/processed"
SAVE_DIR = "data/processed"
K = 8  # number of KNN neighbors per timestep


def df_to_graph(df: pd.DataFrame) -> list:
    """
    Converts ICU tabular time-series into temporal graph snapshots.
    Each snapshot = one unique value of iv_day_1.
    Label rule (corrected): snapshot = 1 if ANY patient entry that day has day_28_flg = 1.
    """

    y_all = torch.tensor(df["day_28_flg"].values, dtype=torch.float)

    time_col = "iv_day_1"
    time_steps = sorted(df[time_col].unique())

    drop_cols = [
        "day_28_flg", "mort_day_censored", "censor_flg",
        "hosp_exp_flg", "icu_exp_flg", "service_unit", time_col
    ]

    # keep only feature columns
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Make absolutely everything numeric
    feature_df = pd.get_dummies(feature_df, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(0)

    x_all = torch.tensor(feature_df.to_numpy(dtype=np.float32))

    graphs = []
    for t in time_steps:
        idx = np.where(df[time_col].values == t)[0]
        if len(idx) < 3:
            continue

        x_t = x_all[idx]

        # KNN edges
        knn = NearestNeighbors(n_neighbors=min(K + 1, len(idx)))
        knn.fit(x_t)
        _, neighbors = knn.kneighbors(x_t)

        edges = []
        for i, neigh in enumerate(neighbors):
            for n in neigh[1:]:   # ignore self-loop
                edges.append([idx[i], idx[n]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # ------------------------------------------
        # 🔥 NEW LABEL RULE → snapshot label = ANY 1
        # ------------------------------------------
        y_snapshot = (y_all[idx].sum() > 0).float()  # 1 if ANY mortality in this timestep

        graphs.append(
            Data(
                x=x_all,                     # full feature matrix
                edge_index=edge_index,       # dynamic edges
                y=y_snapshot.unsqueeze(0),   # shape [1]
                t=torch.tensor([t])          # timestep id
            )
        )

    return graphs


def process_split(split: str):
    csv_path = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(csv_path)
    graphs = df_to_graph(df)
    save_path = os.path.join(SAVE_DIR, f"{split}.pt")
    torch.save(graphs, save_path)
    print(f"✔ Saved {split}.pt → {len(graphs)} temporal snapshots")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
    print("🔥 Temporal graph building complete.")
