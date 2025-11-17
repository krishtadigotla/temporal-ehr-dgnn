import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from model import DGNN                     # ← DGNN now imported from model.py
import os

DATA_DIR = "data/processed"
SAVE_MODEL_PATH = "models/dgnn_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 120


def load_split(name):
    graphs = torch.load(os.path.join(DATA_DIR, f"{name}.pt"))
    return DataLoader(graphs, batch_size=1, shuffle=(name == "train"))


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for g in loader:
            g = g.to(DEVICE)

            node_logits = model(g.x, g.edge_index)
            graph_logit = node_logits.mean()       # scalar per snapshot

            prob = torch.sigmoid(graph_logit).item()
            y = float(g.y.cpu())

            preds.append(prob)
            labels.append(y)

    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds_bin) if len(set(labels)) > 1 else 0.0
    return auc, f1


def train():
    train_loader = load_split("train")
    val_loader   = load_split("val")

    first_graph = torch.load(os.path.join(DATA_DIR, "train.pt"))[0]
    in_dim = first_graph.x.shape[1]
    model = DGNN(in_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1

    print("🚀 Starting DGNN training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for g in train_loader:
            g = g.to(DEVICE)
            optimizer.zero_grad()

            node_logits = model(g.x, g.edge_index)
            graph_logit = node_logits.mean()

            graph_logit = graph_logit.view(1)          # ensures shape match
            graph_target = g.y.float().view(1)

            loss = criterion(graph_logit, graph_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_auc, val_f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"💾 Saved best → {SAVE_MODEL_PATH}")

    print(f"\n🏆 Best Validation AUC = {best_auc:.4f}")
    print("🔥 Training finished.")


if __name__ == "__main__":
    train()
