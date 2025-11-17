import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from model import DGNN
import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/processed"
MODEL_PATH = "models/dgnn_best.pt"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(name):
    graphs = torch.load(os.path.join(DATA_DIR, f"{name}.pt"))
    return DataLoader(graphs, batch_size=1, shuffle=False)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for g in loader:
            g = g.to(DEVICE)

            node_logits = model(g.x, g.edge_index)
            graph_logit = node_logits.mean()

            prob = torch.sigmoid(graph_logit).item()
            y = float(g.y.cpu())

            preds.append(prob)
            labels.append(y)

    preds_bin = [1 if p > 0.5 else 0 for p in preds]

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds_bin) if len(set(labels)) > 1 else 0.0
    cm = confusion_matrix(labels, preds_bin)

    return auc, f1, preds, labels, cm


def plot_curve(y_true, y_prob, filename):
    # ROC or PR curve
    plt.figure()
    if "roc" in filename:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(y_true, y_prob)
    elif "pr" in filename:
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay.from_predictions(y_true, y_prob)

    plt.title(filename.replace(".png", "").upper())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def save_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()


def main():
    test_loader = load_split("test")
    first_graph = torch.load(os.path.join(DATA_DIR, "train.pt"))[0]

    model = DGNN(in_dim=first_graph.x.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    auc, f1, preds, labels, cm = evaluate(model, test_loader)

    print("\n📌 Test Results")
    print(f" AUC: {auc:.4f}")
    print(f" F1 : {f1:.4f}")
    print(f" Confusion Matrix:\n{cm}")

    # Save metrics as JSON
    results = {"AUC": auc, "F1": f1, "labels": labels, "preds": preds}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # plots
    plot_curve(labels, preds, "roc_curve.png")
    plot_curve(labels, preds, "pr_curve.png")
    save_confusion_matrix(cm)

    print("\n📁 Results saved to results/")
    print("📌 metrics.json, roc_curve.png, pr_curve.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
