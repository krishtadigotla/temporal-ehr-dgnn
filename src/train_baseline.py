import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/baseline_model.pkl"

def train_baseline():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_val, y_val = val.drop(columns=["target"]), val["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # Compatible with all XGBoost versions
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Use best iteration if supported
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        model.set_params(n_estimators=model.best_iteration)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n===== BASELINE (XGBoost) Results =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved baseline model → {MODEL_PATH}")

if __name__ == "__main__":
    train_baseline()
