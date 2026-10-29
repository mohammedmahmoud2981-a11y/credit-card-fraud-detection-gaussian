
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, auc)

OUT = "outputs_fraud"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------
# 1) Gaussian anomaly detector from scratch
#    p(x) = prod_j N(x_j ; mu_j, sigma_j^2)  -> we use log p(x) for stability
# ---------------------------------------------------------------
class GaussianAnomalyDetector:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.var = X.var(axis=0) + 1e-9  # avoid division by zero
        return self

    def log_prob(self, X):
        return (-0.5 * np.log(2 * np.pi * self.var)
                - (X - self.mu) ** 2 / (2 * self.var)).sum(axis=1)

    def predict(self, X, log_eps):
        return (self.log_prob(X) < log_eps).astype(int)  # 1 = anomaly


def select_threshold(log_p, y):
    """Pick epsilon (on log scale) that maximizes F1 on the CV set."""
    best_eps, best_f1 = None, -1
    for eps in np.linspace(log_p.min(), np.percentile(log_p, 99.5), 1000):
        pred = (log_p < eps).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_eps, best_f1 = eps, f1
    return best_eps, best_f1


# ---------------------------------------------------------------
# 2) Data
# ---------------------------------------------------------------
def load_data(demo=False, path="creditcard.csv"):
    if demo or not os.path.exists(path):
        print(">> Using synthetic demo data (creditcard.csv not found)")
        n, nf = 20000, 28
        X = rng.normal(0, 1, (n, nf))
        y = np.zeros(n, dtype=int)
        idx = rng.choice(n, 60, replace=False)
        X[idx] += rng.normal(0, 3, (60, nf)) + 2
        y[idx] = 1
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, nf + 1)])
        df["Time"] = np.arange(n)
        df["Amount"] = np.abs(rng.lognormal(3, 1.2, n))
        df["Class"] = y
        return df
    return pd.read_csv(path)


def main(demo):
    df = load_data(demo)
    print(f"Shape: {df.shape}")
    print(df["Class"].value_counts().rename({0: "Normal", 1: "Fraud"}))
    print(f"Fraud rate: {df['Class'].mean():.4%}\n")

    # ------------------ feature engineering ------------------
    # V1..V28 are already PCA outputs. Amount is heavily skewed -> log transform
    # to make it closer to a Gaussian. Time is dropped (not informative).
    df["LogAmount"] = np.log1p(df["Amount"])
    features = [c for c in df.columns if c.startswith("V")] + ["LogAmount"]

    # Optional: keep only features that look reasonably Gaussian / informative.
    # Anomaly detection works best with features that differ between classes.
    X_all, y_all = df[features].values, df["Class"].values

    # ------------------ split (the course recipe) ------------------
    # Train: ONLY normal examples (60%)
    # CV   : 20% normal + half of frauds
    # Test : 20% normal + other half of frauds
    normal = np.where(y_all == 0)[0]; fraud = np.where(y_all == 1)[0]
    rng.shuffle(normal); rng.shuffle(fraud)
    n_tr = int(0.6 * len(normal)); n_cv = int(0.2 * len(normal))
    tr_idx = normal[:n_tr]
    cv_idx = np.r_[normal[n_tr:n_tr + n_cv], fraud[: len(fraud) // 2]]
    te_idx = np.r_[normal[n_tr + n_cv:], fraud[len(fraud) // 2:]]

    X_tr = X_all[tr_idx]
    X_cv, y_cv = X_all[cv_idx], y_all[cv_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]
    print(f"Train: {len(X_tr)} (all normal) | CV: {len(X_cv)} ({y_cv.sum()} fraud) "
          f"| Test: {len(X_te)} ({y_te.sum()} fraud)\n")

    # ------------------ fit + choose epsilon on CV ------------------
    det = GaussianAnomalyDetector().fit(X_tr)
    eps, best_f1 = select_threshold(det.log_prob(X_cv), y_cv)
    print(f"Best log(epsilon) = {eps:.2f}  |  CV F1 = {best_f1:.3f}\n")

    # ------------------ evaluate on TEST only once ------------------
    pred = det.predict(X_te, eps)
    p = precision_score(y_te, pred, zero_division=0)
    r = recall_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred, zero_division=0)
    cm = confusion_matrix(y_te, pred)
    print("=== TEST RESULTS ===")
    print(f"Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f}")
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm, "\n")
    print("NOTE: accuracy is NOT used - predicting 'all normal' gives ~99.8% accuracy!\n")

    # ------------------ plots ------------------
    lp_te = det.log_prob(X_te)
    plt.figure(figsize=(8, 4))
    lo = np.percentile(lp_te, 0.1)
    plt.hist(lp_te[y_te == 0], bins=80, alpha=0.6, label="Normal", density=True, range=(lo, lp_te.max()))
    plt.hist(lp_te[y_te == 1], bins=40, alpha=0.7, label="Fraud", density=True, range=(lo, lp_te.max()))
    plt.axvline(eps, color="k", ls="--", label="threshold")
    plt.xlabel("log p(x)"); plt.legend(); plt.title("Score distribution (test)")
    plt.tight_layout(); plt.savefig(f"{OUT}/score_distribution.png", dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(y_te, -lp_te)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AUC = {auc(rec, prec):.3f})")
    plt.tight_layout(); plt.savefig(f"{OUT}/pr_curve.png", dpi=150); plt.close()

    # ------------------ business view: top-N alerts ------------------
    order = np.argsort(lp_te)  # most suspicious first
    for n in (100, 500):
        caught = y_te[order[:n]].sum()
        print(f"Review top {n} suspicious transactions -> catch {caught}/{y_te.sum()} frauds")
    print(f"\nDone. Results saved in ./{OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    main(ap.parse_args().demo)
