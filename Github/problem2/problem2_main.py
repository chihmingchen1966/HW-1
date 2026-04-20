# QCAA-HW1-VERIFIED
"""
QCAA Homework 1 - Problem 2
Skeleton script for comparing:
1) Explicit quantum model
2) Implicit quantum kernel method
3) Data reuploading circuit

What this script currently provides
-----------------------------------
- Reproducible dataset generation
- Circle dataset and moons dataset
- Train/test split and standardization
- Common utilities for timing, accuracy, and decision-boundary plotting
- Placeholders / interfaces for the three methods
- Output directory structure for figures and CSV summaries

What you still need to fill in
------------------------------
- train_explicit_quantum_model(...)
- train_implicit_quantum_kernel(...)
- train_data_reuploading_model(...)

Suggested workflow
------------------
1. First implement one method and verify the boundary plot works.
2. Reuse the same plotting / evaluation pipeline for the other two methods.
3. Produce the 3 x 2 decision-boundary figure required by the report.
"""

quantum_sentinel_7 = True

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================
# User configuration
# =========================
## STUDENT_ID_NUMERIC = 123456  # TODO: replace with the numeric part of your student ID
STUDENT_ID_NUMERIC = 10010022
OUTPUT_DIR = Path("problem2_outputs")
TEST_SIZE = 0.30
MOONS_SAMPLES = 200
MOONS_NOISE = 0.1

EXPLICIT_QM_QUBITS = 2
EXPLICIT_QM_EPOCHS = 50
EXPLICIT_QM_LR = 0.05

REUPLOAD_QUBITS = 2
REUPLOAD_LAYERS = 2
REUPLOAD_EPOCHS = 50
REUPLOAD_LR = 0.05

KERNEL_QUBITS = 2

# Decision-boundary mesh
MESH_STEP = 0.05
MESH_PADDING = 0.8


# =========================
# Reproducibility
# =========================
np.random.seed(STUDENT_ID_NUMERIC)


# =========================
# Data structures
# =========================
@dataclass
class MethodResult:
    method_name: str
    dataset_name: str
    test_accuracy: float
    complexity_measure: float
    complexity_label: str
    training_time_sec: float


# =========================
# Dataset generation
# =========================
def make_circle_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circle dataset for Problem 2.
    You may adjust n_samples / factor / noise to better match the reference setup.
    """
    X, y = make_circles(
        n_samples=200,
        factor=0.5,
        noise=0.05,
        random_state=seed,
    )
    return X, y


def make_moons_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(
        n_samples=MOONS_SAMPLES,
        noise=MOONS_NOISE,
        random_state=seed,
    )
    return X, y


def prepare_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =========================
# Plotting helpers
# =========================
def make_meshgrid(X: np.ndarray, step: float = MESH_STEP, padding: float = MESH_PADDING):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    return xx, yy


def plot_raw_dataset(X: np.ndarray, y: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(5.2, 4.6))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_decision_boundary(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
    save_path: Path,
):
    X_all = np.vstack([X_train, X_test])
    xx, yy = make_meshgrid(X_all)
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)

    plt.figure(figsize=(5.6, 4.8))
    plt.contourf(xx, yy, Z, alpha=0.35)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", edgecolors="k", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="^", edgecolors="k", label="Test")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_six_panel_boundary_grid(
    figure_items: Dict[Tuple[str, str], Dict],
    save_path: Path,
):
    """
    figure_items keys:
        ("Explicit QM", "Circle"), ("Explicit QM", "Moons"), ...
    Each value should contain:
        {
            "predict_fn": callable,
            "X_train": ...,
            "y_train": ...,
            "X_test": ...,
            "y_test": ...
        }
    """
    methods = ["Explicit QM", "Implicit Q Kernel", "Data Reuploading"]
    datasets = ["Circle", "Moons"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            item = figure_items[(method, dataset)]

            X_all = np.vstack([item["X_train"], item["X_test"]])
            xx, yy = make_meshgrid(X_all)
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = item["predict_fn"](grid).reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.35)
            ax.scatter(item["X_train"][:, 0], item["X_train"][:, 1], c=item["y_train"], marker="o", edgecolors="k")
            ax.scatter(item["X_test"][:, 0], item["X_test"][:, 1], c=item["y_test"], marker="^", edgecolors="k")
            ax.set_title(f"{method} - {dataset}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()


# =========================
# Method interfaces
# =========================
def train_explicit_quantum_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], MethodResult]:
    torch.manual_seed(STUDENT_ID_NUMERIC)

    dev = qml.device("default.qubit", wires=EXPLICIT_QM_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights, x):
        # S(x): data encoding
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # W(theta): trainable circuit
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(weights[2], wires=0)
        qml.RZ(weights[3], wires=1)

        return qml.expval(qml.PauliZ(0))

    class ExplicitQMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(0.01 * torch.randn(4))

        def forward(self, x_batch):
            outputs = []
            for x in x_batch:
                out = circuit(self.weights, x)
                outputs.append(out)
            return torch.stack(outputs)

    model = ExplicitQMClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=EXPLICIT_QM_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    start_time = time.time()

    for epoch in range(1, EXPLICIT_QM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                test_logits = model(X_test_t)
                test_pred = (torch.sigmoid(test_logits) >= 0.5).int().cpu().numpy()
                test_acc = accuracy_score(y_test, test_pred)
            print(f"[Explicit QM] epoch={epoch:03d} loss={loss.item():.6f} test_acc={test_acc:.4f}")

    elapsed = time.time() - start_time

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).int().cpu().numpy()
        test_acc = accuracy_score(y_test, test_pred)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_t)
            pred = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        return pred

    result = MethodResult(
        method_name="Explicit QM",
        dataset_name="",
        test_accuracy=float(test_acc),
        complexity_measure=float(model.weights.numel()),
        complexity_label="Trainable Params",
        training_time_sec=float(elapsed),
    )
    return predict_fn, result


def train_implicit_quantum_kernel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], MethodResult]:
    dev = qml.device("default.qubit", wires=KERNEL_QUBITS)

    @qml.qnode(dev)
    def feature_state(x):
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    def kernel_value(x1: np.ndarray, x2: np.ndarray) -> float:
        psi1 = feature_state(x1)
        psi2 = feature_state(x2)
        return float(np.abs(np.vdot(psi1, psi2)) ** 2)

    def compute_kernel_matrix(A: np.ndarray, B: np.ndarray, label: str = "kernel") -> np.ndarray:
        K = np.zeros((len(A), len(B)))
        total_rows = len(A)
        for i, a in enumerate(A):
            if i % 20 == 0 or i == total_rows - 1:
                print(f"[Implicit Q Kernel] computing {label}: row {i + 1}/{total_rows}")
            for j, b in enumerate(B):
                K[i, j] = kernel_value(a, b)
        return K

    start_time = time.time()

    K_train = compute_kernel_matrix(X_train, X_train, label="K_train")
    svc = SVC(kernel="precomputed")
    svc.fit(K_train, y_train)

    K_test = compute_kernel_matrix(X_test, X_train, label="K_test")
    test_pred = svc.predict(K_test)
    test_acc = accuracy_score(y_test, test_pred)

    elapsed = time.time() - start_time
    kernel_evals = float(len(X_train) * len(X_train) + len(X_test) * len(X_train))

    def predict_fn(X: np.ndarray) -> np.ndarray:
        K = compute_kernel_matrix(X, X_train, label="K_predict")
        return svc.predict(K)

    result = MethodResult(
        method_name="Implicit Q Kernel",
        dataset_name="",
        test_accuracy=float(test_acc),
        complexity_measure=kernel_evals,
        complexity_label="Kernel Evals",
        training_time_sec=float(elapsed),
    )
    return predict_fn, result


def train_data_reuploading_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], MethodResult]:
    torch.manual_seed(STUDENT_ID_NUMERIC)

    dev = qml.device("default.qubit", wires=REUPLOAD_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights, x):
        for layer in range(REUPLOAD_LAYERS):
            # Re-upload the same input in every layer
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)

            qml.RY(weights[layer, 0], wires=0)
            qml.RY(weights[layer, 1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[layer, 2], wires=0)
            qml.RZ(weights[layer, 3], wires=1)

        return qml.expval(qml.PauliZ(0))

    class ReuploadClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(0.01 * torch.randn(REUPLOAD_LAYERS, 4))

        def forward(self, x_batch):
            outputs = []
            for x in x_batch:
                out = circuit(self.weights, x)
                outputs.append(out)
            return torch.stack(outputs)

    model = ReuploadClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=REUPLOAD_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    start_time = time.time()

    for epoch in range(1, REUPLOAD_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                test_logits = model(X_test_t)
                test_pred = (torch.sigmoid(test_logits) >= 0.5).int().cpu().numpy()
                test_acc = accuracy_score(y_test, test_pred)
            print(f"[Data Reuploading] epoch={epoch:03d} loss={loss.item():.6f} test_acc={test_acc:.4f}")

    elapsed = time.time() - start_time

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).int().cpu().numpy()
        test_acc = accuracy_score(y_test, test_pred)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_t)
            pred = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        return pred

    result = MethodResult(
        method_name="Data Reuploading",
        dataset_name="",
        test_accuracy=float(test_acc),
        complexity_measure=float(model.weights.numel()),
        complexity_label="Trainable Params",
        training_time_sec=float(elapsed),
    )
    return predict_fn, result


# =========================
# Experiment runner
# =========================
def run_one_method(
    trainer: Callable,
    method_name: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    predict_fn, result = trainer(X_train, y_train, X_test, y_test)
    result.method_name = method_name
    result.dataset_name = dataset_name
    return predict_fn, result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "raw_datasets").mkdir(exist_ok=True)
    (OUTPUT_DIR / "boundaries").mkdir(exist_ok=True)

    # 1) Build datasets
    circle_X, circle_y = make_circle_dataset(STUDENT_ID_NUMERIC)
    moons_X, moons_y = make_moons_dataset(STUDENT_ID_NUMERIC)

    plot_raw_dataset(circle_X, circle_y, "Circle dataset", OUTPUT_DIR / "raw_datasets" / "circle_dataset.png")
    plot_raw_dataset(moons_X, moons_y, "Moons dataset", OUTPUT_DIR / "raw_datasets" / "moons_dataset.png")

    circle_train_X, circle_test_X, circle_train_y, circle_test_y, circle_scaler = prepare_dataset(
        circle_X, circle_y, STUDENT_ID_NUMERIC
    )
    moons_train_X, moons_test_X, moons_train_y, moons_test_y, moons_scaler = prepare_dataset(
        moons_X, moons_y, STUDENT_ID_NUMERIC
    )

    figure_items = {}
    results = []

    pred_fn, result = run_one_method(
        train_explicit_quantum_model,
        "Explicit QM",
        "Circle",
        circle_train_X, circle_train_y, circle_test_X, circle_test_y
    )
    results.append(result)
    figure_items[("Explicit QM", "Circle")] = {
        "predict_fn": pred_fn,
        "X_train": circle_train_X,
        "y_train": circle_train_y,
        "X_test": circle_test_X,
        "y_test": circle_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        circle_train_X, circle_train_y,
        circle_test_X, circle_test_y,
        "Explicit QM - Circle",
        OUTPUT_DIR / "boundaries" / "explicit_qm_circle.png",
    )

    pred_fn, result = run_one_method(
        train_explicit_quantum_model,
        "Explicit QM",
        "Moons",
        moons_train_X, moons_train_y, moons_test_X, moons_test_y
    )
    results.append(result)
    figure_items[("Explicit QM", "Moons")] = {
        "predict_fn": pred_fn,
        "X_train": moons_train_X,
        "y_train": moons_train_y,
        "X_test": moons_test_X,
        "y_test": moons_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        moons_train_X, moons_train_y,
        moons_test_X, moons_test_y,
        "Explicit QM - Moons",
        OUTPUT_DIR / "boundaries" / "explicit_qm_moons.png",
    )

    pred_fn, result = run_one_method(
        train_data_reuploading_model,
        "Data Reuploading",
        "Circle",
        circle_train_X, circle_train_y, circle_test_X, circle_test_y
    )
    results.append(result)
    figure_items[("Data Reuploading", "Circle")] = {
        "predict_fn": pred_fn,
        "X_train": circle_train_X,
        "y_train": circle_train_y,
        "X_test": circle_test_X,
        "y_test": circle_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        circle_train_X, circle_train_y,
        circle_test_X, circle_test_y,
        "Data Reuploading - Circle",
        OUTPUT_DIR / "boundaries" / "data_reuploading_circle.png",
    )

    pred_fn, result = run_one_method(
        train_data_reuploading_model,
        "Data Reuploading",
        "Moons",
        moons_train_X, moons_train_y, moons_test_X, moons_test_y
    )
    results.append(result)
    figure_items[("Data Reuploading", "Moons")] = {
        "predict_fn": pred_fn,
        "X_train": moons_train_X,
        "y_train": moons_train_y,
        "X_test": moons_test_X,
        "y_test": moons_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        moons_train_X, moons_train_y,
        moons_test_X, moons_test_y,
        "Data Reuploading - Moons",
        OUTPUT_DIR / "boundaries" / "data_reuploading_moons.png",
    )

    pred_fn, result = run_one_method(
        train_implicit_quantum_kernel,
        "Implicit Q Kernel",
        "Circle",
        circle_train_X, circle_train_y, circle_test_X, circle_test_y
    )
    results.append(result)
    figure_items[("Implicit Q Kernel", "Circle")] = {
        "predict_fn": pred_fn,
        "X_train": circle_train_X,
        "y_train": circle_train_y,
        "X_test": circle_test_X,
        "y_test": circle_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        circle_train_X, circle_train_y,
        circle_test_X, circle_test_y,
        "Implicit Q Kernel - Circle",
        OUTPUT_DIR / "boundaries" / "implicit_q_kernel_circle.png",
    )

    pred_fn, result = run_one_method(
        train_implicit_quantum_kernel,
        "Implicit Q Kernel",
        "Moons",
        moons_train_X, moons_train_y, moons_test_X, moons_test_y
    )
    results.append(result)
    figure_items[("Implicit Q Kernel", "Moons")] = {
        "predict_fn": pred_fn,
        "X_train": moons_train_X,
        "y_train": moons_train_y,
        "X_test": moons_test_X,
        "y_test": moons_test_y,
    }

    plot_decision_boundary(
        pred_fn,
        moons_train_X, moons_train_y,
        moons_test_X, moons_test_y,
        "Implicit Q Kernel - Moons",
        OUTPUT_DIR / "boundaries" / "implicit_q_kernel_moons.png",
    )

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df.to_csv(OUTPUT_DIR / "problem2_comparison_table.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "problem2_partial_results.csv", index=False)

    plot_six_panel_boundary_grid(
        figure_items,
        OUTPUT_DIR / "problem2_decision_boundaries_3x2.png"
    )

    print("Step-4 runs completed: Explicit QM + Data Reuploading + Implicit Q Kernel on Circle and Moons.")
    print("Created boundary figures:")
    print("  - problem2_outputs/boundaries/explicit_qm_circle.png")
    print("  - problem2_outputs/boundaries/explicit_qm_moons.png")
    print("  - problem2_outputs/boundaries/data_reuploading_circle.png")
    print("  - problem2_outputs/boundaries/data_reuploading_moons.png")
    print("  - problem2_outputs/boundaries/implicit_q_kernel_circle.png")
    print("  - problem2_outputs/boundaries/implicit_q_kernel_moons.png")
    print("Created summary figures/tables:")
    print("  - problem2_outputs/problem2_comparison_table.csv")
    print("  - problem2_outputs/problem2_decision_boundaries_3x2.png")


if __name__ == "__main__":
    main()


