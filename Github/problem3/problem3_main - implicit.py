import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pennylane as qml
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# ==========================================
# 1. 基礎設定與隨機種子
# ==========================================
seed = 10010022
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的設備: {device}")

OUTPUT_DIR = Path("problem3_outputs_variational_implicit")
OUTPUT_DIR.mkdir(exist_ok=True)

# Important:
# Variational implicit quantum kernel methods scale much worse than explicit heads.
# Therefore the default subset is smaller than the explicit-QNN version.
## TRAIN_SUBSET_SIZE = 1000
TRAIN_SUBSET_SIZE = 8000
## TEST_SUBSET_SIZE = 300
TEST_SUBSET_SIZE = 1000
BATCH_SIZE = 64

# Baseline settings
## MLP_EPOCHS = 20
MLP_EPOCHS = 20
MLP_LR = 0.001

# Variational implicit model settings
REDUCED_DIM = 6
N_QUBITS = 6
N_LAYERS = 2
## ALIGN_EPOCHS = 8
ALIGN_EPOCHS = 20
ALIGN_LR = 0.003

# ==========================================
# 2. 資料預處理 (CIFAR-10)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(Subset(trainset, range(TRAIN_SUBSET_SIZE)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(Subset(testset, range(TEST_SUBSET_SIZE)), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. 固定 CNN Backbone (架構固定)
# ==========================================
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=3)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.cnn1(x)))
        x = self.pool(self.relu(self.cnn2(x)))
        x = self.pool(self.relu(self.cnn3(x)))
        return x.view(x.size(0), -1)

class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))

# ==========================================
# 4. Variational Implicit Quantum Model
# ==========================================
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def variational_feature_state(x, weights):
    # Data encoding
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)

    # Variational feature map
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.state()

class VariationalImplicitQM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.reducer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, REDUCED_DIM),
            nn.BatchNorm1d(REDUCED_DIM),
            nn.Tanh()
        )
        self.q_params = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.05)

    def reduced_features(self, x):
        z = self.backbone(x)
        z = self.reducer(z) * np.pi
        return z

    def batch_states(self, z_batch):
        states = []
        for z in z_batch:
            states.append(variational_feature_state(z, self.q_params))
        return torch.stack(states)

# ==========================================
# 5. 輔助函式
# ==========================================
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def save_history_plots(history, prefix):
    epochs = history['epoch']

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{prefix} Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower()}_loss_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['test_acc'], label='Test Accuracy')
    plt.axhline(y=40, color='r', linestyle='--', label='40% Target')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{prefix} Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower()}_accuracy_curve.png", dpi=220)
    plt.close()

def train_baseline(model, train_loader, test_loader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100. * correct / total
        test_acc = evaluate_test_accuracy(model, test_loader)

        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs}: Loss {total_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")

    duration = time.time() - start
    return history, history['test_acc'][-1], duration

def pairwise_kernel_from_states(states_a, states_b):
    # states are complex tensors shaped [N, 2**n]
    overlap = torch.matmul(torch.conj(states_a), states_b.T)
    kernel = torch.abs(overlap) ** 2
    return kernel.real

def train_variational_implicit(model, train_loader, test_loader, epochs=8, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}
    start = time.time()

    # Alignment-style training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            z = model.reduced_features(x)
            states = model.batch_states(z)

            K = pairwise_kernel_from_states(states, states)
            target = (y.unsqueeze(1) == y.unsqueeze(0)).float()

            # simple kernel-target alignment loss
            loss = ((K - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # For logging, fit SVM with current kernel parameters
        train_acc, test_acc = evaluate_variational_implicit_accuracy(model, train_loader, test_loader)
        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs}: Loss {total_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")

    duration = time.time() - start
    return history, history['test_acc'][-1], duration

def collect_reduced_features_and_labels(model, loader):
    model.eval()
    all_z = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.reduced_features(x)
            all_z.append(z.cpu())
            all_y.append(y)
    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)

def collect_states(model, z_tensor):
    states = []
    for zi in z_tensor:
        zi_dev = zi.to(device)
        st = variational_feature_state(zi_dev, model.q_params)
        states.append(st.detach().cpu())
    return torch.stack(states)

def evaluate_variational_implicit_accuracy(model, train_loader, test_loader):
    z_train, y_train = collect_reduced_features_and_labels(model, train_loader)
    z_test, y_test = collect_reduced_features_and_labels(model, test_loader)

    states_train = collect_states(model, z_train)
    states_test = collect_states(model, z_test)

    K_train = pairwise_kernel_from_states(states_train, states_train).numpy()
    K_test = pairwise_kernel_from_states(states_test, states_train).numpy()

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train.numpy())

    train_pred = clf.predict(K_train)
    test_pred = clf.predict(K_test)

    train_acc = 100. * (train_pred == y_train.numpy()).mean()
    test_acc = 100. * (test_pred == y_test.numpy()).mean()
    return train_acc, test_acc

# ==========================================
# 6. 執行
# ==========================================
print("\n--- 訓練古典 Baseline (CNN+MLP) ---")
mlp_model = CNN_MLP().to(device)
mlp_hist, mlp_test_acc, mlp_time = train_baseline(
    mlp_model, train_loader, test_loader, epochs=MLP_EPOCHS, lr=MLP_LR
)
save_history_plots(mlp_hist, "CNN_MLP")

print("\n--- 訓練 Variational Implicit Quantum Model ---")
vim_model = VariationalImplicitQM().to(device)
vim_hist, vim_test_acc, vim_time = train_variational_implicit(
    vim_model, train_loader, test_loader, epochs=ALIGN_EPOCHS, lr=ALIGN_LR
)
save_history_plots(vim_hist, "VIM_QNN")

comparison_df = pd.DataFrame([
    {
        "model_name": "CNN_MLP",
        "test_accuracy_percent": mlp_test_acc,
        "trainable_parameters": count_trainable_parameters(mlp_model),
        "training_time_sec": mlp_time,
    },
    {
        "model_name": "Variational_Implicit_QM",
        "test_accuracy_percent": vim_test_acc,
        "trainable_parameters": count_trainable_parameters(vim_model),
        "training_time_sec": vim_time,
    },
])
comparison_df.to_csv(OUTPUT_DIR / "problem3_comparison_table.csv", index=False)
pd.DataFrame(mlp_hist).to_csv(OUTPUT_DIR / "cnn_mlp_history.csv", index=False)
pd.DataFrame(vim_hist).to_csv(OUTPUT_DIR / "vim_qnn_history.csv", index=False)

print("\n" + "=" * 72)
print(f"{'模型':<25} | {'測試準確度 (%)':<15} | {'參數量':<12} | {'耗時 (秒)':<10}")
print("-" * 72)
print(f"{'CNN+MLP':<25} | {mlp_test_acc:<15.2f} | {count_trainable_parameters(mlp_model):<12} | {mlp_time:<10.2f}")
print(f"{'Variational_Implicit_QM':<25} | {vim_test_acc:<15.2f} | {count_trainable_parameters(vim_model):<12} | {vim_time:<10.2f}")
print(f"\n已輸出至資料夾: {OUTPUT_DIR.resolve()}")

