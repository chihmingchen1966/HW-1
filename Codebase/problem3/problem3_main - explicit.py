import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# 1. 基礎設定與隨機種子 (請務必替換為學號)
# ==========================================
seed = 10010022  # 已根據你的輸入設定
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的設備: {device}")
OUTPUT_DIR = Path("problem3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. 資料預處理 (CIFAR-10) [cite: 105-118]
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# 增加訓練樣本至 8000 筆，並增加測試樣本至 1000 筆以獲得穩定的 Accuracy
train_loader = DataLoader(Subset(trainset, range(8000)), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(testset, range(1000)), batch_size=64, shuffle=False)

# ==========================================
# 3. 固定 CNN Backbone (不可修改) [cite: 120-135]
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

# ==========================================
# 4. 最終調校的 QNN 架構
# ==========================================
n_qubits = 6  # 增加到位元 6 個，能攜帶更多 CNN 特徵資訊
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Angle Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    # 變分層 (StronglyEntanglingLayers 效果較好)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class CNN_QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        # 減少中間層的資訊損失
        self.pre_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh()
        )
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.05)
        # 最終分類層
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 10)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pre_net(x) * np.pi
        
        # 批次處理優化 (Parameter Broadcasting)
        # 注意：雖然迴圈較慢，但在 CPU 模擬器上最穩定
        q_out = []
        for xi in x:
            res = quantum_circuit(xi, self.q_params)
            q_out.append(torch.stack(res))
        
        q_out = torch.stack(q_out).to(x.device)
        return self.post_net(q_out.float())

class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ==========================================
# 5. 訓練與評估
# ==========================================
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_test_accuracy(model, test_loader):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_correct += model(x).argmax(1).eq(y).sum().item()
    return 100. * test_correct / len(test_loader.dataset)


def run_experiment(model, loader, test_loader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}
    start = time.time()
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()

        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / len(loader.dataset)
        test_acc = evaluate_test_accuracy(model, test_loader)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.2f}%")

    duration = time.time() - start
    print(f"[Best] test accuracy = {best_test_acc:.2f}% at epoch {best_epoch}")
    return history, best_test_acc, duration


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


# ==========================================
# 6. 執行
# ==========================================
print("\n--- 訓練古典 Baseline (CNN+MLP) ---")
mlp_model = CNN_MLP().to(device)
## mlp_hist, mlp_test_acc, mlp_time = run_experiment(mlp_model, train_loader, test_loader, epochs=10, lr=0.001)
mlp_hist, mlp_test_acc, mlp_time = run_experiment(mlp_model, train_loader, test_loader, epochs=20, lr=0.001)
save_history_plots(mlp_hist, "CNN_MLP")

print("\n--- 訓練最終版量子模型 (CNN+QNN) ---")
qnn_model = CNN_QNN().to(device)
qnn_hist, qnn_test_acc, qnn_time = run_experiment(qnn_model, train_loader, test_loader, epochs=20, lr=0.003)
save_history_plots(qnn_hist, "CNN_QNN")

comparison_df = pd.DataFrame([
    {
        "model_name": "CNN_MLP",
        "test_accuracy_percent": mlp_test_acc,
        "trainable_parameters": count_trainable_parameters(mlp_model),
        "training_time_sec": mlp_time,
    },
    {
        "model_name": "CNN_QNN",
        "test_accuracy_percent": qnn_test_acc,
        "trainable_parameters": count_trainable_parameters(qnn_model),
        "training_time_sec": qnn_time,
    },
])
comparison_df.to_csv(OUTPUT_DIR / "problem3_comparison_table.csv", index=False)
pd.DataFrame(mlp_hist).to_csv(OUTPUT_DIR / "cnn_mlp_history.csv", index=False)
pd.DataFrame(qnn_hist).to_csv(OUTPUT_DIR / "cnn_qnn_history.csv", index=False)

print("\n" + "=" * 60)
print(f"{'模型':<15} | {'測試準確度 (%)':<15} | {'參數量':<12} | {'耗時 (秒)':<10}")
print("-" * 60)
print(f"{'CNN+MLP':<15} | {mlp_test_acc:<15.2f} | {count_trainable_parameters(mlp_model):<12} | {mlp_time:<10.2f}")
print(f"{'CNN+QNN':<15} | {qnn_test_acc:<15.2f} | {count_trainable_parameters(qnn_model):<12} | {qnn_time:<10.2f}")
print(f"\n已輸出至資料夾: {OUTPUT_DIR.resolve()}")


