# Multi-Lead-ECG-for-Early-Cardiac-Disease-Detection
AI-Based Reconstruction and Classification  of Multi-Lead ECG Signals from Reduced Lead Data for Early Cardiac Disease  Detection
# **Installations & Imports**


```python
# !pip install --force-reinstall pandas==2.1.4 wfdb --no-cache-dir

import os
import ast
import re
import random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wfdb
from scipy.signal import resample

import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
```

# **Device & Config**


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class Config:
    seed = 2021
    device = device

    ptbxl_subfolder = '/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    chapman_wfdb_path = '/kaggle/input/chapmanshaoxing-12lead-ecg-database/WFDB_ChapmanShaoxing/'

    target_fs = 500
    target_length = 5000
    num_leads = 12
    num_classes = 5

    batch_size = 64
    num_workers = 4
    learning_rate = 1e-3
    weight_decay = 1e-5
    epochs = 50
    patience = 10
    grad_clip = 1.0

    scheduler_patience = 5
    scheduler_factor = 0.5

    output_dir = '/kaggle/working/'
    model_save_path = os.path.join(output_dir, 'best_cnn_ecg.pth')
```

# **Seed**


```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(Config.seed)
```

# **Re-sample**


```python
def resample_signal(signal, original_length, original_fs):
    if original_fs == Config.target_fs and len(signal) == Config.target_length:
        return signal

    resampled = resample(signal, int(original_length * (Config.target_fs / original_fs)))

    if len(resampled) > Config.target_length:
        return resampled[:Config.target_length]
    elif len(resampled) < Config.target_length:
        return np.pad(resampled, (0, Config.target_length - len(resampled)), mode='constant')

    return resampled
```

# **PTB-XL Label Map**


```python
def map_ptbxl_label(scp_str):
    scp = ast.literal_eval(scp_str) if isinstance(scp_str, str) else {}

    if 'NORM' in scp:
        return 0

    arrhythmia_keys = {'AFIB','AFL','SVT','SVTAC','AT','VT','TACHY'}
    block_keys = {'LBBB','RBBB','AVB','1AVB','2AVB','3AVB'}
    hyp_keys = {'LVH','RVH','HYP'}
    mi_keys = {'MI','IMI','AMI','ALMI','ISC','ISC_'}

    if any(k in scp for k in arrhythmia_keys): return 1
    if any(k in scp for k in block_keys): return 2
    if any(k in scp for k in hyp_keys): return 3
    if any(k in scp for k in mi_keys): return 4

    return 4
```

# **CNN Model**


```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)

        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.norm3 = nn.BatchNorm1d(hidden_size)

        self.pool = nn.MaxPool1d(2)
        self.swish = Swish()

    def forward(self, x):
        conv1 = self.conv_1(x)
        x = self.swish(self.norm1(conv1))
        x = F.pad(x, (self.kernel_size - 1, 0))

        x = self.swish(self.norm2(self.conv_2(x)))
        x = F.pad(x, (self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.swish(self.norm3(conv1 + conv3))
        x = F.pad(x, (self.kernel_size - 1, 0))

        return self.pool(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvNormPool(12, 256, 5)
        self.conv2 = ConvNormPool(256, 128, 5)
        self.conv3 = ConvNormPool(128, 64, 5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, Config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

# **PTB-XL Loading**


```python
print("Loading PTB-XL...")

ptbxl_meta = pd.read_csv(os.path.join(Config.ptbxl_subfolder, 'ptbxl_database.csv'))
ptbxl_meta['label'] = ptbxl_meta['scp_codes'].apply(map_ptbxl_label)

ptbxl_signals = []
ptbxl_labels = []

for _, row in ptbxl_meta.iterrows():
    try:
        record_path = os.path.join(Config.ptbxl_subfolder, row['filename_hr'])
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.T.astype(np.float32)

        if signal.shape[1] != Config.target_length:
            signal = np.apply_along_axis(
                resample_signal, 1, signal, signal.shape[1], record.fs
            )

        ptbxl_signals.append(signal)
        ptbxl_labels.append(row['label'])

    except Exception:
        continue

ptbxl_signals = np.array(ptbxl_signals)
ptbxl_labels = np.array(ptbxl_labels)

print("PTB-XL loaded:", len(ptbxl_labels))
print("PTB-XL label distribution:", Counter(ptbxl_labels))
```

# **Chapman DX Extraction**


```python
DX_RE = re.compile(r'^\s*#?\s*Dx\s*[:=]\s*(.*)$', re.IGNORECASE)

def extract_dx_codes_from_header(header):
    for c in getattr(header, "comments", []) or []:
        m = DX_RE.match(str(c).strip())
        if m:
            return set(re.findall(r"\d{5,18}", m.group(1)))
    return set()

def extract_dx_codes_from_hea(hea_path):
    if not os.path.exists(hea_path):
        return set()
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = DX_RE.match(ln.strip())
            if m:
                return set(re.findall(r"\d{5,18}", m.group(1)))
    return set()
```

# **Chapman Taxonomy**


```python
NORMAL_CODES = {'426783006'}

ARRHYTHMIA_CODES = {
    '164889003','164890007','713427006',
    '426761007','17338001','284470004'
}

SINUS_VARIANT_CODES = {
    '426177001','427084000','427393009'
}

BLOCK_CODES = {
    '59118001','164909002','270492004',
    '713426002','445118002','39732003',
    '27885002','6374002','698252002'
}

HYP_CODES = {
    '55827005','164873001','89792004',
    '446358003','67741000119109'
}

STT_MI_CODES = {
    '64934002','164931005','429622005',
    '428750005','22298006','164865005',
    '57054005','426396005','413444003',
    '164867002'
}

OTHER_ABNORMAL_CODES = {
    '164917005','251146004','251199005','47665007'
}

def map_chapman_label(codes, sinus_as_normal=True):

    if not codes:
        return 0

    if codes & STT_MI_CODES:
        return 4
    if codes & BLOCK_CODES:
        return 2
    if codes & HYP_CODES:
        return 3
    if codes & ARRHYTHMIA_CODES:
        return 1

    if codes & SINUS_VARIANT_CODES:
        return 0 if sinus_as_normal else 4

    if codes & NORMAL_CODES:
        return 0

    if codes & OTHER_ABNORMAL_CODES:
        return 4

    return 4
```

# **Chapman Loading**


```python
print("Loading Chapman dataset...")

chapman_files = [
    f[:-4] for f in os.listdir(Config.chapman_wfdb_path)
    if f.endswith('.hea')
]

chapman_signals = []
chapman_labels = []

for file_name in chapman_files:
    try:
        full_path = os.path.join(Config.chapman_wfdb_path, file_name)

        header = wfdb.rdheader(full_path)
        codes = extract_dx_codes_from_header(header)

        if not codes:
            codes = extract_dx_codes_from_hea(full_path + ".hea")

        label = map_chapman_label(codes, sinus_as_normal=True)

        record = wfdb.rdrecord(full_path)
        signal = record.p_signal.T.astype(np.float32)

        if signal.shape[1] != Config.target_length:
            signal = np.apply_along_axis(
                resample_signal, 1, signal, signal.shape[1], record.fs
            )

        chapman_signals.append(signal)
        chapman_labels.append(label)

    except Exception:
        continue

chapman_signals = np.array(chapman_signals)
chapman_labels = np.array(chapman_labels)

print("Chapman loaded:", len(chapman_labels))
print("Chapman label distribution:", Counter(chapman_labels))
```

# **Train/Val Split**


```python
X_train_val, y_train_val = ptbxl_signals, ptbxl_labels
X_test, y_test = chapman_signals, chapman_labels

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2,
    stratify=y_train_val,
    random_state=Config.seed
)
```

# **Normalization**


```python
def normalize_signals(signals):
    mu = signals.mean(axis=2, keepdims=True)
    std = signals.std(axis=2, keepdims=True) + 1e-6
    return (signals - mu) / std

X_train = normalize_signals(X_train)
X_val   = normalize_signals(X_val)
X_test  = normalize_signals(X_test)

print("Shapes:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)
```

# **Dataset & Data Loader**


```python
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

train_dataset = ECGDataset(X_train, y_train)
val_dataset   = ECGDataset(X_val, y_val)
test_dataset  = ECGDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
    pin_memory=True
)
```

# **Meter Class**


```python
class Meter:
    def __init__(self, n_classes=Config.num_classes):
        self.n_classes = n_classes
        self.reset()
        self.confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)

    def reset(self):
        self.metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        self.total_samples = 0
        self.count = 0

    def update(self, preds, targets, loss):
        batch_size = targets.size(0)
        self.total_samples += batch_size
        self.metrics["loss"] += loss.item() * batch_size

        preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()

        self.metrics["accuracy"] += accuracy_score(targets_np, preds_argmax)
        self.metrics["f1"] += f1_score(targets_np, preds_argmax, average="macro", zero_division=0)
        self.metrics["precision"] += precision_score(targets_np, preds_argmax, average="macro", zero_division=0)
        self.metrics["recall"] += recall_score(targets_np, preds_argmax, average="macro", zero_division=0)

        for t, p in zip(targets_np, preds_argmax):
            self.confusion[t, p] += 1

        self.count += 1

    def compute(self):
        return {
            "loss": self.metrics["loss"] / max(1, self.total_samples),
            "accuracy": self.metrics["accuracy"] / max(1, self.count),
            "f1": self.metrics["f1"] / max(1, self.count),
            "precision": self.metrics["precision"] / max(1, self.count),
            "recall": self.metrics["recall"] / max(1, self.count),
        }
```

# **Model Setup**


```python
model = CNN().to(Config.device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    model.parameters(),
    lr=Config.learning_rate,
    weight_decay=Config.weight_decay
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=Config.scheduler_factor,
    patience=Config.scheduler_patience
)

print("Model initialized.")
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
```

# **Training Loop**


```python
train_logs = []
val_logs = []

best_val_f1 = 0.0
best_epoch = -1
epochs_no_improve = 0

print("Starting training...\n")

for epoch in range(Config.epochs):

    print(f"Epoch {epoch+1}/{Config.epochs}")

    # ------------------
    # TRAIN
    # ------------------
    model.train()
    train_meter = Meter()

    for signals, labels in train_loader:
        signals = signals.to(Config.device)
        labels = labels.to(Config.device)

        optimizer.zero_grad()

        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        optimizer.step()

        train_meter.update(outputs, labels, loss)

    train_metrics = train_meter.compute()
    train_logs.append(train_metrics)

    # ------------------
    # VALIDATION
    # ------------------
    model.eval()
    val_meter = Meter()

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            val_meter.update(outputs, labels, loss)

    val_metrics = val_meter.compute()
    val_logs.append(val_metrics)

    print(
        f"Train - Loss: {train_metrics['loss']:.4f} | "
        f"Acc: {train_metrics['accuracy']:.4f} | "
        f"F1: {train_metrics['f1']:.4f}"
    )

    print(
        f"Val   - Loss: {val_metrics['loss']:.4f} | "
        f"Acc: {val_metrics['accuracy']:.4f} | "
        f"F1: {val_metrics['f1']:.4f}"
    )

    scheduler.step(val_metrics["f1"])

    # Save best model
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), Config.model_save_path)
        print("New best model saved.\n")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= Config.patience:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")
print("Best epoch:", best_epoch)
```

# **Test Evaluation**


```python
model.load_state_dict(torch.load(Config.model_save_path))
model.eval()

test_meter = Meter()

with torch.no_grad():
    for signals, labels in test_loader:
        signals = signals.to(Config.device)
        labels = labels.to(Config.device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        test_meter.update(outputs, labels, loss)

test_metrics = test_meter.compute()

print("\n===== FINAL CHAPMAN TEST RESULTS =====")
print("Loss:", test_metrics["loss"])
print("Accuracy:", test_metrics["accuracy"])
print("F1 macro:", test_metrics["f1"])
print("Precision macro:", test_metrics["precision"])
print("Recall macro:", test_metrics["recall"])
```

# **Confusion Matrix**


```python
plt.figure(figsize=(6,5))
sns.heatmap(
    test_meter.confusion.numpy(),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=['NORM','Arrhythmia','Block','Hypertrophy','MI/Abnormal'],
    yticklabels=['NORM','Arrhythmia','Block','Hypertrophy','MI/Abnormal']
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Chapman Test Confusion Matrix")
plt.show()
```

# **Training Curves**


```python
train_df = pd.DataFrame(train_logs)
val_df = pd.DataFrame(val_logs)

fig, axes = plt.subplots(1, 3, figsize=(18,5))

axes[0].plot(train_df["loss"], label="Train")
axes[0].plot(val_df["loss"], label="Val")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(train_df["accuracy"], label="Train")
axes[1].plot(val_df["accuracy"], label="Val")
axes[1].set_title("Accuracy")
axes[1].legend()

axes[2].plot(train_df["f1"], label="Train")
axes[2].plot(val_df["f1"], label="Val")
axes[2].set_title("F1 Macro")
axes[2].legend()

plt.show()
```

# **Save Logs**


```python
logs = pd.concat(
    [train_df.add_prefix("train_"),
     val_df.add_prefix("val_")],
    axis=1
)

logs.to_csv("/kaggle/working/training_logs_cnn.csv", index=False)

print("Logs saved.")
print("Best model saved at:", Config.model_save_path)
```
