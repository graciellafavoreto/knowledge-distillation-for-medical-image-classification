"""Funções compartilhadas pelos notebooks: dados, modelos, treino, métricas e gráficos."""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from medmnist import PneumoniaMNIST
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             matthews_corrcoef, confusion_matrix)

DATA_ROOT = "./data"   # mesmo cache para todos os notebooks -> 1 download só
IMG_SIZE = 128         # medmnist suporta 28/64/128/224
NUM_CLASSES = 2


# 1- Reprodutibilidade e dados

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Fixa as sementes"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def balance_dataset(dataset, seed=42):
    """Undersampling: reduz a classe majoritaria até igualar a minoritaria
    Seed fixa >> todos os notebooks obtem o mesmo subconjuntoe"""
    rng = np.random.default_rng(seed)
    labels = dataset.labels.flatten()   # le rotulo, mas n carrega img

    idx_normal = np.where(labels == 0)[0]
    idx_pneumonia = np.where(labels == 1)[0]
    n = min(len(idx_normal), len(idx_pneumonia))

    indices = np.concatenate([
        rng.choice(idx_normal, n, replace=False),
        rng.choice(idx_pneumonia, n, replace=False),
    ])
    rng.shuffle(indices)
    return Subset(dataset, indices)


def get_labels(dataset):
    """Rotulos sem aplicar transform nas imagens"""
    if isinstance(dataset, Subset):
        return dataset.dataset.labels.flatten()[dataset.indices]
    return dataset.labels.flatten()

def get_dataloaders(splits=("train", "val", "test"), batch_size=32, size=IMG_SIZE,
                    num_workers=0, balance=("train", "val", "test")):
    """
    Único ponto do curso que instancia o dataset.
    balance: splits que passam por undersampling (50/50). Os demais mantêm a distribuição original. Retorna {"train": loader, "val": loader e "test": loader}
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    os.makedirs(DATA_ROOT, exist_ok=True)
    loaders = {}

    for split in splits:
        dataset = PneumoniaMNIST(
          split=split, 
          transform=transform, 
          root=DATA_ROOT,
          size=size, 
          download=True)

        if split in balance:
            dataset = balance_dataset(dataset)

        loaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(split == "train"),
            num_workers=num_workers, 
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


# 2- Modelos

def Teacher(num_classes=NUM_CLASSES):
    """
    ResNet18 adaptada para entrada em escala de cinza (1 canal)
    e num_classes saídas.
    """
    weights = ResNet18_Weights.DEFAULT
    teacher = models.resnet18(weights=weights)

    # Entrada: imagens em escala de cinza
    teacher.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Saída: número de classes
    teacher.fc = nn.Linear(
        teacher.fc.in_features,
        num_classes
    )

    return teacher

class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

########## carrega teacher congelado para kd
def load_teacher(weights_path, device, num_classes=NUM_CLASSES):
    """
    Constrói a ResNet18 e carrega os pesos treinados.
    O teacher fica congelado para a Knowledge Distillation
    """

    teacher = Teacher(num_classes=num_classes)

    teacher.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_student(weights_path, device):
    return load_weights(SmallCNN(), weights_path, device)


# 3- Treino e avaliacao
def train_teacher_and_student_baseline(model, train_loader, val_loader, epochs, criterion, optimizer, device):

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze().long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.squeeze().long()).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    
    return history

def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs,
                                 T, alpha, optimizer, device, verbose=True):
    ce_loss = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    teacher.eval()  # teacher sempre em avaliação (BatchNorm) e sem gradiente

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            labels = labels.squeeze().long()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():               # o teacher não é atualizado
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)

            loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = validate(student, val_loader, ce_loss, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    return history

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze().long())
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels.squeeze()).sum().item()
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    Loss de KD offline: combina a loss "hard" (rótulo real) com a loss "soft" (imitar a distribuição de probabilidade do teacher))
    """

    hard_loss = F.cross_entropy(student_logits, labels)

    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)

    soft_loss = F.kl_div(
        soft_predictions,
        soft_targets,
        reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss

# 4. Inferência e métricas
@torch.inference_mode()
def get_logits(model, loader, device):
    """Retorna (rótulos, logits) de todo o loader"""
    model.eval()
    targets, logits = [], []
    for images, labels in loader:
        logits.append(model(images.to(device, non_blocking=True)).cpu())
        targets.append(labels.squeeze(1).long())
    return torch.cat(targets), torch.cat(logits)


def get_predictions(model, loader, device):
    targets, logits = get_logits(model, loader, device)
    return targets.numpy(), logits.argmax(1).numpy()


def metrics_table(y_true, preds):
    """preds: {"nome": y_pred}. Retorna DataFrame com as métricas por model"""
    rows = {
        name: {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        for name, y_pred in preds.items()
    }
    return pd.DataFrame(rows).T.round(4)


def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


@torch.inference_mode()
def measure_latency(model, device, batch_size=32, size=IMG_SIZE, runs=20):
    """Tempo médio (ms) de um forward com um batch de imagens aleatórias."""
    model.eval()
    x = torch.randn(batch_size, 1, size, size, device=device)
    for _ in range(3):                      # aquecimento
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def compare_models(models_dict, device):
    """models_dict: {"nome": modelo}. Tabela de parâmetros, tamanho e latência."""
    rows = {
        name: {
            "Parâmetros": count_parameters(m),
            "Tamanho (MB)": round(model_size_mb(m), 2),
            "Latência (ms/batch)": round(measure_latency(m.to(device), device), 2),
        }
        for name, m in models_dict.items()
    }
    return pd.DataFrame(rows).T


# 5. Gráficos
def plot_temperature(logits, temps=(1, 2, 4, 10), class_names=None):
    """Barras de softmax(logits / T) para vários T (usa um vetor de logits 1D)."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    k = len(logits)
    class_names = class_names or [f"classe {i}" for i in range(k)]

    fig, axes = plt.subplots(1, len(temps), figsize=(3.2 * len(temps), 3), sharey=True)
    for ax, T in zip(axes, temps):
        probs = F.softmax(logits / T, dim=0).numpy()
        ax.bar(class_names, probs, color="#4C78A8")
        ax.set_title(f"T = {T}")
        ax.set_ylim(0, 1)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f"{p:.2f}", ha="center")
    axes[0].set_ylabel("probabilidade")
    plt.tight_layout()
    plt.show()


def plot_prob_hist(logits, temps=(1, 4, 10), cls=1):
    """Histograma de P(classe=cls) sobre um conjunto de amostras, para vários T"""
    fig, axes = plt.subplots(1, len(temps), figsize=(3.6 * len(temps), 3), sharey=True)
    for ax, T in zip(axes, temps):
        ax.hist(F.softmax(logits / T, dim=1)[:, cls].numpy(), bins=20,
                range=(0, 1), color="#F58518")
        ax.set_title(f"T = {T}")
        ax.set_xlabel(f"P(classe {cls})")
    axes[0].set_ylabel("nº de imagens")
    plt.tight_layout()
    plt.show()


def plot_history(histories, keys=("val_loss", "val_acc")):
    """histories: {"nome": history}. Uma curva por modelo em cada métrica."""
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        for name, h in histories.items():
            ax.plot(range(1, len(h[key]) + 1), h[key], marker="o", label=name)
        ax.set_title(key)
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(y_true, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
                    cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.show()


# 6. Persistência

def save_weights(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model, path, device):
    """Carrega pesos, move para o device e deixa em modo eval."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
