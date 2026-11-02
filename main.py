"""
Project 38: SimCLR Self-Supervised Contrastive Learning on CIFAR-10
===================================================================
Description:
End-to-end implementation of the SimCLR framework (Chen et al., ICML 2020) for
self-supervised visual representation learning without human annotations.
Employs dual stochastic augmentation views, a residual convolutional encoder (MiniResNet),
a non-linear MLP projection head, vectorized Normalized Temperature-scaled Cross Entropy
(NT-Xent) loss, downstream linear probe verification, and 2D representation manifold analysis.
"""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 5
PROBE_EPOCHS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 2000
NUM_VAL_SAMPLES = 1000
TEMPERATURE = 0.5
REPRESENTATION_DIM = 256
PROJECTION_DIM = 64

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Stochastic Multi-View Data Augmentation Module
# -----------------------------------------------------------------------------
class SimCLRAugmentation:
    """
    Applies two independent stochastic augmentations to an input image,
    generating correlated views (x_i, x_j) for contrastive pairing.
    """
    def __init__(self, size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

    def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


class SimCLRDataset(Dataset):
    """Wraps a PyTorch dataset to return paired stochastic views without labels."""
    def __init__(self, base_dataset: Dataset, transform: SimCLRAugmentation):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_img, _ = self.base_dataset[idx]
        v1, v2 = self.transform(raw_img)
        return v1, v2


# -----------------------------------------------------------------------------
# Base Encoder & Non-Linear Projection Head Architecture
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class BaseEncoder(nn.Module):
    """
    Base Convolutional Encoder f(.) extracting high-level semantic representation h in R^d.
    """
    def __init__(self, representation_dim: int = REPRESENTATION_DIM):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(32, 64, stride=2)   # 32x32 -> 16x16
        self.stage2 = ResidualBlock(64, 128, stride=2)  # 16x16 -> 8x8
        self.stage3 = ResidualBlock(128, representation_dim, stride=2) # 8x8 -> 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        return torch.flatten(out, 1)


class SimCLRModel(nn.Module):
    """
    Full SimCLR framework combining base encoder f(.) and non-linear MLP projection head g(.).
    Maps image x -> representation h -> normalized projection z.
    """
    def __init__(
        self,
        representation_dim: int = REPRESENTATION_DIM,
        projection_dim: int = PROJECTION_DIM,
    ):
        super().__init__()
        self.encoder = BaseEncoder(representation_dim)

        # Non-linear projection head g(.): MLP with one hidden layer
        self.projection_head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.BatchNorm1d(representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(representation_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projection_head(h)
        # Project onto the unit hypersphere S^(k-1)
        z = F.normalize(z, dim=1)
        return h, z


# -----------------------------------------------------------------------------
# Vectorized NT-Xent Loss Function
# -----------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent):
    Maximizes cosine similarity between positive pairs while pushing away
    all other 2*(N-1) augmented views in the mini-batch.
    """
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        # Concatenate 2N views: [2N, D]
        z = torch.cat([z_i, z_j], dim=0)

        # Full cosine similarity matrix [2N, 2N]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity along the main diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)

        # Ground truth targets for positive pairing:
        # For view i in [0, N-1], positive match is i + N
        # For view i in [N, 2N-1], positive match is i - N
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        return self.criterion(sim_matrix, targets)


# -----------------------------------------------------------------------------
# Downstream Linear Evaluation Protocol (Linear Probing)
# -----------------------------------------------------------------------------
def train_linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = PROBE_EPOCHS,
) -> List[float]:
    """
    Trains a linear classification layer on top of frozen encoder representations
    to rigorously measure representation quality without fine-tuning backbone weights.
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(REPRESENTATION_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    acc_history = []

    for ep in range(1, epochs + 1):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = encoder(x)
            logits = classifier(feat)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                feat = encoder(x)
                preds = classifier(feat).argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

        acc = (correct / total) * 100.0
        acc_history.append(acc)

    return acc_history


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_simclr_dashboard(
    pretrain_loss: List[float],
    probe_acc_simclr: List[float],
    probe_acc_random: List[float],
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    sample_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    save_path: Path,
) -> None:
    """Renders a 4-panel publication-grade evaluation figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Multi-View Stochastic Augmentation Samples
    ax_aug = axes[0, 0]
    ax_aug.axis("off")
    ax_aug.set_title("(a) SimCLR Correlated Stochastic Views ($x_i, x_j$)", fontsize=13, fontweight="bold", pad=10)

    sub_spec = axes[0, 0].get_subplotspec()
    inner_grid = fig.add_subfigure(sub_spec)
    sub_axes = inner_grid.subplots(2, 4)

    inv_mean = np.array([0.4914, 0.4822, 0.4465])
    inv_std = np.array([0.2023, 0.1994, 0.2010])

    for col in range(min(4, len(sample_pairs))):
        v1, v2 = sample_pairs[col]
        img1 = (v1.permute(1, 2, 0).numpy() * inv_std + inv_mean).clip(0, 1)
        img2 = (v2.permute(1, 2, 0).numpy() * inv_std + inv_mean).clip(0, 1)

        sub_axes[0, col].imshow(img1)
        sub_axes[0, col].set_title(f"View A #{col+1}", fontsize=8.5, fontweight="semibold")
        sub_axes[0, col].axis("off")

        sub_axes[1, col].imshow(img2)
        sub_axes[1, col].set_title(f"View B #{col+1}", fontsize=8.5, fontweight="semibold")
        sub_axes[1, col].axis("off")

    # 2. Self-Supervised NT-Xent Loss Trajectory
    ax_loss = axes[0, 1]
    epochs = range(1, len(pretrain_loss) + 1)
    ax_loss.plot(epochs, pretrain_loss, "o-", color="#1f77b4", linewidth=2.2, label="NT-Xent Loss")
    ax_loss.set_title(f"(b) Self-Supervised Pretraining Loss ($\\tau={TEMPERATURE}$)", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Pretraining Epoch", fontsize=11)
    ax_loss.set_ylabel("NT-Xent Loss", fontsize=11)
    ax_loss.set_xticks(list(epochs))
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 3. Downstream Linear Probe Accuracy (SimCLR vs Random Baseline)
    ax_probe = axes[1, 0]
    probe_epochs = range(1, len(probe_acc_simclr) + 1)
    ax_probe.plot(probe_epochs, probe_acc_simclr, "s-", color="#2ca02c", linewidth=2.2, label="SimCLR Pretrained Encoder")
    ax_probe.plot(probe_epochs, probe_acc_random, "x--", color="#d62728", linewidth=2.0, label="Random Untrained Encoder")
    ax_probe.set_title("(c) Linear Evaluation Protocol (Frozen Representations)", fontsize=13, fontweight="bold", pad=10)
    ax_probe.set_xlabel("Probe Training Epoch", fontsize=11)
    ax_probe.set_ylabel("Downstream Top-1 Accuracy (%)", fontsize=11)
    ax_probe.set_xticks(list(probe_epochs))
    ax_probe.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_probe.grid(True, linestyle=":", alpha=0.6)

    # 4. 2D Representation Space PCA Projection (Unsupervised Clustering)
    ax_pca = axes[1, 1]
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    reduced_2d = pca.fit_transform(test_embeddings[:500])
    sub_targets = test_targets[:500]

    scatter = ax_pca.scatter(
        reduced_2d[:, 0], reduced_2d[:, 1],
        c=sub_targets, cmap="tab10", alpha=0.75, s=22, edgecolors="none"
    )
    ax_pca.set_title("(d) 2D PCA of Learned Feature Space (Unsupervised)", fontsize=13, fontweight="bold", pad=10)
    ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)", fontsize=11)
    ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)", fontsize=11)
    ax_pca.grid(True, linestyle=":", alpha=0.6)

    cbar = fig.colorbar(scatter, ax=ax_pca, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels(CLASSES, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved SimCLR diagnostic dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 38: SimCLR Self-Supervised Contrastive Learning on CIFAR-10")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Base Encoder          : Residual CNN (MiniResNet, 256-D representations)")
    print(f"Projection Head       : 2-Layer Non-Linear MLP (64-D unit hypersphere)")
    print(f"Loss Function         : NT-Xent (Normalized Temp-scaled Cross Entropy, tau={TEMPERATURE})")
    print(f"Pretraining Samples   : {NUM_TRAIN_SAMPLES} Unlabeled Images ({PRETRAIN_EPOCHS} epochs)")
    print(f"Evaluation Protocol   : Downstream Linear Probe on Frozen Encoder")
    print("-" * 75)

    # 1. Dataset Construction
    raw_train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False)
    raw_test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False)

    train_subset = Subset(raw_train, list(range(min(NUM_TRAIN_SAMPLES, len(raw_train)))))
    test_subset = Subset(raw_test, list(range(min(NUM_VAL_SAMPLES, len(raw_test)))))

    # Self-supervised contrastive dataset
    simclr_dataset = SimCLRDataset(train_subset, SimCLRAugmentation(size=32))
    simclr_loader = DataLoader(simclr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Standard downstream evaluation datasets
    std_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    eval_train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=std_transform)
    eval_test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=std_transform)

    eval_train_loader = DataLoader(
        Subset(eval_train, list(range(min(NUM_TRAIN_SAMPLES, len(eval_train))))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    eval_test_loader = DataLoader(
        Subset(eval_test, list(range(min(NUM_VAL_SAMPLES, len(eval_test))))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 2. SimCLR Architecture & Optimization
    model = SimCLRModel(REPRESENTATION_DIM, PROJECTION_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    print("-" * 75)

    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-5)

    # 3. Self-Supervised Pretraining Loop
    pretrain_losses = []
    sample_pairs = []
    start_time = time.perf_counter()

    print(">>> Commencing Self-Supervised Contrastive Pretraining...")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (v1, v2) in enumerate(simclr_loader):
            if epoch == 1 and batch_idx == 0:
                for i in range(min(4, v1.size(0))):
                    sample_pairs.append((v1[i].cpu(), v2[i].cpu()))

            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)
            _, z2 = model(v2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * v1.size(0)
            total_samples += v1.size(0)

        scheduler.step()
        ep_loss = running_loss / total_samples
        pretrain_losses.append(ep_loss)
        ep_duration = time.perf_counter() - ep_start
        print(f"Pretrain Epoch [{epoch:02d}/{PRETRAIN_EPOCHS:02d}] | NT-Xent Loss: {ep_loss:.4f} | Time: {ep_duration:.2f}s")

    pretrain_time = time.perf_counter() - start_time
    print(f"Self-Supervised Pretraining Finished in {pretrain_time:.2f}s.")
    print("-" * 75)

    # 4. Downstream Linear Evaluation Protocol
    print(">>> Evaluating Learned Representations via Downstream Linear Probing...")
    # A. Probe on SimCLR Pretrained Encoder
    probe_simclr = train_linear_probe(
        model.encoder, eval_train_loader, eval_test_loader, device, epochs=PROBE_EPOCHS
    )
    print(f"    SimCLR Encoder Linear Probe Top-1 Acc : {probe_simclr[-1]:.2f}%")

    # B. Probe on Randomly Initialized Encoder (Baseline)
    random_encoder = BaseEncoder(REPRESENTATION_DIM).to(device)
    probe_random = train_linear_probe(
        random_encoder, eval_train_loader, eval_test_loader, device, epochs=PROBE_EPOCHS
    )
    print(f"    Random Baseline Encoder Top-1 Acc     : {probe_random[-1]:.2f}%")
    print(f"    Self-Supervised Representation Gain   : +{probe_simclr[-1] - probe_random[-1]:.2f}%")
    print("-" * 75)

    # 5. Extract Embeddings for Manifold Analysis
    model.encoder.eval()
    all_embs = []
    all_targets = []
    with torch.no_grad():
        for x, y in eval_test_loader:
            x = x.to(device)
            feat = model.encoder(x)
            all_embs.append(feat.cpu().numpy())
            all_targets.append(y.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 6. Save Model Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "simclr_encoder.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "projection_head_state_dict": model.projection_head.state_dict(),
            "final_nt_xent_loss": pretrain_losses[-1],
            "probe_accuracy": probe_simclr[-1],
        },
        checkpoint_path,
    )
    print(f"[Checkpoint] Saved pretrained SimCLR encoder to: {checkpoint_path}")

    # 7. Generate Publication Dashboard
    dashboard_path = PROJECT_DIR / "simclr_contrastive_learning_result.png"
    plot_simclr_dashboard(
        pretrain_loss=pretrain_losses,
        probe_acc_simclr=probe_simclr,
        probe_acc_random=probe_random,
        test_embeddings=all_embs,
        test_targets=all_targets,
        sample_pairs=sample_pairs,
        save_path=dashboard_path,
    )

    print("=" * 75)
    print("SimCLR Self-Supervised Learning Experiment Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

