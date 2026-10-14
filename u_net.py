import copy
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 128
NUM_CLASSES = 2
N_TRAIN = 4
N_VAL = 4
EPOCHS = 300
LR = 1e-3


def make_sample(size, rng):
    """Noisy image with random ellipses ("cells") and its binary mask."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.uniform(0.15, 0.85, size=2) * size
        ry, rx = rng.uniform(0.05, 0.15, size=2) * size
        angle = rng.uniform(0, np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        u = ((xx - cx) * cos + (yy - cy) * sin) / rx
        v = (-(xx - cx) * sin + (yy - cy) * cos) / ry
        mask[u ** 2 + v ** 2 <= 1] = 1
    image = 0.3 + 0.35 * mask + 0.2 * (xx / size - 0.5)
    image += rng.normal(0.0, 0.3, size=(size, size))
    return image.astype(np.float32), mask


class ShapesDataset(Dataset):
    def __init__(self, n, size, seed):
        rng = np.random.default_rng(seed)
        samples = [make_sample(size, rng) for _ in range(n)]
        self.images = torch.from_numpy(np.stack([img for img, _ in samples])).unsqueeze(1)
        self.masks = torch.from_numpy(np.stack([mask for _, mask in samples]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def check_data(dataset, num_classes):
    images, masks = dataset.images, dataset.masks
    print(f"images  {list(images.shape)}  {images.dtype}  "
          f"range [{images.min().item():.2f}, {images.max().item():.2f}]")
    print(f"masks   {list(masks.shape)}  {masks.dtype}  values {masks.unique().tolist()}")
    print(f"foreground fraction {masks.float().mean().item():.3f}")
    assert images.shape[-2:] == masks.shape[-2:], "image and mask sizes differ"
    assert masks.dtype == torch.int64, "CrossEntropyLoss expects int64 class indices"
    assert masks.min() >= 0 and masks.max() < num_classes, "mask values out of range"



class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Contracting path
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x, verbose=False):
        def show(name, t):
            if verbose:
                print(f"  {name:<12} {list(t.shape)}")

        show("input", x)
        e1 = self.enc1(x); show("enc1", e1)
        e2 = self.enc2(self.pool(e1)); show("enc2", e2)
        e3 = self.enc3(self.pool(e2)); show("enc3", e3)
        b = self.bottleneck(self.pool(e3)); show("bottleneck", b)

        u3 = self.up3(b); show("up3", u3)
        x3 = torch.cat([u3, e3], dim=1); show("concat3", x3)
        d3 = self.dec3(x3); show("dec3", d3)

        u2 = self.up2(d3); show("up2", u2)
        x2 = torch.cat([u2, e2], dim=1); show("concat2", x2)
        d2 = self.dec2(x2); show("dec2", d2)

        u1 = self.up1(d2); show("up1", u1)
        x1 = torch.cat([u1, e1], dim=1); show("concat1", x1)
        d1 = self.dec1(x1); show("dec1", d1)

        out = self.head(d1); show("output", out)
        return out



def iou(logits, masks, cls=1):
    pred = logits.argmax(dim=1)
    inter = ((pred == cls) & (masks == cls)).sum().item()
    union = ((pred == cls) | (masks == cls)).sum().item()
    return inter / union if union else 1.0


@torch.no_grad()
def evaluate(model, dataset, loss_fn):
    model.eval()
    logits = model(dataset.images)
    return loss_fn(logits, dataset.masks).item(), iou(logits, dataset.masks)


def train(model, train_set, val_set, epochs, lr):
    loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train": [], "val": []}
    best_loss, best_state = float("inf"), None

    init_loss, _ = evaluate(model, train_set, loss_fn)
    print(f"initial loss {init_loss:.3f}  (ln {NUM_CLASSES} = {math.log(NUM_CLASSES):.3f})")

    for epoch in range(1, epochs + 1):
        model.train()
        for images, masks in loader:
            loss = loss_fn(model(images), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss, train_iou = evaluate(model, train_set, loss_fn)
        val_loss, val_iou = evaluate(model, val_set, loss_fn)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:4d}  train loss {train_loss:.4f}  IoU {train_iou:.4f}  |  "
                  f"val loss {val_loss:.4f}  IoU {val_iou:.4f}")


    model.load_state_dict(best_state)
    print(f"restored weights with the lowest train loss {best_loss:.4f}")
    return history



def plot_history(history, path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


@torch.no_grad()
def plot_predictions(model, train_set, val_set, path, n=3):
    model.eval()
    rows = [("train", train_set, i) for i in range(n)] + [("val", val_set, i) for i in range(n)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.5 * len(rows)))
    for row, (split, dataset, i) in enumerate(rows):
        image, mask = dataset[i]
        logits = model(image.unsqueeze(0))
        pred = logits.argmax(dim=1)[0]
        score = iou(logits, mask.unsqueeze(0))
        panels = [
            (image[0], f"{split} {i}: image", {}),
            (mask, "mask", {"vmin": 0, "vmax": NUM_CLASSES - 1}),
            (pred, f"prediction, IoU {score:.4f}", {"vmin": 0, "vmax": NUM_CLASSES - 1}),
        ]
        for col, (data, title, kwargs) in enumerate(panels):
            axes[row, col].imshow(data, cmap="gray", **kwargs)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    train_set = ShapesDataset(N_TRAIN, IMAGE_SIZE, seed=0)
    val_set = ShapesDataset(N_VAL, IMAGE_SIZE, seed=1)

    print("=== data ===")
    check_data(train_set, NUM_CLASSES)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES, base=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== model: {n_params / 1e3:.0f}k parameters ===")
    with torch.no_grad():
        model(train_set.images[:1], verbose=True)

    print("\n=== training ===")
    history = train(model, train_set, val_set, EPOCHS, LR)

    plot_history(history, "loss.png")
    plot_predictions(model, train_set, val_set, "predictions.png")
    print("\nsaved loss.png and predictions.png")


if __name__ == "__main__":
    main()
