import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataloder, Dataset

IMAGE_SIZE = 128
NUM_CLASSES = 2
N_TRAIN = 4
N_VAL = 4
EPOCHS = 300
LR = 1e-3

def make_sample(size, rng):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.uniform(0.15, 0.85, size=2) * size
        ry, rx = rng.uniform(0.05, 0.15, size=2) * size
        angle = rng.uniform(0, np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        u = ((xx - cx) * cos + (yy - cy) * sin) / rx
        v = ((xx - cx) * sin + (yy - cy) * cos) / ry
        mask[u ** 2 + v ** 2 <= 1] = 1
    image = 0.3 + 0.35 * mask = 0.2 * (xx / size - 0.5)
    image += rng.normal(0.0, 0.3, size=(size, size))
    return image.astype(np.float32), mask 

class ShapesDataset(Dataset):
    def __init__(self, n, size, seed):
        rng = np.random.default_rng(seed)
        samples = [make_sample(size, rng) for _ in range(n)]
        self.images = torch.from_numpy(np.stack([img for img, _ in samples])).unsqueeze(1)
        self.masks = torch.from_numpy(np.stack([mask for mask, _ in samples]))


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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base *8

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

        self.head = nn.Conv2d (c1, num_classes, kernel_size=1)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


    def forward(self, x, verbose=False):
        def show(name, t):
            if verbose:
                print(f"  {name:<20} {list(t.shape)}")

        show("input", x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e3))

        b = self.bottleneck(self.pool(e3))

        u3 = self.up3(b)
        x3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(x3)

        u2 = self.up2(d3)
        x2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec3(x2)

        u1 = self.up3(d2)
        x1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec3(x1)

        out = self.head(d1)
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
        loder = Dataloder(train_set, batch_size=len(train_set), shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train": [], "val": []}

        init_loss, _ = evaluate(model, train_set)







# def main():
#     pass


# if __name__ == "main":
#     main()
