import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from u_net import (EPOCHS, IMAGE_SIZE, LR, N_TRAIN, N_VAL, NUM_CLASSES,
                  ShapesDataset, UNet, iou, make_sample, train)

N_TEST = 50
TEST_SEED = 123
THRESHOLD = 0.3 + 0.35 / 2  
MAX_SIGMA = 0.45           


def clean_image(mask):
    """The image from make_sample without noise."""
    size = mask.shape[-1]
    xx = np.tile(np.arange(size, dtype=np.float32), (size, 1))
    return 0.3 + 0.35 * mask + 0.2 * (xx / size - 0.5)



def gaussian(sigma):
    return lambda img, rng: img + rng.normal(0.0, sigma, img.shape)


def salt_pepper(p, low=-0.5, high=1.5):
    """A fraction p of pixels becomes very dark or very bright."""
    def add(img, rng):
        out = img.copy()
        r = rng.random(img.shape)
        out[r < p / 2] = low
        out[(r >= p / 2) & (r < p)] = high
        return out
    return add


def speckle(sigma):
    """Multiplicative noise: brighter regions get stronger noise."""
    return lambda img, rng: img * (1.0 + rng.normal(0.0, sigma, img.shape))


CONDITIONS = {  
    "clean": (lambda img, rng: img, True),
    "gauss 0.15": (gaussian(0.15), True),
    "gauss 0.30 (train)": (gaussian(0.30), True),
    "gauss 0.45": (gaussian(0.45), True),
    "salt-pepper 10%": (salt_pepper(0.10), False),
    "speckle 0.5": (speckle(0.5), False),
}


def make_test_masks():
    rng = np.random.default_rng(TEST_SEED)
    return np.stack([make_sample(IMAGE_SIZE, rng)[1] for _ in range(N_TEST)])


def make_test_images(masks, add_noise):
    rng = np.random.default_rng(TEST_SEED + 1)
    return np.stack([add_noise(clean_image(m), rng) for m in masks]).astype(np.float32)


# augmentation

def add_train_noise(img, rng):
    """Random brightness/contrast, then Gaussian noise of random strength."""
    mean = img.mean()
    img = mean + rng.uniform(0.7, 1.3) * (img - mean) + rng.uniform(-0.15, 0.15)
    return img + rng.normal(0.0, rng.uniform(0.0, MAX_SIGMA), img.shape)


class AugmentedShapes(Dataset):
    """The same images and masks as `base`, with a new random augmentation on every access."""

    def __init__(self, base, seed=0):
        self.images = base.images  
        self.masks = base.masks
        self.clean = [clean_image(m.numpy()) for m in base.masks]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        img, mask = self.clean[idx], self.masks[idx].numpy()
        k = self.rng.integers(4) 
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        if self.rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        img = add_train_noise(img, self.rng)
        image = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).unsqueeze(0)
        return image, torch.from_numpy(np.ascontiguousarray(mask))


def fit(name, train_set, val_set):
    print(f"\n=== {name} ===")
    torch.manual_seed(0)  
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, base=16)
    train(model, train_set, val_set, EPOCHS, LR)
    return model.eval()


@torch.no_grad()
def predict(model, images, batch_size=10):
    x = torch.from_numpy(images).unsqueeze(1)
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def threshold_iou(images, masks):
    pred = images > THRESHOLD
    true = masks == 1
    return (pred & true).sum() / (pred | true).sum()



def save_augmentation_preview(dataset, path, n=6):
    """Several augmented versions of training image 0: the mask must follow the image."""
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for k in range(n):
        image, mask = dataset[0]
        axes[0, k].imshow(image[0], cmap="gray")
        axes[1, k].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_results(rows, path):
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.5 * len(rows)))
    for r, (name, image, mask, pred_b, pred_a) in enumerate(rows):
        panels = [(image, name), (mask, "true mask"), (pred_b, "baseline"), (pred_a, "augmented")]
        for c, (data, title) in enumerate(panels):
            kwargs = {} if c == 0 else {"vmin": 0, "vmax": 1}
            axes[r, c].imshow(data, cmap="gray", **kwargs)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# main

def main():
    base = ShapesDataset(N_TRAIN, IMAGE_SIZE, seed=0)  
    val_set = ShapesDataset(N_VAL, IMAGE_SIZE, seed=1)
    save_augmentation_preview(AugmentedShapes(base, seed=1), "robust_augmentations.png")

    model_b = fit("baseline: no augmentation", base, val_set)
    model_a = fit("augmented", AugmentedShapes(base, seed=0), val_set)

    masks = make_test_masks()
    y = torch.from_numpy(masks)
    print(f"\n=== test: {N_TEST} new images per condition, IoU of the cell class ===")
    print(f"{'condition':<20} {'seen in aug':>11} {'baseline':>9} {'augmented':>10} {'threshold':>10}")
    rows = []
    for name, (add_noise, seen) in CONDITIONS.items():
        images = make_test_images(masks, add_noise)
        logits_b = predict(model_b, images)
        logits_a = predict(model_a, images)
        print(f"{name:<20} {'yes' if seen else 'no':>11} {iou(logits_b, y):>9.4f} "
              f"{iou(logits_a, y):>10.4f} {threshold_iou(images, masks):>10.4f}")
        rows.append((name, images[0], masks[0], logits_b[0].argmax(dim=0), logits_a[0].argmax(dim=0)))

    save_results(rows, "robust_results.png")
    print("\nsaved robust_augmentations.png and robust_results.png")


if __name__ == "__main__":
    main()
