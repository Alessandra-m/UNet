import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from u_net import (EPOCHS, IMAGE_SIZE, LR, N_TRAIN, N_VAL, NUM_CLASSES,
                  ShapesDataset, UNet, iou, train)

N_TEST = 50
THRESHOLD = 0.3 + 0.35 / 2 


def make_shuffled(dataset):
    """Give every image the mask of another image (no image keeps its own mask)."""
    perm = torch.roll(torch.arange(len(dataset)), shifts=1)
    dataset.masks = dataset.masks[perm]
    return perm


def threshold_iou(dataset):
    """Baseline without a network: a pixel is a cell if it is brighter than THRESHOLD."""
    pred = dataset.images[:, 0] > THRESHOLD
    mask = dataset.masks == 1
    return (pred & mask).sum().item() / (pred | mask).sum().item()


def run(name, train_set, val_set):
    print(f"\n=== {name} ===")
    torch.manual_seed(0)  
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, base=16)
    history = train(model, train_set, val_set, EPOCHS, LR)
    return model.eval(), history


def score(logits, mask):
    """IoU with four decimals and the number of pixels that differ from the mask."""
    wrong = (logits.argmax(dim=1)[0] != mask).sum().item()
    return f"IoU {iou(logits, mask.unsqueeze(0)):.4f}"


@torch.no_grad()
def final_iou(model, dataset, batch_size=10):
    logits = torch.cat([model(dataset.images[i:i + batch_size])
                        for i in range(0, len(dataset), batch_size)])
    return iou(logits, dataset.masks)



def plot_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ["train", "val"]):
        for name, history in histories.items():
            ax.plot(history[split], label=name)
        ax.set_yscale("log")
        ax.set_title(f"{split} loss")
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("cross-entropy loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_train(model_c, train_correct, model_s, train_shuffled, path):
    """Each model is compared with the masks it was trained on."""
    n = len(train_correct)
    fig, axes = plt.subplots(n, 5, figsize=(12.5, 2.5 * n))
    for i in range(n):
        image, mask_c = train_correct[i]
        _, mask_s = train_shuffled[i]
        logits_c = model_c(image.unsqueeze(0))
        logits_s = model_s(image.unsqueeze(0))
        panels = [
            (image[0], f"train {i}: image"),
            (mask_c, "correct: its target"),
            (logits_c.argmax(dim=1)[0], f"correct: {score(logits_c, mask_c)}"),
            (mask_s, "shuffled: its target"),
            (logits_s.argmax(dim=1)[0], f"shuffled: {score(logits_s, mask_s)}"),
        ]
        for col, (data, title) in enumerate(panels):
            kwargs = {} if col == 0 else {"vmin": 0, "vmax": 1}
            axes[i, col].imshow(data, cmap="gray", **kwargs)
            axes[i, col].set_title(title, fontsize=8)
            axes[i, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_new(model_c, model_s, dataset, path, n=4):
    """New images: both models are compared with the correct masks."""
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    for i in range(n):
        image, mask = dataset[i]
        logits_c = model_c(image.unsqueeze(0))
        logits_s = model_s(image.unsqueeze(0))
        panels = [
            (image[0], f"test {i}: image"),
            (mask, "true mask"),
            (logits_c.argmax(dim=1)[0], f"correct, {score(logits_c, mask)}"),
            (logits_s.argmax(dim=1)[0], f"shuffled, {score(logits_s, mask)}"),
        ]
        for col, (data, title) in enumerate(panels):
            kwargs = {} if col == 0 else {"vmin": 0, "vmax": 1}
            axes[i, col].imshow(data, cmap="gray", **kwargs)
            axes[i, col].set_title(title, fontsize=9)
            axes[i, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)



def main():
    train_correct = ShapesDataset(N_TRAIN, IMAGE_SIZE, seed=0)
    train_shuffled = ShapesDataset(N_TRAIN, IMAGE_SIZE, seed=0)  
    perm = make_shuffled(train_shuffled)
    val_set = ShapesDataset(N_VAL, IMAGE_SIZE, seed=1)
    test_set = ShapesDataset(N_TEST, IMAGE_SIZE, seed=2)

    print(f"shuffled pairing: image i gets mask {perm.tolist()}")
    assert torch.equal(train_correct.images, train_shuffled.images)
    assert not torch.equal(train_correct.masks, train_shuffled.masks)

    model_c, hist_c = run("correct masks", train_correct, val_set)
    model_s, hist_s = run("shuffled masks", train_shuffled, val_set)

    print(f"\n=== summary: IoU of the cell class ===")
    print(f"{'run':<10} {'train (4)':>10} {'val (4)':>9} {'test (50)':>10}")
    print(f"{'correct':<10} {final_iou(model_c, train_correct):>10.4f} "
          f"{final_iou(model_c, val_set):>9.4f} {final_iou(model_c, test_set):>10.4f}")
    print(f"{'shuffled':<10} {final_iou(model_s, train_shuffled):>10.4f} "
          f"{final_iou(model_s, val_set):>9.4f} {final_iou(model_s, test_set):>10.4f}")
    print(f"{'threshold':<10} {threshold_iou(train_correct):>10.4f} "
          f"{threshold_iou(val_set):>9.4f} {threshold_iou(test_set):>10.4f}")
    print("train is scored against the masks each model was trained on; val and test against the true masks")

    plot_curves({"correct": hist_c, "shuffled": hist_s}, "shuffle_curves.png")
    plot_train(model_c, train_correct, model_s, train_shuffled, "shuffle_train.png")
    plot_new(model_c, model_s, test_set, "shuffle_test.png")
    print("\nsaved shuffle_curves.png, shuffle_train.png, shuffle_test.png")


if __name__ == "__main__":
    main()
