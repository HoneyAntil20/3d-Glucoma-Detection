"""
Visualize predictions with balanced glaucoma and no-glaucoma cases
using the same data pipeline as training (get_data_loaders).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from utils.dataset import GlaucomaDataset   # keep if you still need it elsewhere
from utils.dataset import get_data_loaders
from utils.model import create_model


def visualize_balanced_predictions(
    model,
    dataset,
    device,
    num_images=10,
    save_path="balanced_predictions.png",
    use_metadata=True,
):
    """
    Create visualization with balanced glaucoma and no-glaucoma cases.

    Args:
        model: trained model
        dataset: Dataset object (e.g. test_loader.dataset)
        device: torch.device
        num_images: total images to display (will be ~half/half per class)
        save_path: output PNG path
        use_metadata: whether to pass metadata to the model
    """
    model.eval()

    # Collect indices for each class
    glaucoma_indices = []
    no_glaucoma_indices = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"].item()
        if label == 1:
            glaucoma_indices.append(i)
        else:
            no_glaucoma_indices.append(i)

    print(f"Total glaucoma samples in split: {len(glaucoma_indices)}")
    print(f"Total no-glaucoma samples in split: {len(no_glaucoma_indices)}")

    if len(glaucoma_indices) == 0 or len(no_glaucoma_indices) == 0:
        raise ValueError(
            "One of the classes has 0 samples in this split. "
            "Choose another split or check your data."
        )

    num_each = num_images // 2
    num_each_gl = min(num_each, len(glaucoma_indices))
    num_each_no = min(num_each, len(no_glaucoma_indices))

    selected_glaucoma = np.random.choice(glaucoma_indices, num_each_gl, replace=False)
    selected_no_glaucoma = np.random.choice(
        no_glaucoma_indices, num_each_no, replace=False
    )

    # Combine & shuffle
    indices = np.concatenate([selected_glaucoma, selected_no_glaucoma])
    np.random.shuffle(indices)

    total_to_show = len(indices)
    rows = 2
    cols = max(1, int(np.ceil(total_to_show / rows)))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)

    correct_predictions = 0

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]

            image = sample["image"].unsqueeze(0).to(device)  # (1, C, H, W)
            label = sample["label"].item()
            age = float(sample.get("age", torch.tensor(0.0)))
            md = float(sample.get("md", torch.tensor(0.0)))

            if use_metadata and "metadata" in sample:
                metadata = sample["metadata"].unsqueeze(0).to(device)  # (1, M)
                output = model(image, metadata)
            else:
                output = model(image)

            probs = torch.softmax(output, dim=1)
            prob_glaucoma = probs[0, 1].item()
            predicted = torch.argmax(output, dim=1).item()

            is_correct = predicted == label
            if is_correct:
                correct_predictions += 1

            # Take first channel (RNFL thickness map)
            img_display = image[0, 0].cpu().numpy()

            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])

            im = ax.imshow(img_display, cmap="hot", aspect="auto", vmin=0, vmax=1)
            ax.axis("off")

            actual_label = "Glaucoma" if label == 1 else "No Glaucoma"
            pred_label = "Glaucoma" if predicted == 1 else "No Glaucoma"

            border_color = "green" if is_correct else "red"
            status = "CORRECT" if is_correct else "INCORRECT"

            info_text = (
                f"Actual: {actual_label}\n"
                f"Predicted: {pred_label}\n"
                f"Status: {status}\n"
                f"Glaucoma Prob: {prob_glaucoma*100:.1f}%\n"
                f"Age: {age:.0f} yrs | MD: {md:.2f} dB"
            )

            props = dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor=border_color,
                linewidth=2.5,
            )
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
                color="black",
                fontweight="bold",
                family="monospace",
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)

            # Colorbar only on the first subplot
            if idx == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("RNFL Thickness (Normalized)", fontsize=8)

    acc = 100.0 * correct_predictions / total_to_show
    fig.suptitle(
        f"Glaucoma Detection: Balanced Sample Predictions\n"
        f"Correct: {correct_predictions}/{total_to_show} ({acc:.1f}%) | "
        f"Green = Correct, Red = Incorrect",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Balanced visualization saved to {save_path}")
    return correct_predictions, total_to_show


def main():
    parser = argparse.ArgumentParser(description="Visualize balanced predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset-001",
        help="Path to dataset directory (root with train/val/test npz folders)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Total number of images to visualize (approx half from each class)",
    )
    parser.add_argument(
        "--use_metadata",
        action="store_true",
        default=True,
        help="Use metadata in classification",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (only used to build loaders; dataset is used directly)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to visualize from",
    )
    args = parser.parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (reuse same pipeline as training)
    print("Loading datasets via get_data_loaders ...")
    _, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.split == "val":
        dataset = val_loader.dataset
    else:
        dataset = test_loader.dataset

    print(f"Selected split: {args.split}")
    print(f"Total images in split: {len(dataset)}")

    # Count classes
    glaucoma_count = sum(1 for i in range(len(dataset)) if dataset[i]["label"].item() == 1)
    no_glaucoma_count = len(dataset) - glaucoma_count
    print(f"Glaucoma cases: {glaucoma_count}, No Glaucoma cases: {no_glaucoma_count}")

    # Model
    print("Creating model...")
    model = create_model(
        num_classes=2,
        pretrained=False,
        use_metadata=args.use_metadata,
        device=device,
    )

    # Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Visualize
    out_name = f"balanced_predictions_{args.split}.png"
    print(f"\nCreating balanced visualization of {args.num_images} images...")
    correct, total = visualize_balanced_predictions(
        model,
        dataset,
        device,
        num_images=args.num_images,
        save_path=out_name,
        use_metadata=args.use_metadata,
    )

    print(
        f"\nResults: {correct}/{total} correct predictions ({100.0 * correct / total:.1f}%)"
    )
    print(f"Visualization saved to: {out_name}")


if __name__ == "__main__":
    main()
