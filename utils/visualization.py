import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_depth_for_vis(depth, max_depth=80.0):
    """
    Normalize depth for visualization only.
    This MUST NOT be used for training or evaluation.
    """
    depth = np.clip(depth, 0, max_depth)
    return depth / max_depth


def visualize_sample(
    rgb,
    gt_depth,
    pred_depth,
    save_path=None,
    max_depth=80.0,
    cmap="plasma"
):
    """
    Visualize RGB image, ground-truth depth, and predicted depth.

    Args:
        rgb (Tensor or np.ndarray): shape [3,H,W] or [H,W,3]
        gt_depth (Tensor or np.ndarray): [H,W] in meters
        pred_depth (Tensor or np.ndarray): [H,W] in meters
        save_path (str): optional path to save the figure
        max_depth (float): KITTI max depth (default 80m)
        cmap (str): matplotlib colormap
    """

    # Convert tensors to numpy
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    if hasattr(gt_depth, "detach"):
        gt_depth = gt_depth.detach().cpu().numpy()
    if hasattr(pred_depth, "detach"):
        pred_depth = pred_depth.detach().cpu().numpy()

    # Ensure correct RGB shape
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    # Normalize depth ONLY for visualization
    gt_vis = normalize_depth_for_vis(gt_depth, max_depth)
    pred_vis = normalize_depth_for_vis(pred_depth, max_depth)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].imshow(rgb)
    axs[0].set_title("Input RGB")
    axs[0].axis("off")

    axs[1].imshow(gt_vis, cmap=cmap)
    axs[1].set_title("Ground Truth Depth (m)")
    axs[1].axis("off")

    axs[2].imshow(pred_vis, cmap=cmap)
    axs[2].set_title("Predicted Depth (m)")
    axs[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()