import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
    
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def show_frame(frame, title=None):
    """
    Display a BGR frame inline (e.g. in a Jupyter notebook).

    Args:
        frame: Frame as a BGR numpy array.
        title: Optional title to show above the image.
    """
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_keypoints(frame, keypoints, names=COCO_KEYPOINT_NAMES, title=None):
    """
    Display a BGR frame with labeled pose keypoints overlaid.

    Args:
        frame: Frame as a BGR numpy array.
        keypoints: Array of (x, y) keypoint coordinates, e.g.
            results[0].keypoints.xy[0] from a YOLO-Pose prediction.
        names: Label for each keypoint, in the same order as keypoints.
            Defaults to the 17 COCO keypoint names used by YOLO-Pose.
        title: Optional title to show above the image.
    """
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for name, (x, y) in zip(names, keypoints):
        plt.scatter(x, y, c="red", s=20)
        plt.text(x + 5, y, name, color="yellow", fontsize=8)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def plot_signal_frames(frames, results, value_key, threshold, flag_label, ok_label="OK",
                        higher_is_flag=True, n_cols=4, figsize_per_cell=(4, 5)):
    n = len(results)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)
    )
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, results):
        if isinstance(r, dict):
            frame_idx = r["frame"]
            value = r[value_key]
        else:
            frame_idx, value = r  # assume tuple (frame_idx, value)

        flagged = value > threshold if higher_is_flag else value < threshold
        label = flag_label if flagged else ok_label

        frame = frames[frame_idx]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {frame_idx}\n{value_key}: {value:.3f} — {label}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    


def plot_both_ankles(x_smooth_L, y_smooth_L, x_smooth_R, y_smooth_R):
    """
    Plot smoothed X/Y coordinates for both ankles on one combined graph.

    Parameters:
        x_smooth_L, y_smooth_L : smoothed left ankle coordinate arrays
        x_smooth_R, y_smooth_R : smoothed right ankle coordinate arrays
    """
    frames = np.arange(len(x_smooth_L))

    plt.figure(figsize=(12, 5))

    # Left ankle
    plt.plot(frames, x_smooth_L, color="tab:blue", linewidth=2, label="Left Ankle X")
    # plt.plot(frames, y_smooth_L, color="tab:cyan", linewidth=2, label="Left Ankle Y")

    # Right ankle
    plt.plot(frames, x_smooth_R, color="tab:red", linewidth=2, label="Right Ankle X")
    # plt.plot(frames, y_smooth_R, color="tab:orange", linewidth=2, label="Right Ankle Y")

    plt.xlabel("Frame")
    plt.ylabel("Pixel coordinate")
    plt.title("Left vs Right Ankle Coordinates (Smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    


def plot_frames_grid(frames, frame_indices, n_cols=2, figsize_per_cell=(4, 5), title_prefix=""):
    """
    Display a grid of frames, n_cols per row.

    frames: dict mapping frame_idx -> image (BGR, from cv2)
    frame_indices: list/array of frame indices to display
    n_cols: number of images per row
    """
    n = len(frame_indices)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, frame_indices):
        frame = frames[idx]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title_prefix}Frame {idx}")
        ax.axis("off")

    # hide any leftover empty subplots so they don't show a bare coordinate grid
    for ax in axes[n:]:
        ax.axis("off")


def plot_vertical_oscillation(hip_y, osc_results, title="Vertical Oscillation"):
    plt.figure(figsize=(14, 5))
    plt.plot(hip_y, label="Hip Y", color="steelblue")

    ymin = np.min(hip_y)
    ymax = np.max(hip_y)
    padding = (ymin - ymax) * 0.15  # extra space, proportional to data range

    plt.ylim(ymax - padding, ymin + padding)  # inverted axis, extra room top and bottom

    for r in osc_results:
        color = "red" if r.get("excessive") else "green"
        plt.axvspan(r["start_frame"], r["end_frame"], alpha=0.15, color=color)
        mid = (r["start_frame"] + r["end_frame"]) // 2
        plt.text(mid, ymin + padding * 0.5, f"{r['normalized_oscillation']:.2f}",
                ha="center", fontsize=8)

    plt.xlabel("Frame")
    plt.ylabel("Hip Y (pixels)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

