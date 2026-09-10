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


def visualize_trunk_lean(video_path, frame_idx, shoulder_mid_x, shoulder_mid_y,
                          hip_mid_x, hip_mid_y, lean_deg, label=""):
    """
    Draws the hip->shoulder line on the actual video frame, the same way
    visualize_leg_length draws the hip->ankle line, so trunk lean can be
    sanity-checked visually rather than trusted as a number alone.
    Line color signals direction: green = forward lean, orange = backward lean.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {frame_idx}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hx, hy = int(hip_mid_x[frame_idx]), int(hip_mid_y[frame_idx])
    sx, sy = int(shoulder_mid_x[frame_idx]), int(shoulder_mid_y[frame_idx])

    line_color = (0, 200, 0) if lean_deg[frame_idx] >= 0 else (255, 140, 0)  # green fwd, orange back

    cv2.circle(frame_rgb, (hx, hy), 8, (255, 0, 0), -1)
    cv2.circle(frame_rgb, (sx, sy), 8, (0, 0, 255), -1)
    cv2.line(frame_rgb, (hx, hy), (sx, sy), line_color, 3)

    # vertical reference line from the hip, so the lean angle is visible at a glance
    cv2.line(frame_rgb, (hx, hy), (hx, hy - 150), (200, 200, 200), 1)

    plt.figure(figsize=(6, 8))
    plt.imshow(frame_rgb)
    direction_label = "forward" if lean_deg[frame_idx] >= 0 else "backward"
    plt.title(f"{label} — Frame {frame_idx}\n{direction_label} lean: {lean_deg[frame_idx]:.1f} deg")
    plt.axis("off")
    plt.show()


def visualize_kickback_frame(video_path, frame_idx, hip_x, hip_y, ankle_x, ankle_y,
                              kickback_score, low_kickback_threshold, label=""):
    """
    Draws hip and ankle points plus the vertical gap between them at the
    peak-kick frame. Color signals whether this swing phase was flagged as
    shuffling (low_kickback_threshold).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {frame_idx}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hx, hy = int(hip_x[frame_idx]), int(hip_y[frame_idx])
    ax_, ay = int(ankle_x[frame_idx]), int(ankle_y[frame_idx])

    color = (255, 60, 60) if kickback_score < low_kickback_threshold else (0, 200, 0)

    cv2.circle(frame_rgb, (hx, hy), 8, (255, 0, 0), -1)
    cv2.circle(frame_rgb, (ax_, ay), 8, (0, 255, 0), -1)
    cv2.line(frame_rgb, (ax_, hy), (ax_, ay), color, 3)          # vertical gap being measured
    cv2.line(frame_rgb, (hx, hy), (ax_, ay), (255, 255, 0), 1)   # thin hip-ankle reference

    plt.figure(figsize=(6, 8))
    plt.imshow(frame_rgb)
    status = "SHUFFLING" if kickback_score < low_kickback_threshold else "OK"
    plt.title(f"{label} — Frame {frame_idx}\nKickback score: {kickback_score:.2f} ({status})")
    plt.axis("off")
    plt.show()


def print_misposture_summary(summary):
    """Formats the summary dict as a readable report instead of a raw dict dump."""
    print("=" * 50)
    print("MISPOSTURE SUMMARY")
    print("=" * 50)

    print("\n[Overstriding]")
    for side, stats in summary["overstriding"].items():
        if stats:
            print(f"  {side}: {stats['flagged']}/{stats['contacts_checked']} contacts flagged "
                  f"({stats['pct_flagged']}%)")

    print("\n[Knee Flexion at Foot Strike]")
    for side, stats in summary["knee_flexion"].items():
        if stats:
            print(f"  {side}: {stats['flagged_straight']}/{stats['contacts_checked']} contacts flagged straight "
                  f"({stats['pct_flagged']}%) | avg angle {stats['avg_angle_deg']} deg")

    print("\n[Vertical Oscillation]")
    for side, stats in summary["vertical_oscillation"].items():
        if stats:
            print(f"  {side}: {stats['flagged_excessive']}/{stats['strides_checked']} strides flagged excessive "
                  f"({stats['pct_flagged']}%) | avg normalized osc {stats['avg_normalized_osc']}")

    tl = summary["trunk_lean"]
    print("\n[Trunk Lean]")
    print(f"  mean: {tl['mean_deg']} deg | range: [{tl['max_backward_deg']}, {tl['max_forward_deg']}] deg")
    print(f"  excess forward: {tl['pct_frames_excess_forward']}% of frames | "
          f"excess backward: {tl['pct_frames_excess_backward']}% of frames")

    cad = summary["cadence"]
    print("\n[Cadence]")
    print(f"  {cad['spm']} spm" + (" — LOW" if cad["low_cadence"] else ""))

    print("\n[Heel Whip / Kickback]")
    for side, stats in summary["kickback"].items():
        if stats:
            print(f"  {side}: {stats['flagged_shuffling']}/{stats['swings_checked']} swings flagged shuffling "
                  f"(avg score {stats['avg_score']})")

    # print("\n[Arm Swing]")
    # for side, stats in summary["arm_swing"].items():
    #     if stats:
    #         print(f"  {side}: avg amplitude {stats['avg_amplitude_deg']} deg")


def plot_misposture_summary(summary, low_cadence_threshold):
    """
    Same stacked red/green bars as before, now labeling both segments with
    their percentage (white text on red, black text on green) instead of
    just the flagged portion. A segment's label is skipped entirely when
    that segment is 0%, rather than printing a "0%" that clutters the bar
    with no useful information.
    """
    labels, flagged_pct = [], []

    ovr = summary["overstriding"]["right"]
    if ovr:
        labels.append("Overstriding\n(R)")
        flagged_pct.append(ovr["pct_flagged"])

    ovl = summary["overstriding"]["left"]
    if ovl:
        labels.append("Overstriding\n(L)")
        flagged_pct.append(ovl["pct_flagged"])

    for side in ("right", "left"):
        kf = summary["knee_flexion"][side]
        if kf:
            labels.append(f"Knee Flexion\n({side[0].upper()})")
            flagged_pct.append(kf["pct_flagged"])

    # for side in ("right", "left"):
    for side in ("right",):
        vo = summary["vertical_oscillation"][side]
        if vo:
            labels.append(f"Vertical Osc.\n({side[0].upper()})")
            flagged_pct.append(vo["pct_flagged"])


    tl = summary["trunk_lean"]
    labels.append("Trunk Lean\n(Forward)")
    flagged_pct.append(tl["pct_frames_excess_forward"])

    labels.append("Trunk Lean\n(Backward)")
    flagged_pct.append(tl["pct_frames_excess_backward"])

    for side in ("right", "left"):
        kb = summary["kickback"][side]
        if kb:
            pct = round(100 * kb["flagged_shuffling"] / kb["swings_checked"], 1)
            labels.append(f"Kickback\n({side[0].upper()})")
            flagged_pct.append(pct)

    ok_pct = [100 - p for p in flagged_pct]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].bar(labels, flagged_pct, color="tab:red", label="Flagged")
    axes[0].bar(labels, ok_pct, bottom=flagged_pct, color="tab:green", label="OK")
    axes[0].set_ylabel("% of frames / contacts / swings")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Misposture Summary")
    axes[0].legend()

    for i, (p_flagged, p_ok) in enumerate(zip(flagged_pct, ok_pct)):
        if p_flagged > 0:
            axes[0].text(i, p_flagged / 2, f"{p_flagged:.1f}%", ha="center", va="center",
                         color="white", fontweight="bold")
        if p_ok > 0:
            axes[0].text(i, p_flagged + p_ok / 2, f"{p_ok:.1f}%", ha="center", va="center",
                         color="black", fontweight="bold")

    # --- Cadence (separate unit, separate panel) ---
    cad = summary["cadence"]
    cad_color = "tab:red" if cad["low_cadence"] else "tab:green"
    axes[1].bar(["Cadence"], [cad["spm"]], color=cad_color)
    axes[1].axhline(low_cadence_threshold, color="black", linestyle="--", linewidth=1, label="Threshold")
    axes[1].set_ylabel("steps/min")
    axes[1].set_ylim(0, max(cad["spm"], low_cadence_threshold) * 1.2)
    axes[1].legend()
    axes[1].text(0, cad["spm"] + 3, f"{cad['spm']} spm", ha="center")

    plt.tight_layout()
    plt.show()


def plot_toe_off_combined(y_smooth, candidates, toe_off_frames, velocity, label="Ankle"):
    frames = np.arange(len(y_smooth))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(frames, y_smooth, color="tab:blue", label="Y-position")
    axes[0].scatter(candidates, y_smooth[candidates], color="gray", marker="x",
                     s=50, label="Candidates (unconfirmed)", zorder=4)
    axes[0].scatter(toe_off_frames, y_smooth[toe_off_frames], color="red", marker="v",
                     s=70, label="Confirmed toe-off", zorder=5)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Y-position (px)")
    axes[0].set_title(f"{label}: Candidates vs Confirmed Toe-off")
    axes[0].legend()

    axes[1].plot(frames, velocity, color="tab:purple", linewidth=1.2)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].scatter(toe_off_frames, velocity[toe_off_frames], color="red", marker="v", s=70, zorder=5)
    axes[1].set_ylabel("Velocity (px/sec)")
    axes[1].set_xlabel("Frame")

    plt.tight_layout()
    plt.show()

def plot_trunk_lean(trunk_lean_deg, threshold_deg):
    """
    Plots trunk lean over time with excess forward/backward lean thresholds,
    and returns the frame indices where lean exceeds each threshold.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(trunk_lean_deg, color="tab:green", label="Trunk lean (deg)")
    plt.axhline(0, color="gray", linestyle=":", label="Upright")
    plt.axhline(threshold_deg, color="red", linestyle="--", linewidth=1, label="Excess forward lean")
    plt.axhline(-threshold_deg, color="orange", linestyle="--", linewidth=1, label="Excess backward lean")
    plt.xlabel("Frame")
    plt.ylabel("Lean from vertical (deg)")
    plt.title("Trunk Lean Over Time (+ = forward, - = backward)")
    plt.legend()
    plt.tight_layout()
    plt.show()
