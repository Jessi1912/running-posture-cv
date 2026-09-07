import cv2
import matplotlib.pyplot as plt

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
