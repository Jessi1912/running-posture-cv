import cv2


def get_video_info(video_path):
    """
    Read basic video properties without decoding frames.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with fps, width, height, frame_count, and rotation (degrees
        of rotation from container metadata, 0 if none is present).

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "rotation": int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
    }
    cap.release()
    return info


def get_frames(video_path):
    """
    Read all frames of a video into memory.

    Args:
        video_path: Path to the video file.

    Returns:
        List of frames as BGR numpy arrays, in order.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def trim_frames(frames, fps, start_time=None, end_time=None):
    """
    Cut a list of frames down to a time range.

    Args:
        frames: List of frames, as returned by get_frames.
        fps: Frame rate of the video, from get_video_info.
        start_time: Start of the clip in seconds (default: from the start).
        end_time: End of the clip in seconds (default: to the end).

    Returns:
        The frames within [start_time, end_time).
    """
    start_frame = int(start_time * fps) if start_time is not None else 0
    end_frame = int(end_time * fps) if end_time is not None else len(frames)
    return frames[start_frame:end_frame]


def resize_frames(frames, height):
    """
    Downscale frames to a target height, preserving aspect ratio.

    Args:
        frames: List of frames to resize.
        height: Target height in pixels. Frames already at or below this
            height are left unchanged.

    Returns:
        The resized frames.
    """
    resized = []
    for frame in frames:
        if frame.shape[0] <= height:
            resized.append(frame)
            continue
        scale = height / frame.shape[0]
        resized.append(cv2.resize(frame, (int(frame.shape[1] * scale), height)))
    return resized


def write_video(frames, output_path, fps):
    """
    Write a list of frames to a video file.

    Args:
        frames: List of frames as BGR numpy arrays, in order.
        output_path: Path to write the video file to.
        fps: Frame rate for the output video.

    Raises:
        ValueError: If frames is empty or the video writer cannot be opened.
    """
    if not frames:
        raise ValueError("Cannot write video: no frames given")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()
