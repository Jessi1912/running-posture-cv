
import numpy as np

def calculate_overstriding(ankle_x, hip_x, contact_frames, leg_lengths, threshold):
    results = []
    for frame in contact_frames:
        distance_px = ankle_x[frame] - hip_x[frame]
        normalized_distance = distance_px / leg_lengths[frame]
        results.append({
            "frame": frame,
            "distance_px": distance_px,
            "normalized_distance": normalized_distance,
            "overstriding": normalized_distance > threshold  # threshold for overstriding
        })
    return results

def analyze_knee_flexion(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y, contact_frames, threshold_angle=160):
    """
    Calculate knee flexion angle at each contact frame and flag straight (risky) landings.
    """
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    results = []
    for frame in contact_frames:
        hip = (hip_x[frame], hip_y[frame])
        knee = (knee_x[frame], knee_y[frame])
        ankle = (ankle_x[frame], ankle_y[frame])

        angle = calculate_angle(hip, knee, ankle)
        straight_landing = angle > threshold_angle

        results.append({
            "frame": frame,
            "knee_angle": angle,
            "straight_landing": straight_landing
        })

    return results

def analyze_vertical_oscillation(hip_y, contact_frames, leg_lengths, threshold_ratio=None):
    results = []
    contact_frames = sorted(contact_frames)

    for i in range(len(contact_frames) - 1):
        start = contact_frames[i]
        end = contact_frames[i + 1]

        hip_segment = hip_y[start:end + 1]
        oscillation_px = np.max(hip_segment) - np.min(hip_segment)
        normalized_oscillation = oscillation_px / leg_lengths[start]

        result = {
            "start_frame": start,
            "end_frame": end,
            "oscillation_px": oscillation_px,
            "normalized_oscillation": normalized_oscillation
        }
        if threshold_ratio is not None:
            result["excessive"] = normalized_oscillation > threshold_ratio

        results.append(result)

    return results