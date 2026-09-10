
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

def cadence_from_stride_gap(avg_gap_frames, fps):
    """
    Convert an average same-foot stride gap (frames between consecutive
    strikes of ONE foot) into that foot's own strike rate, in strikes/min.
    This is the single formula both estimate_stride_frames_from_contacts
    and calculate_cadence rely on -- neither re-derives it independently.
    """
    return (fps / avg_gap_frames) * 60

def compute_direction_of_travel(hip_x_mid):
    """
    +1 if the runner moves toward increasing x (rightward in frame),
    -1 if moving toward decreasing x. Derived from net hip displacement
    across the whole clip so lean/arm-swing angles can be signed
    consistently ("forward" always means "in the direction of travel"),
    regardless of which way the runner faces the camera.
    """
    return np.sign(hip_x_mid[-1] - hip_x_mid[0])

def calculate_trunk_lean(shoulder_x_L, shoulder_y_L, shoulder_x_R, shoulder_y_R,
                          hip_x_L, hip_y_L, hip_x_R, hip_y_R, direction):
    """
    Trunk lean = angle of the hip->shoulder line relative to vertical.

    Uses the midpoint of both shoulders and both hips (rather than a single
    side) per frame. In a side-view video the far-side keypoints are
    partially occluded and noisier than the near-side ones, so averaging
    both sides cancels out most of that left/right jitter and gives one
    clean trunk-line estimate per frame.

    Sign convention: positive = forward lean (torso ahead of hips in the
    direction of travel), negative = backward lean. `direction` flips the
    raw angle so this convention holds regardless of which way the runner
    faces the camera.
    """
    shoulder_mid_x = (shoulder_x_L + shoulder_x_R) / 2
    shoulder_mid_y = (shoulder_y_L + shoulder_y_R) / 2
    hip_mid_x = (hip_x_L + hip_x_R) / 2
    hip_mid_y = (hip_y_L + hip_y_R) / 2

    dx = shoulder_mid_x - hip_mid_x
    dy = hip_mid_y - shoulder_mid_y  # positive: shoulders sit above hips (smaller y in image coords)

    raw_angle_deg = np.degrees(np.arctan2(dx, dy))  # 0 deg = perfectly upright
    trunk_lean_deg = raw_angle_deg * direction

    return trunk_lean_deg, shoulder_mid_x, shoulder_mid_y, hip_mid_x, hip_mid_y

def calculate_cadence(stride_frames_L, stride_frames_R, fps):
    """
    Combined (both-feet) cadence in steps/min. Reuses cadence_from_stride_gap
    per foot -- the same conversion estimate_stride_frames_from_contacts uses
    -- then doubles each (one strike from this foot + one from the other foot
    per stride cycle = 2 steps) and averages the two feet for robustness.
    """
    cadence_L = cadence_from_stride_gap(stride_frames_L, fps) * 2
    cadence_R = cadence_from_stride_gap(stride_frames_R, fps) * 2
    return (cadence_L + cadence_R) / 2

def calculate_kickback_height(ankle_y, hip_y, swing_phases, leg_length_px):
    """
    For each swing phase (toe-off -> next contact of the same leg), measures
    how close the ankle gets to the hip vertically -- i.e. how high the heel
    kicks up toward the glutes.

    In image coordinates that raw distance is actually LARGEST when the leg is
    extended (right at toe-off, or again just before the next landing) and
    SMALLEST at the moment of peak heel flexion. To measure kickback height itself,
    this takes the MINIMUM ankle-hip vertical distance within each swing phase
    instead, normalized by leg length so it's comparable across people/
    camera distances:

        kickback_score = 1 - (min_vertical_distance / leg_length_px)

    A score near 1 means the heel rose almost all the way to hip height
    (efficient recovery); a score near 0 means the ankle barely left its
    stance-phase height ("shuffling" gait).
    """
    results = []
    for start, end in swing_phases:
        window = range(start, end + 1)
        vertical_dist = np.array([abs(hip_y[f] - ankle_y[f]) for f in window])
        min_idx = int(np.argmin(vertical_dist))
        min_dist = vertical_dist[min_idx]
        peak_frame = start + min_idx

        frame_leg_length = leg_length_px[peak_frame]
        results.append({
            "swing_start": start,
            "swing_end": end,
            "peak_kick_frame": start + min_idx,
            "min_vertical_distance_px": min_dist,
            "kickback_score": 1 - (min_dist / frame_leg_length),
        })
    return results

def build_misposture_summary(overstride_R, overstride_L,
                              trunk_lean_deg, trunk_lean_threshold,
                              cadence_spm, low_cadence_threshold,
                              kickback_R, kickback_L, low_kickback_threshold,
                              knee_flexion_R, knee_flexion_L,
                              vertical_osc_R):
    """
    Pulls every misposture signal computed so far into one summary dict --
    one entry per misposture, each with a plain-language flag plus the
    numbers behind it. Doesn't recompute anything; just aggregates results
    that already exist in the notebook (right_overstride, trunk_lean_deg,
    cadence_spm, right_kickback/left_kickback, right_arm_amplitude/left_arm_amplitude),
    so if any of those change, re-run this cell to refresh the summary.
    Left-side inputs are optional (pass None) since only right-leg
    overstriding has been computed so far in this notebook.
    """
    summary = {}

    # --- Overstriding ---
    def overstride_stats(results):
        if not results:
            return None
        flagged = sum(1 for r in results if r["overstriding"])
        return {"contacts_checked": len(results), "flagged": flagged,
                "pct_flagged": round(100 * flagged / len(results), 1)}

    summary["overstriding"] = {
        "right": overstride_stats(overstride_R),
        "left": overstride_stats(overstride_L),
    }

    # --- Knee flexion at foot strike ---
    def knee_flexion_stats(results):
        if not results:
            return None
        flagged = sum(1 for r in results if r["straight_landing"])
        avg_angle = np.mean([r["knee_angle"] for r in results])
        return {"contacts_checked": len(results), "flagged_straight": flagged,
                "pct_flagged": round(100 * flagged / len(results), 1),
                "avg_angle_deg": round(float(avg_angle), 1)}

    summary["knee_flexion"] = {
        "right": knee_flexion_stats(knee_flexion_R),
        "left": knee_flexion_stats(knee_flexion_L),
    }

     # --- Vertical oscillation ---
    def vertical_osc_stats(results):
        if not results:
            return None
        flagged = sum(1 for r in results if r.get("excessive"))
        avg_osc = np.mean([r["normalized_oscillation"] for r in results])
        return {"strides_checked": len(results), "flagged_excessive": flagged,
                "pct_flagged": round(100 * flagged / len(results), 1),
                "avg_normalized_osc": round(float(avg_osc), 3)}

    summary["vertical_oscillation"] = {
        "right": vertical_osc_stats(vertical_osc_R),
        # "left": vertical_osc_stats(vertical_osc_L),
    }

    # --- Trunk lean ---
    excess_fwd = int(np.sum(trunk_lean_deg > trunk_lean_threshold))
    excess_bwd = int(np.sum(trunk_lean_deg < -trunk_lean_threshold))
    summary["trunk_lean"] = {
        "mean_deg": round(float(np.mean(trunk_lean_deg)), 1),
        "max_forward_deg": round(float(np.max(trunk_lean_deg)), 1),
        "max_backward_deg": round(float(np.min(trunk_lean_deg)), 1),
        "pct_frames_excess_forward": round(100 * excess_fwd / len(trunk_lean_deg), 1),
        "pct_frames_excess_backward": round(100 * excess_bwd / len(trunk_lean_deg), 1),
    }

    # --- Cadence ---
    summary["cadence"] = {
        "spm": round(cadence_spm, 1),
        "low_cadence": cadence_spm < low_cadence_threshold,
    }

    # --- Kickback / heel whip ---
    def kickback_stats(results):
        if not results:
            return None
        flagged = sum(1 for r in results if r["kickback_score"] < low_kickback_threshold)
        avg_score = np.mean([r["kickback_score"] for r in results])
        return {"swings_checked": len(results), "flagged_shuffling": flagged,
                "avg_score": round(float(avg_score), 2)}

    summary["kickback"] = {
        "right": kickback_stats(kickback_R),
        "left": kickback_stats(kickback_L),
    }

    return summary