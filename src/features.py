import numpy as np

# MediaPipe Pose Landmark Indices
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_KNEE_IDX = 25
RIGHT_KNEE_IDX = 26
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28


def calculate_joint_angle(A, B, C):
    """
    Calculate 3D joint angle (in degrees) at vertex B formed by points A, B, C.
    
    Args:
        A, B, C: np.ndarray of shape (T, 3) representing 3D coordinates over T frames.
        
    Returns:
        np.ndarray of shape (T,) containing joint angles in degrees.
    """
    BA = A - B
    BC = C - B
    
    norm_ba = np.linalg.norm(BA, axis=-1, keepdims=True) + 1e-8
    norm_bc = np.linalg.norm(BC, axis=-1, keepdims=True) + 1e-8
    
    cosine = np.sum(BA * BC, axis=-1, keepdims=True) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    
    angle = np.arccos(cosine) * (180.0 / np.pi)
    return angle.squeeze(-1)


def extract_window_features(window):
    """
    Compute kinematic features for a single 60-frame window.
    
    Args:
        window (np.ndarray): Keypoints array of shape (T, 33, 3).
        
    Returns:
        np.ndarray: 1D feature vector of summary statistics.
    """
    # 1. Joint Positions over time (T, 3)
    l_shoulder = window[:, LEFT_SHOULDER_IDX, :]
    r_shoulder = window[:, RIGHT_SHOULDER_IDX, :]
    l_hip = window[:, LEFT_HIP_IDX, :]
    r_hip = window[:, RIGHT_HIP_IDX, :]
    l_knee = window[:, LEFT_KNEE_IDX, :]
    r_knee = window[:, RIGHT_KNEE_IDX, :]
    l_ankle = window[:, LEFT_ANKLE_IDX, :]
    r_ankle = window[:, RIGHT_ANKLE_IDX, :]

    # 2. Compute Joint Angles (T,)
    l_knee_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
    r_knee_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
    
    l_hip_angle = calculate_joint_angle(l_shoulder, l_hip, l_knee)
    r_hip_angle = calculate_joint_angle(r_shoulder, r_hip, r_knee)

    # 3. Compute Left/Right Asymmetry (T,)
    knee_asymmetry = np.abs(l_knee_angle - r_knee_angle)
    hip_asymmetry = np.abs(l_hip_angle - r_hip_angle)

    # 4. Compute Ankle Distance / Stride Proxy (T,)
    ankle_distance = np.linalg.norm(l_ankle - r_ankle, axis=-1)

    # 5. Extract Mean and Std for each metric across the window
    metrics = [
        l_knee_angle,
        r_knee_angle,
        l_hip_angle,
        r_hip_angle,
        knee_asymmetry,
        hip_asymmetry,
        ankle_distance
    ]

    features = []
    for m in metrics:
        features.append(np.mean(m))
        features.append(np.std(m))

    return np.array(features, dtype=np.float32)


def extract_dataset_features(X_dataset):
    """
    Compute features for an entire dataset split of windows.
    
    Args:
        X_dataset (np.ndarray): Keypoints array of shape (N_windows, T, 33, 3).
        
    Returns:
        np.ndarray: Feature matrix of shape (N_windows, N_features).
    """
    if X_dataset.ndim != 4:
        raise ValueError(f"Expected X_dataset shape (N, T, 33, 3), got shape {X_dataset.shape}")

    feature_matrix = []
    for i in range(X_dataset.shape[0]):
        feat = extract_window_features(X_dataset[i])
        feature_matrix.append(feat)

    if not feature_matrix:
        return np.empty((0, 14), dtype=np.float32)

    return np.array(feature_matrix, dtype=np.float32)


FEATURE_NAMES = [
    "left_knee_angle_mean", "left_knee_angle_std",
    "right_knee_angle_mean", "right_knee_angle_std",
    "left_hip_angle_mean", "left_hip_angle_std",
    "right_hip_angle_mean", "right_hip_angle_std",
    "knee_asymmetry_mean", "knee_asymmetry_std",
    "hip_asymmetry_mean", "hip_asymmetry_std",
    "stride_distance_mean", "stride_variability_std"
]


if __name__ == "__main__":
    dummy_window = np.random.randn(60, 33, 3)
    feats = extract_window_features(dummy_window)
    print("Features extracted successfully for 1 window:")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:25s}: {val:.4f}")
