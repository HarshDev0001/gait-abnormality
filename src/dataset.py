import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# MediaPipe Pose Landmark Indices for Hips
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

LABEL_MAP = {
    'normal': 0,
    'abnormal': 1,
    'Normal': 0,
    'Abnormal': 1,
    'Normal Gait': 0,
    'Abnormal Gait': 1
}

def normalize_keypoints(keypoints):
    """
    Normalize 3D pose keypoints relative to the hip midpoint.
    
    Args:
        keypoints (np.ndarray): Keypoints array of shape (T, 33, 3)
        
    Returns:
        np.ndarray: Hip-centered keypoints array of shape (T, 33, 3)
    """
    # Calculate hip midpoint per frame: shape (T, 3)
    left_hip = keypoints[:, LEFT_HIP_IDX, :]
    right_hip = keypoints[:, RIGHT_HIP_IDX, :]
    hip_midpoint = (left_hip + right_hip) / 2.0
    
    # Expand dims for broadcasting: shape (T, 1, 3)
    hip_midpoint_expanded = np.expand_dims(hip_midpoint, axis=1)
    
    # Subtract hip midpoint from all keypoints
    normalized_keypoints = keypoints - hip_midpoint_expanded
    return normalized_keypoints


def slice_sequence_into_windows(keypoints, window_size=60, stride=30):
    """
    Cut a sequence of keypoints into fixed-length sliding windows.
    
    Args:
        keypoints (np.ndarray): Keypoint array of shape (T, 33, 3)
        window_size (int): Number of frames per window (default 60)
        stride (int): Step size between consecutive windows (default 30)
        
    Returns:
        list of np.ndarray: List of windows, each of shape (window_size, 33, 3)
    """
    num_frames = keypoints.shape[0]
    windows = []
    
    if num_frames < window_size:
        # Pad sequence with edge padding if shorter than window_size
        pad_len = window_size - num_frames
        padded = np.pad(keypoints, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
        windows.append(padded)
    else:
        for start in range(0, num_frames - window_size + 1, stride):
            window = keypoints[start : start + window_size]
            windows.append(window)
            
    return windows


def load_raw_dataset(labels_path="data/labels.csv", keypoints_dir="data/keypoints"):
    """
    Load keypoint files along with labels metadata.
    
    Returns:
        pd.DataFrame: Dataframe containing matched video_id, keypoints array, label, and group_id.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file '{labels_path}' not found.")
        
    labels_df = pd.read_csv(labels_path)
    
    records = []
    for _, row in labels_df.iterrows():
        vid_id = row['video_id']
        label_str = str(row['label']).strip()
        label_int = LABEL_MAP.get(label_str, 1 if 'abnormal' in label_str.lower() else 0)
        
        # Determine group_id for leak-free splitting (subject_id > video_id)
        subject_id = row.get('subject_id', None)
        if pd.isna(subject_id) or str(subject_id).strip() == "":
            # Fallback to base video ID (extract youtube id if possible)
            group_id = vid_id
        else:
            group_id = str(subject_id).strip()
            
        npy_path = os.path.join(keypoints_dir, f"{vid_id}.npy")
        if os.path.exists(npy_path):
            keypoints = np.load(npy_path)
            records.append({
                'video_id': vid_id,
                'keypoints': keypoints,
                'label': label_int,
                'group_id': group_id
            })
        else:
            print(f"Warning: Keypoint file '{npy_path}' not found. Skipping.")
            
    return pd.DataFrame(records)


def prepare_gait_dataset(
    labels_path="data/labels.csv",
    keypoints_dir="data/keypoints",
    window_size=60,
    stride=30,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Load keypoints, perform leak-free group split (70/15/15), normalize relative to hips,
    and slice into fixed-length windows.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
        Shapes: X_*: (N_windows, 60, 33, 3), y_*: (N_windows,)
    """
    raw_df = load_raw_dataset(labels_path, keypoints_dir)
    if raw_df.empty:
        raise ValueError("No valid keypoint records loaded.")
        
    groups = raw_df['group_id'].values
    
    # First split: Train vs (Val + Test)
    test_val_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(raw_df, raw_df['label'], groups=groups))
    
    train_df = raw_df.iloc[train_idx].copy()
    temp_df = raw_df.iloc[temp_idx].copy()
    
    # Second split: Val vs Test
    relative_test_ratio = test_ratio / test_val_ratio
    temp_groups = temp_df['group_id'].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df['label'], groups=temp_groups))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    def process_split_df(df):
        X_list = []
        y_list = []
        
        for _, row in df.iterrows():
            kp = row['keypoints']
            lbl = row['label']
            
            # 1. Normalize relative to hip midpoint
            kp_norm = normalize_keypoints(kp)
            
            # 2. Slice into windows
            windows = slice_sequence_into_windows(kp_norm, window_size=window_size, stride=stride)
            
            for w in windows:
                X_list.append(w)
                y_list.append(lbl)
                
        if not X_list:
            return np.empty((0, window_size, 33, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
            
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int64)
        return X_arr, y_arr

    X_train, y_train = process_split_df(train_df)
    X_val, y_val = process_split_df(val_df)
    X_test, y_test = process_split_df(test_df)
    
    print(f"Dataset Split Summary:")
    print(f"  - Train: {X_train.shape[0]} windows, shape {X_train.shape} (Labels: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)})")
    print(f"  - Val:   {X_val.shape[0]} windows, shape {X_val.shape} (Labels: 0={np.sum(y_val==0)}, 1={np.sum(y_val==1)})")
    print(f"  - Test:  {X_test.shape[0]} windows, shape {X_test.shape} (Labels: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# PyTorch Dataset wrapper if PyTorch is available
try:
    import torch
    from torch.utils.data import Dataset

    class GaitDataset(Dataset):
        """PyTorch Dataset for Gait Abnormality 3D pose sequence windows."""
        def __init__(self, X, y, flatten_features=False):
            """
            Args:
                X (np.ndarray): Keypoints array of shape (N, 60, 33, 3)
                y (np.ndarray): Labels array of shape (N,)
                flatten_features (bool): If True, flattens (33, 3) to (99,) per frame.
            """
            if flatten_features and X.ndim == 4:
                # Shape: (N, 60, 99)
                X = X.reshape(X.shape[0], X.shape[1], -1)
                
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

except ImportError:
    pass


if __name__ == "__main__":
    print("Testing src/dataset.py data loading and splitting pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_gait_dataset()
