import os
import glob
import cv2
import numpy as np
import mediapipe as mp

def process_video(video_path, pose):
    """
    Extract 33 pose keypoints (x, y, z) per frame from a video using MediaPipe Pose.
    
    Returns:
        np.ndarray of shape (frames, 33, 3) if pose keypoints are detected, else None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_keypoints = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe requires RGB images
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract (x, y, z) for all 33 landmarks
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                ) # Shape: (33, 3)
                frame_keypoints.append(landmarks)
            else:
                # If no landmark detected in this frame, append zeros to keep array structure
                frame_keypoints.append(np.zeros((33, 3), dtype=np.float32))

    finally:
        cap.release()

    if not frame_keypoints:
        return None

    keypoints_array = np.array(frame_keypoints, dtype=np.float32) # Shape: (frames, 33, 3)
    
    # Check if any pose was detected across all frames (non-zero array)
    if not np.any(keypoints_array):
        return None

    return keypoints_array


def extract_keypoints_from_videos(video_dir="data/videos", output_dir="data/keypoints"):
    """
    Process all .mp4 videos in video_dir and save keypoints to output_dir.
    Skips already processed videos and videos with no detected pose.
    Handles corrupted video files gracefully.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Find all .mp4 files and deduplicate paths
    raw_files = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.MP4"))
    seen = set()
    video_files = []
    for f in raw_files:
        norm_f = os.path.normcase(os.path.abspath(f))
        if norm_f not in seen:
            seen.add(norm_f)
            video_files.append(f)
    video_files = sorted(video_files)

    if not video_files:
        print(f"No .mp4 video files found in '{video_dir}'.")
        return

    print(f"Found {len(video_files)} video(s) in '{video_dir}'. Starting keypoint extraction...")

    # Initialize MediaPipe Pose instance once for performance
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for idx, video_path in enumerate(video_files, start=1):
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]
            output_file = os.path.join(output_dir, f"{base_name}.npy")

            print(f"[{idx}/{len(video_files)}] Processing '{video_name}'...")

            # Skip if already processed
            if os.path.exists(output_file):
                print(f"  -> Skipped: Output file '{output_file}' already exists.")
                continue

            try:
                keypoints = process_video(video_path, pose)

                if keypoints is None:
                    print(f"  -> Skipped: No pose detected in '{video_name}'.")
                    continue

                np.save(output_file, keypoints)
                print(f"  -> Saved keypoints with shape {keypoints.shape} to '{output_file}'.")

            except Exception as e:
                print(f"  -> Error processing '{video_name}': {e}. Skipping video.")


if __name__ == "__main__":
    extract_keypoints_from_videos()
