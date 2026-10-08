import os
import glob
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import yt_dlp

def load_gavd_annotations(gavd_dir="external/GAVD/data"):
    """
    Load and aggregate GAVD CSV annotation files into sequence-level metadata.
    """
    csv_files = sorted(glob.glob(os.path.join(gavd_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No GAVD annotation CSV files found in '{gavd_dir}'.")
    
    print(f"Loading {len(csv_files)} GAVD annotation CSV file(s)...")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    # Map dataset label to normalized 'normal' vs 'abnormal'
    df_all['label'] = df_all['dataset'].apply(
        lambda x: 'normal' if str(x).strip().lower() == 'normal gait' else 'abnormal'
    )

    # Aggregate annotations per sequence (seq)
    seq_gb = df_all.groupby('seq')
    
    seq_df = seq_gb.agg(
        youtube_id=('id', 'first'),
        source_url=('url', 'first'),
        label=('label', 'first'),
        gait_pat=('gait_pat', 'first'),
        min_frame=('frame_num', 'min'),
        max_frame=('frame_num', 'max'),
        total_frames=('frame_num', 'count')
    ).reset_index()

    # Check for subject_id column if present in dataset
    if 'subject_id' in df_all.columns:
        sub_map = df_all.groupby('seq')['subject_id'].first()
        seq_df['subject_id'] = seq_df['seq'].map(sub_map)
    else:
        seq_df['subject_id'] = ""

    return seq_df


def select_balanced_subset(seq_df, total_target=150):
    """
    Select a balanced subset of sequences (equal normal and abnormal).
    For abnormal samples, stratify across pathology types (gait_pat) for diversity.
    """
    target_per_class = total_target // 2

    normal_df = seq_df[seq_df['label'] == 'normal'].copy()
    abnormal_df = seq_df[seq_df['label'] == 'abnormal'].copy()

    # Select normal sequences (sample evenly across unique YouTube videos)
    if len(normal_df) > target_per_class:
        normal_sampled = normal_df.sample(n=target_per_class, random_state=42)
    else:
        normal_sampled = normal_df

    # Select abnormal sequences (stratified sample across gait_pat classes)
    if len(abnormal_df) > target_per_class:
        # Group sample by gait_pat to ensure balanced pathology coverage
        gait_pats = abnormal_df['gait_pat'].unique()
        per_pat_count = max(1, target_per_class // len(gait_pats))
        
        sampled_parts = []
        for pat in gait_pats:
            pat_df = abnormal_df[abnormal_df['gait_pat'] == pat]
            n_samples = min(len(pat_df), per_pat_count)
            sampled_parts.append(pat_df.sample(n=n_samples, random_state=42))
        
        abnormal_sampled = pd.concat(sampled_parts, ignore_index=True)
        # Fill remaining if needed
        remaining_needed = target_per_class - len(abnormal_sampled)
        if remaining_needed > 0:
            remaining_pool = abnormal_df[~abnormal_df['seq'].isin(abnormal_sampled['seq'])]
            if len(remaining_pool) > 0:
                additional = remaining_pool.sample(
                    n=min(remaining_needed, len(remaining_pool)), random_state=42
                )
                abnormal_sampled = pd.concat([abnormal_sampled, additional], ignore_index=True)
    else:
        abnormal_sampled = abnormal_df

    balanced_df = pd.concat([normal_sampled, abnormal_sampled], ignore_index=True)
    print(f"Selected subset: {len(normal_sampled)} 'normal' sequences, {len(abnormal_sampled)} 'abnormal' sequences.")
    return balanced_df


def download_and_trim_video(url, output_path, start_frame, end_frame):
    """
    Download raw video via yt-dlp at max 480p, trim using OpenCV to start_frame:end_frame,
    and save audio-free video file to output_path.
    """
    temp_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    raw_template = os.path.join(temp_dir, f"temp_raw_{base_name}.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/best[height<=480]/best',
        'outtmpl': raw_template,
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 15,
        'retries': 3,
    }

    raw_file = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded raw file
        possible_files = glob.glob(os.path.join(temp_dir, f"temp_raw_{base_name}.*"))
        if not possible_files:
            raise FileNotFoundError(f"yt-dlp completed but raw video file not found for {url}")
        raw_file = possible_files[0]

        # Open raw video with OpenCV
        cap = cv2.VideoCapture(raw_file)
        if not cap.isOpened():
            raise ValueError("OpenCV failed to open downloaded raw video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 30.0

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate target dimensions (480p max)
        if orig_h > 480:
            target_h = 480
            target_w = int(orig_w * (480.0 / orig_h))
            # Ensure width is even for codec compatibility
            if target_w % 2 != 0:
                target_w -= 1
        else:
            target_w, target_h = orig_w, orig_h

        # Set start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        curr_frame = start_frame
        frames_written = 0

        while cap.isOpened() and curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (target_w, target_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (target_w, target_h))

            writer.write(frame)
            frames_written += 1
            curr_frame += 1

        writer.release()
        cap.release()

        if frames_written == 0:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise ValueError(f"No frames written during trimming ({start_frame} to {end_frame}).")

    finally:
        # Cleanup temporary raw video file
        if raw_file and os.path.exists(raw_file):
            try:
                os.remove(raw_file)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Download and trim a balanced GAVD video subset using yt-dlp.")
    parser.add_argument("--gavd-dir", type=str, default="external/GAVD/data", help="Directory containing GAVD CSV files.")
    parser.add_argument("--output-dir", type=str, default="data/videos", help="Directory to save downloaded videos.")
    parser.add_argument("--labels-file", type=str, default="data/labels.csv", help="Path for labels CSV file.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for failed download log.")
    parser.add_argument("--total-videos", type=int, default=150, help="Total target subset size (equal normal & abnormal).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_file), exist_ok=True)

    failed_log_path = os.path.join(args.results_dir, "failed_downloads.txt")

    # Load annotations and select subset
    seq_df = load_gavd_annotations(args.gavd_dir)
    subset_df = select_balanced_subset(seq_df, total_target=args.total_videos)

    print(f"\nProcessing {len(subset_df)} video sequences...")

    labels_data = []
    
    # Load existing labels if file exists
    if os.path.exists(args.labels_file):
        existing_labels_df = pd.read_csv(args.labels_file)
        existing_vids = set(existing_labels_df['video_id'].dropna())
        labels_data = existing_labels_df.to_dict(orient='records')
    else:
        existing_vids = set()

    success_count = 0
    fail_count = 0

    for idx, row in subset_df.iterrows():
        seq_id = row['seq']
        yt_id = row['youtube_id']
        label = row['label']
        url = row['source_url']
        start_frame = int(row['min_frame'])
        end_frame = int(row['max_frame'])
        subject_id = row['subject_id']

        # Clear file ID format: <label>_<youtube_id>_<seq_id>
        file_id = f"{label}_{yt_id}_{seq_id}"
        output_path = os.path.join(args.output_dir, f"{file_id}.mp4")

        print(f"[{success_count + fail_count + 1}/{len(subset_df)}] {file_id} (Frames {start_frame}-{end_frame})...")

        # Check if already processed
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  -> Already exists. Skipping download.")
            success_count += 1
            if file_id not in existing_vids:
                labels_data.append({
                    'video_id': file_id,
                    'label': label,
                    'subject_id': subject_id,
                    'source_url': url
                })
                existing_vids.add(file_id)
            continue

        try:
            download_and_trim_video(url, output_path, start_frame, end_frame)
            print(f"  -> Successfully saved to '{output_path}'.")
            success_count += 1
            
            if file_id not in existing_vids:
                labels_data.append({
                    'video_id': file_id,
                    'label': label,
                    'subject_id': subject_id,
                    'source_url': url
                })
                existing_vids.add(file_id)

        except Exception as e:
            fail_count += 1
            err_msg = f"{file_id} | URL: {url} | Frames: {start_frame}-{end_frame} | Error: {e}\n"
            print(f"  -> FAILED: {e}")
            with open(failed_log_path, "a", encoding="utf-8") as f_log:
                f_log.write(err_msg)

        # Update labels.csv after each successful processing
        if labels_data:
            pd.DataFrame(labels_data).to_csv(args.labels_file, index=False)

    print(f"\nDownload batch complete!")
    print(f"  - Successfully processed: {success_count}")
    print(f"  - Failed/Unavailable: {fail_count} (Logged to '{failed_log_path}')")
    print(f"  - Labels saved to: '{args.labels_file}'")


if __name__ == "__main__":
    main()
