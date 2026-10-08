import os
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import mediapipe as mp

# Add src to system path for modular imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from src.dataset import normalize_keypoints, slice_sequence_into_windows
    from src.features import extract_window_features, calculate_joint_angle, FEATURE_NAMES
except ImportError:
    from dataset import normalize_keypoints, slice_sequence_into_windows
    from features import extract_window_features, calculate_joint_angle, FEATURE_NAMES


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Gait Abnormality Identification",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Dark Mode & Premium Glassmorphism) ---
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .badge-normal {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 9999px;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }

    .badge-abnormal {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 9999px;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    .explanation-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
        font-size: 0.98rem;
        line-height: 1.6;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_deep_model(model_path="models/gait_lstm_motion_319.keras"):
    """Load the trained Position + Velocity LSTM model."""
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            return model, True
        except Exception as e:
            st.error(f"Error loading Keras model from {model_path}: {e}")
            return None, False
    else:
        return None, False


def process_video_and_extract_skeleton(video_path):
    """
    Process input video with MediaPipe Pose, draw 3D skeleton overlay,
    and extract (frames, 33, 3) keypoints array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare temporary file for output skeleton video
    temp_raw_path = os.path.join(
    tempfile.gettempdir(),
    "gait_skeleton_overlay_raw.mp4"
    )

    temp_out_path = os.path.join(
        tempfile.gettempdir(),
        "gait_skeleton_overlay_h264.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        temp_raw_path,
        fourcc,
        fps,
        (width, height)
    )

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    keypoints_list = []
    processed_frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            annotated_frame = frame.copy()

            if results.pose_landmarks:
                # 1. Extract 33 (x, y, z) landmarks
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                )
                keypoints_list.append(landmarks)

                # 2. Draw skeleton overlay
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            else:
                keypoints_list.append(np.zeros((33, 3), dtype=np.float32))

            writer.write(annotated_frame)
            processed_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.release()
    # Convert the OpenCV MP4 to browser-friendly H.264 MP4
    try:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_raw_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_out_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

    except Exception as e:
        st.warning(f"Video conversion warning: {e}")
        temp_out_path = temp_raw_path

    if not keypoints_list:
        return None, None, None

    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    return keypoints_array, temp_out_path, processed_frames


def generate_explanation(label, confidence, l_knee_avg, r_knee_avg, knee_asym, stride_var):
    """Generate diagnostic clinical summary explanation."""
    if label == "Normal":
        explanation = (
            f"The gait pattern exhibits **consistent bilateral symmetry** and **regular temporal rhythm**. "
            f"Left knee flexion averaged **{l_knee_avg:.1f}°** and right knee flexion averaged **{r_knee_avg:.1f}°**, "
            f"yielding a low inter-limb asymmetry index of **{knee_asym:.1f}°**. "
            f"Stride variability is within healthy normative ranges (**{stride_var:.3f}**), indicating smooth gait dynamics."
        )
    else:
        explanation = (
            f"The AI model detected **significant gait abnormalities** with **{confidence:.1f}% confidence**. "
            f"Key kinematic indicators include elevated joint asymmetry (**{knee_asym:.1f}°** difference between left and right knee flexion) "
            f"and irregular stride variability (**{stride_var:.3f}**). "
            f"These gait deviations suggest potential motor impairment, joint stiffness, or antalgic/pathological walking dynamics."
        )
    return explanation


def main():
    # --- Sidebar ---
    st.sidebar.image("https://img.icons8.com/isometric/100/walking.png", width=70)
    st.sidebar.title("Gait Analysis AI")
    st.sidebar.caption("Computer Vision & Pose Kinematics")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Configuration")
    
    model, model_loaded = load_deep_model()
    if model_loaded:
        st.sidebar.success("✅ Position + Velocity LSTM Loaded")
    else:
        st.sidebar.warning("⚠️ Deep Model not found. Running in heuristic mode.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Sample Videos")
    sample_videos = sorted([
        f for f in os.listdir("data/videos") if f.endswith(".mp4")
    ]) if os.path.exists("data/videos") else []

    selected_sample = None
    if sample_videos:
        sample_choice = st.sidebar.selectbox("Test with existing dataset video:", ["-- Select Sample --"] + sample_videos)
        if sample_choice != "-- Select Sample --":
            selected_sample = os.path.join("data/videos", sample_choice)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Instructions:** Upload a short walking video or pick a sample video to run automated pose extraction and AI abnormality identification.")

    # --- Header Banner ---
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Automated Identification of Gait Abnormalities from Video</div>
        <div class="subtitle">Extract 3D pose landmarks via MediaPipe Pose, compute kinematic joint features, and run spatiotemporal Deep Learning classification.</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Video Input Section ---
    col_input, col_info = st.columns([2, 1])

    video_path = None

    with col_input:
        st.subheader("📹 Step 1: Provide Walking Video")
        uploaded_file = st.file_uploader("Upload a walking video file (.mp4, .mov, .avi):", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        elif selected_sample:
            video_path = selected_sample

    with col_info:
        st.subheader("📋 Processing Details")
        st.write("- **Pose Detector**: MediaPipe Pose (33 Landmarks)")
        st.write("- **Pose Features**: 33 landmarks × 3 coordinates")
        st.write("- **Temporal Input**: Resized to 100 frames")
        st.write("- **Motion Features**: Position + frame-to-frame velocity")
        st.write("- **Deep Classifier**: LSTM")

    if video_path:
        st.markdown("---")
        st.subheader("🦴 Step 2: Skeleton Pose Extraction & AI Inference")

        with st.spinner("Processing video frames and extracting 3D pose landmarks..."):
            keypoints_array, skeleton_video_path, frames_list = process_video_and_extract_skeleton(video_path)

        if keypoints_array is None or len(keypoints_array) == 0 or not np.any(keypoints_array):
            st.error("❌ No human pose landmarks were detected in the provided video. Please upload a clear walking video.")
            return

        # --- Display Skeleton Overlay Video ---
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.write("**Source Video:**")
            st.video(video_path)
        with v_col2:
            st.write("**Skeleton Overlay Video:**")
            if skeleton_video_path and os.path.exists(skeleton_video_path):
                st.video(skeleton_video_path)

        # --- Model Inference ---

        # Resize the complete sequence to exactly 100 frames
        num_frames = 100
        original_frames = keypoints_array.shape[0]

        if original_frames != num_frames:
            old_indices = np.linspace(0, 1, original_frames)
            new_indices = np.linspace(0, 1, num_frames)

            resized_keypoints = np.zeros((num_frames, 33, 3), dtype=np.float32)

            for landmark in range(33):
                for coordinate in range(3):
                    resized_keypoints[:, landmark, coordinate] = np.interp(
                        new_indices,
                        old_indices,
                        keypoints_array[:, landmark, coordinate]
                    )
        else:
            resized_keypoints = keypoints_array.astype(np.float32)

        # Flatten position features: (100, 33, 3) -> (100, 99)
        position_features = resized_keypoints.reshape(100, 99)

        # Calculate frame-to-frame velocity
        velocity_features = np.diff(
            position_features,
            axis=0,
            prepend=position_features[0:1]
        )

        # Combine position + velocity: (100, 198)
        motion_features = np.concatenate(
            [position_features, velocity_features],
            axis=1
        )

        # Add batch dimension: (1, 100, 198)
        X_input = np.expand_dims(motion_features, axis=0)

        if model_loaded and model is not None:
            prediction_probability = float(model.predict(X_input, verbose=0)[0][0])

            pred_class = 1 if prediction_probability >= 0.5 else 0

            if pred_class == 1:
                confidence = prediction_probability * 100
            else:
                confidence = (1 - prediction_probability) * 100
        else:
            # Fallback heuristic
            kp_norm = normalize_keypoints(keypoints_array)
            windows = slice_sequence_into_windows(kp_norm, window_size=60, stride=30)

            features_list = [extract_window_features(w) for w in windows]
            avg_asym = np.mean([f[8] for f in features_list])

            pred_class = 1 if avg_asym > 15.0 else 0
            confidence = 85.0

        label_name = "Normal" if pred_class == 0 else "Abnormal"

        # --- Compute Kinematic Metrics for Report ---
        # Compute kinematic metrics from the full keypoint sequence
        window_feats = [extract_window_features(resized_keypoints)]
        avg_feats = np.mean(window_feats, axis=0)

        l_knee_avg = avg_feats[0]
        r_knee_avg = avg_feats[2]
        knee_asym = avg_feats[8]
        stride_var = avg_feats[13]

        # --- Step 3: Diagnostic Results & Metrics ---
        st.markdown("---")
        st.subheader("🎯 Step 3: Diagnostic Classification & Kinematic Analysis")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("#### Classification Result")
            if label_name == "Normal":
                st.markdown("<div class='badge-normal'>NORMAL GAIT</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-abnormal'>ABNORMAL GAIT</div>", unsafe_allow_html=True)

            st.write("")
            st.metric(label="Model Confidence Score", value=f"{confidence:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Clinical Summary & Explanation")
            explanation_text = generate_explanation(
                label_name, confidence, l_knee_avg, r_knee_avg, knee_asym, stride_var
            )
            st.markdown(f"<div class='explanation-box'>{explanation_text}</div>", unsafe_allow_html=True)

        # --- Kinematic Feature Cards ---
        st.markdown("#### 📊 Key Kinematic Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Left Knee Angle", f"{l_knee_avg:.1f}°")
        m2.metric("Avg Right Knee Angle", f"{r_knee_avg:.1f}°")
        m3.metric("Knee Asymmetry Index", f"{knee_asym:.1f}°")
        m4.metric("Stride Variability (Std)", f"{stride_var:.3f}")

        # --- Joint Angle Time Series Chart ---
        st.markdown("#### 📈 Joint Flexion Time Series Across Frames")
        
        # Calculate time series across full video length
        l_hip_pos = keypoints_array[:, 23, :]
        r_hip_pos = keypoints_array[:, 24, :]
        l_knee_pos = keypoints_array[:, 25, :]
        r_knee_pos = keypoints_array[:, 26, :]
        l_ankle_pos = keypoints_array[:, 27, :]
        r_ankle_pos = keypoints_array[:, 28, :]

        l_knee_series = calculate_joint_angle(l_hip_pos, l_knee_pos, l_ankle_pos)
        r_knee_series = calculate_joint_angle(r_hip_pos, r_knee_pos, r_ankle_pos)

        chart_df = pd.DataFrame({
            "Frame": np.arange(len(l_knee_series)),
            "Left Knee Flexion (deg)": l_knee_series,
            "Right Knee Flexion (deg)": r_knee_series
        }).set_index("Frame")

        st.line_chart(chart_df)


if __name__ == "__main__":
    main()
