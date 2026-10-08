# Automated Identification of Gait Abnormalities from Video

An end-to-end machine learning system for identifying **Normal** and
**Abnormal** gait patterns from video using human pose estimation,
temporal motion features, an LSTM classifier, and a Streamlit interface.

## Problem Statement

This project aims to automate the classification of gait sequences into
**Normal** and **Abnormal** categories using pose information extracted
from gait videos. The approach focuses on temporal gait movement rather
than only individual frames.

## Dataset

The project uses gait sequences derived from the **GAVD (Gait
Abnormality Video Dataset)**.

The final working dataset contains:

-   **319 usable gait sequences**
-   **156 Normal samples**
-   **163 Abnormal samples**
-   33 human pose landmarks per frame
-   3 coordinates per landmark
-   100 frames per input sequence after temporal resizing

The dataset itself is not included in the repository.

## Methodology

``` text
Gait Video
    ↓
MediaPipe Pose
    ↓
33 landmarks × 3 coordinates
    ↓
Temporal resizing to 100 frames
    ↓
99 Position Features
    +
99 Frame-to-Frame Velocity Features
    ↓
198 Features per Frame
    ↓
LSTM Classifier
    ↓
Normal / Abnormal
```

### Pose Extraction

MediaPipe Pose detects 33 human body landmarks. With three coordinate
values per landmark, each frame contains **99 position features**.

### Temporal Preprocessing

Each sequence is temporally interpolated to exactly **100 frames**,
providing a fixed-size input for the LSTM.

### Motion Feature Extraction

Frame-to-frame velocity is calculated as:

``` text
Velocity(t) = Position(t) - Position(t-1)
```

The 99 position features and 99 velocity features are concatenated:

``` text
99 position + 99 velocity = 198 features/frame
```

The final classifier input is therefore **100 × 198**.

### LSTM Classifier

The final model contains:

-   LSTM layer with 64 units
-   Dropout
-   LSTM layer with 32 units
-   Dropout
-   Dense layer with 16 units
-   Dropout
-   Sigmoid output layer

The output represents the probability of the **Abnormal** class.

## Baseline and Final Model

A position-only LSTM was evaluated as a baseline. The final model adds
frame-to-frame velocity to explicitly represent temporal movement.

  ---------------------------------------------------------------------------
  Model                 Accuracy      Precision         Recall       F1 Score
  --------------- -------------- -------------- -------------- --------------
  Position-only           68.75%         63.64%         87.50%         73.68%
  LSTM                                                         

  **Position +        **78.12%**     **76.47%**     **81.25%**     **78.79%**
  Velocity LSTM**                                              
  ---------------------------------------------------------------------------

The Position + Velocity model was selected as the final model because it
achieved better test accuracy and F1 score.

### Final Test Confusion Matrix

``` text
                 Predicted
               Normal  Abnormal
Actual Normal     12       4
       Abnormal    3      13
```

## Final Model

``` text
models/gait_lstm_motion_319.keras
```

Input shape:

``` text
(None, 100, 198)
```

Output shape:

``` text
(None, 1)
```

The trained model is loaded directly by the Streamlit application;
retraining is not required to run the demo.

## Streamlit Application

The application:

1.  Accepts a gait video.
2.  Extracts pose landmarks using MediaPipe.
3.  Resizes the sequence to 100 frames.
4.  Computes frame-to-frame velocity.
5.  Combines position and velocity features.
6.  Loads the trained Keras model.
7.  Predicts Normal or Abnormal gait.
8.  Displays prediction confidence and gait-related measurements.
9.  Generates a pose/skeleton overlay video.

## Project Structure

``` text
gait-abnormality/
│
├── app.py
├── src/
├── models/
│   └── gait_lstm_motion_319.keras
├── results/
├── notebooks/
├── Gait_Abnormality_Training.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

Dataset files, virtual environments, caches, and temporary development
files are excluded from version control.

## Requirements

Recommended environment:

-   Python 3.12
-   TensorFlow 2.17.1
-   MediaPipe 0.10.21
-   NumPy 1.26.4
-   OpenCV 4.11.0.86
-   Streamlit 1.41.0
-   Protobuf 4.25.9

All required packages are listed in `requirements.txt`.

## Installation

### 1. Clone the repository

``` bash
git clone <repository-url>
cd gait-abnormality
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 4. Verify the model

Make sure this file exists:

``` text
models/gait_lstm_motion_319.keras
```

## Running the Application

From the project root:

``` powershell
streamlit run app.py
```

Upload a gait video through the interface to obtain the prediction and
visualization.

## Training Notebook

The training workflow is documented in `Gait_Abnormality_Training.ipynb`
and covers:

-   Dataset preparation
-   Pose sequence preprocessing
-   Position-only LSTM baseline
-   Position + velocity feature construction
-   Train/validation/test splitting
-   LSTM architecture
-   Model training
-   Evaluation metrics
-   Confusion matrix
-   Training curves
-   Final model saving

## Limitations

This is an academic machine-learning prototype evaluated on 319 usable
sequences. The current experiment uses a stratified
train/validation/test split rather than a group-aware split, so the
reported performance should be interpreted as an experimental result
rather than a clinical or population-level estimate.

The system is intended for academic demonstration and is **not a
clinical diagnostic system**.

## Conclusion

The project demonstrates an end-to-end approach for automated gait
abnormality identification using pose estimation and temporal deep
learning.

Adding frame-to-frame velocity improved performance over the
position-only baseline. The final Position + Velocity LSTM achieved:

-   **78.12% Accuracy**
-   **76.47% Precision**
-   **81.25% Recall**
-   **78.79% F1 Score**

The trained model is integrated into a working Streamlit application for
video-based inference and visualization.

## Author

**Harsh Raj**

Mini-Project --- UE24CS352A: Machine Learning
