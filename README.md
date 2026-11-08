# AI Sign Language Recognition

Real-time hand gesture to text translation using a webcam. The browser captures video; MediaPipe detects a hand and extracts 21 landmarks, and a locally trained scikit-learn classifier maps normalized landmarks to a trained label. The Flask interface builds sentences and can speak them with a voice installed on the user's device.

The app does not call external APIs or third-party inference services. When deployed, webcam frames are sent over HTTPS to this project's own Render service for inference with the bundled model.

## Features

- Live webcam preview with MediaPipe hand landmark overlay
- Locally trained landmark classifier with smoothed predictions and confidence
- Sentence builder with add, delete, clear, copy, and translation history
- Offline device speech through the browser's speech synthesis
- Light and dark themes, responsive layout, and clear camera/model errors
- Scripts for collecting samples, training, and standalone webcam prediction

## Technology

Python 3.11, Flask, Gunicorn, OpenCV, MediaPipe, NumPy, pandas, scikit-learn, joblib, pyttsx3, HTML, CSS, and JavaScript.

## Installation

Use Python 3.10 or 3.11. MediaPipe wheels may not be available for the newest Python releases; in particular, Python 3.15 is not currently a supported runtime for this project.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, create and activate the environment with `python3.11 -m venv .venv` and `source .venv/bin/activate` instead.

The repository includes a trained model at `landmark_model/live_sign_classifier.joblib`, so a compatible Python environment and installed requirements are enough to try the application. Webcam permission is requested in the browser. The browser uses its installed speech voices; speech availability varies by device.

## Launch

From the project directory, with the virtual environment active:

```powershell
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Approve camera access in the browser when prompted. `PORT` can be set to change the local server port.

For the standalone OpenCV interface:

```powershell
python predict.py
```

## Deploy on Render

The repository includes `render.yaml` and `.python-version`. Create a Render Blueprint from the repository, or create a Python web service manually with these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120 main:app`
- Health check path: `/healthz`

Render supplies `PORT` and HTTPS. The browser requests webcam permission from the HTTPS page and sends resized JPEG frames to this app's `/api/frame` endpoint. Frames are processed by the included MediaPipe/scikit-learn model; no external inference API is used. Keep the one-worker setting because the current sentence and camera session state is held in process memory. The bundled model must be committed at `landmark_model/live_sign_classifier.joblib`; Render's filesystem is not a persistent place to collect training samples. To deploy a newly trained model, commit the updated model with the application.

## Collecting Gesture Data

Choose a concise label, such as `hello`, and collect varied examples in good lighting. The collector saves wrist-relative, scale-normalized 63-value landmark rows into `landmark_model/live_landmarks.csv`.

```powershell
python collect_data.py --label hello --samples 200
```

Repeat for every gesture to train. Vary hand angle, distance, and position. Only collect signs that can be consistently distinguished from the other labels. Existing labels are appended to the CSV.

## Training

The dataset needs at least two labels, with at least two original samples per label. Training splits original samples into train and held-out sets, then mirrors only the training samples for opposite-hand augmentation; this avoids evaluating on augmented copies of training data.

```powershell
python train_model.py
```

The trained model replaces `landmark_model/live_sign_classifier.joblib`. Keep a backup of the existing model before retraining if you need to preserve it. Review the printed held-out metrics; small or repetitive datasets can produce misleadingly high scores.

## Project Layout

- `main.py`: local Flask application entry point
- `src/app.py`: camera service, local inference, sentence and speech routes
- `src/landmarks.py`: 21-point extraction, normalization, model prediction
- `src/templates/` and `src/static/`: dashboard interface
- `collect_data.py`, `train_model.py`, `predict.py`: root workflow commands
- `src/collect_landmark_data.py`, `src/train_live_landmark_classifier.py`: collection and training implementations
- `landmark_model/`: landmark CSV and serialized trained models
- `dataset/`, `annotations/`, `sign_models/`: retained legacy image-detection experiment assets
- `tests/`: focused landmark tests

## Screenshots

No screenshots are currently checked into the repository. Add application captures here after running with a webcam and trained model.

## Limitations

- Recognition is limited to the classes in the loaded model; it does not translate arbitrary sign language or continuous signing.
- The included model's labels and accuracy depend on its original custom dataset. Recollect representative data for the target signer and lighting conditions.
- Mirroring is a simple augmentation strategy and does not guarantee equivalent left- and right-hand accuracy.
- Predictions classify hand poses, not temporal signs that require motion, facial expression, or body posture.
- A working browser webcam and supported MediaPipe/Python combination are required. On Render, frames are transmitted securely to the configured app service. Offline speech depends on browser/device voices.

## Future Enhancements

- Add temporal models for motion-based signs and phrase segmentation.
- Evaluate with signer-independent train/test splits and publish per-class metrics.
- Support configurable camera selection and sample management in the UI.
- Add accessible vocabulary guidance and multilingual offline voices.