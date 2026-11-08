import os
import threading
import time
from collections import Counter, deque
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request

from src.landmarks import extract_hand_landmarks, predict_landmarks


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "landmark_model", "live_sign_classifier.joblib")
CONFIDENCE_THRESHOLD = 0.35
LABEL_TEXT = {"iloveyou": "I love you", "clockit": "clock it"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


def display_label(label):
    return LABEL_TEXT.get(label, label)


class SpeechController:
    def __init__(self):
        self._lock = threading.Lock()
        self._engine = None
        self._thread = None
        self._stop_requested = threading.Event()
        self.status = "idle"
        self.error = None

    def speak(self, text):
        if not text.strip():
            raise ValueError("Build a sentence before using speech.")
        self.stop()
        self._stop_requested.clear()
        self.error = None
        self.status = "speaking"
        self._thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        self._thread.start()

    def _run(self, text):
        try:
            import pyttsx3

            if self._stop_requested.is_set():
                return
            engine = pyttsx3.init()
            with self._lock:
                self._engine = engine
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            if not self._stop_requested.is_set():
                self.status = "idle"
        except Exception as error:
            self.error = f"Offline speech could not start: {error}"
            self.status = "error"
        finally:
            with self._lock:
                self._engine = None

    def stop(self):
        self._stop_requested.set()
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        if self.status == "speaking":
            self.status = "stopped"


class GestureService:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.lock = threading.RLock()
        self.frame_lock = threading.Lock()
        self.cv2 = None
        self.np = None
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.model = None
        self.votes = deque(maxlen=8)
        self.active_label = None
        self.latest_frame = None
        self.sentence = []
        self.history = []
        self.consumed_events = set()
        self.next_event_id = 0
        self.state = {
            "camera_status": "stopped",
            "error": None,
            "prediction": None,
            "confidence": 0.0,
            "event_id": None,
            "handedness": None,
            "top_predictions": [],
        }
        self.speech = SpeechController()

    def start(self):
        with self.frame_lock:
            with self.lock:
                if self.state["camera_status"] == "running":
                    return self.snapshot()
                self.latest_frame = None
                self.state.update(camera_status="starting", error=None, prediction=None,
                                  confidence=0.0, event_id=None, handedness=None,
                                  top_predictions=[])

            try:
                import cv2
                import joblib
                import mediapipe as mp
                import numpy as np
            except ImportError as error:
                self._set_error(
                    f"Missing package '{error.name}'. Install dependencies with "
                    "python -m pip install -r requirements.txt."
                )
                return self.snapshot()

            if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
                self._set_error(
                    "Installed MediaPipe does not include the Hands API. Install the "
                    "project dependencies with python -m pip install -r requirements.txt."
                )
                return self.snapshot()

            if not os.path.isfile(self.model_path):
                self._set_error(
                    "Trained model not found. Collect landmark samples and run "
                    "python train_model.py first."
                )
                return self.snapshot()

            try:
                model = joblib.load(self.model_path)
                if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
                    raise ValueError("The saved model is not a compatible gesture classifier.")
                hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
            except Exception as error:
                self._set_error(f"Could not initialize gesture recognition: {error}")
                return self.snapshot()

            with self.lock:
                self.cv2 = cv2
                self.np = np
                self.hands = hands
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.model = model
                self.votes.clear()
                self.active_label = None
                self.state.update(camera_status="running", error=None)
                return self.snapshot()

    def stop(self):
        with self.frame_lock:
            hands = self.hands
            self.hands = None
            if hands is not None:
                hands.close()
            with self.lock:
                self.latest_frame = None
                self.votes.clear()
                self.active_label = None
                self.state.update(camera_status="stopped", prediction=None,
                                  confidence=0.0, event_id=None, handedness=None,
                                  top_predictions=[])
                return self.snapshot()

    def snapshot(self):
        with self.lock:
            return {
                **self.state,
                "sentence": list(self.sentence),
                "sentence_text": " ".join(display_label(word) for word in self.sentence),
                "history": list(self.history),
                "speech_status": self.speech.status,
                "speech_error": self.speech.error,
            }

    def _set_error(self, message):
        with self.lock:
            self.state.update(camera_status="error", error=message, prediction=None,
                              confidence=0.0, event_id=None, handedness=None,
                              top_predictions=[])

    def process_frame(self, frame_bytes):
        if not frame_bytes or len(frame_bytes) > 2 * 1024 * 1024:
            raise ValueError("The camera frame is empty or larger than 2 MB.")

        with self.frame_lock:
            with self.lock:
                if self.state["camera_status"] != "running" or self.hands is None:
                    raise RuntimeError("Start the camera before sending frames.")
                cv2 = self.cv2
                np = self.np
                hands = self.hands
                model = self.model

            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode the camera frame.")

            height, width = frame.shape[:2]
            scale = min(1.0, 640 / max(width, height))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detected_hands = results.multi_hand_landmarks or []

            if not detected_hands:
                self.votes.clear()
                self.active_label = None
                with self.lock:
                    self.state.update(prediction=None, confidence=0.0, event_id=None,
                                      handedness=None, top_predictions=[])
            else:
                hand = detected_hands[0]
                self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                features = extract_hand_landmarks(hand)
                if features is None:
                    self.votes.clear()
                    self.active_label = None
                else:
                    label, score, probabilities = predict_landmarks(model, features)
                    top = sorted(
                        zip(model.classes_, probabilities),
                        key=lambda item: float(item[1]), reverse=True
                    )[:3]
                    top_predictions = [
                        {"label": str(name), "text": display_label(str(name)),
                         "confidence": float(value)}
                        for name, value in top
                    ]
                    handedness = self._handedness(results)
                    if score >= CONFIDENCE_THRESHOLD:
                        self.votes.append((label, score))
                        common_label, count = Counter(item[0] for item in self.votes).most_common(1)[0]
                        if count >= 4:
                            stable_score = sum(
                                value for name, value in self.votes if name == common_label
                            ) / count
                            with self.lock:
                                if self.active_label != common_label:
                                    self.next_event_id += 1
                                    self.active_label = common_label
                                self.state.update(
                                    prediction=common_label,
                                    confidence=stable_score,
                                    event_id=self.next_event_id,
                                    handedness=handedness,
                                    top_predictions=top_predictions,
                                )
                        else:
                            with self.lock:
                                self.state.update(prediction=None, confidence=0.0,
                                                  event_id=None, handedness=handedness,
                                                  top_predictions=top_predictions)
                    else:
                        self.votes.clear()
                        self.active_label = None
                        with self.lock:
                            self.state.update(prediction=None, confidence=0.0,
                                              event_id=None, handedness=handedness,
                                              top_predictions=top_predictions)

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                raise RuntimeError("Could not encode the processed camera frame.")
            with self.lock:
                self.latest_frame = encoded.tobytes()
            return self.latest_frame

    @staticmethod
    def _handedness(results):
        if not results.multi_handedness:
            return "Hand detected"
        categories = results.multi_handedness[0].classification
        return categories[0].label if categories else "Hand detected"

    def add_word(self, event_id):
        with self.lock:
            if self.state["camera_status"] != "running" or self.state["prediction"] is None:
                raise ValueError("Hold a trained sign steady until it is recognized.")
            if event_id != self.state["event_id"] or event_id in self.consumed_events:
                raise ValueError("That sign was already added. Show it again to add it once more.")
            label = self.state["prediction"]
            self.sentence.append(label)
            self.consumed_events.add(event_id)
            self.history.insert(0, {
                "label": label,
                "text": display_label(label),
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            self.history = self.history[:12]
            return display_label(label)


service = GestureService()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def health_check():
    return jsonify(status="ok")


@app.get("/api/state")
def api_state():
    return jsonify(service.snapshot())


@app.post("/api/camera/start")
def camera_start():
    state = service.start()
    status_code = 200 if state["camera_status"] == "running" else 503
    return jsonify(state), status_code


@app.post("/api/camera/stop")
def camera_stop():
    return jsonify(service.stop())


@app.post("/api/frame")
def api_frame():
    try:
        service.process_frame(request.get_data(cache=False))
    except (ValueError, RuntimeError) as error:
        state = service.snapshot()
        return jsonify(error=str(error), state=state), 400
    except Exception as error:
        service._set_error(f"Camera recognition failed: {error}")
        return jsonify(error=service.snapshot()["error"], state=service.snapshot()), 500
    return jsonify(state=service.snapshot())


@app.get("/video_feed")
def video_feed():
    def frames():
        previous_frame = None
        while True:
            with service.lock:
                frame = service.latest_frame
                running = service.state["camera_status"] in ("starting", "running")
            if frame is not None and frame is not previous_frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                previous_frame = frame
            elif not running:
                break
            time.sleep(0.04)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Camera frame exceeds the 2 MB upload limit."), 413


@app.post("/api/sentence/add")
def sentence_add():
    body = request.get_json(silent=True) or {}
    try:
        event_id = int(body.get("event_id"))
        added = service.add_word(event_id)
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error), state=service.snapshot()), 400
    return jsonify(added=added, state=service.snapshot())


@app.post("/api/sentence/delete-last")
def sentence_delete_last():
    with service.lock:
        removed = service.sentence.pop() if service.sentence else None
    return jsonify(removed=display_label(removed) if removed else None, state=service.snapshot())


@app.post("/api/sentence/clear")
def sentence_clear():
    with service.lock:
        service.sentence.clear()
    return jsonify(state=service.snapshot())


@app.post("/api/speech/speak")
def speech_speak():
    text = service.snapshot()["sentence_text"]
    try:
        service.speech.speak(text)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(state=service.snapshot())


@app.post("/api/speech/stop")
def speech_stop():
    service.speech.stop()
    return jsonify(state=service.snapshot())


def main():
    print("AI Sign Language Recognition is available at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")),
            debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
