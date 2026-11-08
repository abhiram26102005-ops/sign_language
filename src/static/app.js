const $ = (selector) => document.querySelector(selector);
const elements = {
  cameraSource: $("#camera-source"),
  cameraFeed: $("#camera-feed"),
  cameraPlaceholder: $("#camera-placeholder"),
  start: $("#start-camera"),
  stop: $("#stop-camera"),
  cameraError: $("#camera-error"),
  cameraNote: $("#camera-note"),
  handStatus: $("#hand-status"),
  connection: $("#connection"),
  connectionLabel: $("#connection-label"),
  liveIndicator: $("#live-indicator"),
  prediction: $("#prediction"),
  confidence: $("#confidence"),
  confidenceMeter: $("#confidence-meter"),
  alternatives: $("#alternatives"),
  sentenceBoard: $("#sentence-board"),
  emptySentence: $("#empty-sentence"),
  wordCount: $("#word-count"),
  add: $("#add-word"),
  delete: $("#delete-word"),
  clear: $("#clear-sentence"),
  copy: $("#copy-sentence"),
  speak: $("#speak"),
  stopSpeech: $("#stop-speech"),
  speechNote: $("#speech-note"),
  history: $("#history-list"),
  historyCount: $("#history-count"),
  toast: $("#toast"),
};

let lastEventId = null;
let consumedEventId = null;
let pollHandle = null;
let toastHandle = null;
let sentenceText = "";
let mediaStream = null;
let frameCanvas = document.createElement("canvas");
let frameHandle = null;
let frameInFlight = false;
let browserSpeechStatus = null;
let cameraRequestPending = false;
let clientCameraError = null;

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request could not be completed.");
  return result;
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  clearTimeout(toastHandle);
  toastHandle = setTimeout(() => elements.toast.classList.remove("is-visible"), 2400);
}

function render(state) {
  const status = state.camera_status;
  const starting = status === "starting" || (cameraRequestPending && status !== "running");
  const running = status === "running" || starting;
  elements.start.disabled = running;
  elements.stop.disabled = status !== "running" && status !== "starting";
  elements.connection.classList.toggle("is-live", status === "running");
  elements.connectionLabel.textContent = status === "running" ? "Camera active" : starting ? "Starting camera" : status === "error" ? "Camera unavailable" : "Camera off";
  elements.liveIndicator.classList.toggle("is-live", status === "running");
  elements.liveIndicator.innerHTML = `<i></i> ${status === "running" ? "LIVE" : starting ? "CONNECTING" : status === "error" ? "ERROR" : "STANDBY"}`;
  const cameraError = clientCameraError || state.error;
  elements.cameraError.hidden = !cameraError;
  elements.cameraError.textContent = cameraError || "";
  elements.cameraNote.textContent = status === "error" ? "Resolve the issue above, then start the camera again." : "Frames are sent to this app for local-model inference.";

  const hasFeed = running && Boolean(elements.cameraFeed.getAttribute("src"));
  elements.cameraFeed.hidden = !hasFeed;
  elements.cameraPlaceholder.hidden = Boolean(hasFeed);
  elements.handStatus.textContent = status !== "running" ? "Waiting for camera" : state.handedness || "No hand detected";

  const recognized = Boolean(state.prediction && state.event_id !== null);
  elements.prediction.textContent = recognized ? humanize(state.prediction) : status === "running" ? "Show a trained sign" : "No sign yet";
  const percent = recognized ? Math.round(state.confidence * 100) : 0;
  elements.confidence.textContent = recognized ? `${percent}%` : "--%";
  elements.confidenceMeter.style.width = `${percent}%`;
  lastEventId = recognized ? state.event_id : null;
  elements.add.disabled = !recognized || consumedEventId === lastEventId;

  renderAlternatives(state.top_predictions || []);
  renderSentence(state.sentence || [], state.sentence_text || "");
  renderHistory(state.history || []);
  renderSpeech(state.speech_status, state.speech_error);
}

function humanize(label) {
  if (label === "iloveyou") return "I love you";
  if (label === "clockit") return "Clock it";
  return label ? label[0].toUpperCase() + label.slice(1) : "";
}

function renderAlternatives(items) {
  elements.alternatives.replaceChildren();
  const heading = document.createElement("span");
  heading.className = "field-label";
  heading.textContent = "ALTERNATIVES";
  elements.alternatives.append(heading);
  if (!items.length) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "Show a sign to see predictions";
    elements.alternatives.append(empty);
    return;
  }
  for (const item of items) {
    const entry = document.createElement("span");
    entry.className = "alt-item";
    entry.append(document.createTextNode(item.text || humanize(item.label)));
    const score = document.createElement("b");
    score.textContent = `${Math.round(item.confidence * 100)}%`;
    entry.append(score);
    elements.alternatives.append(entry);
  }
}

function renderSentence(words, text) {
  sentenceText = text;
  elements.sentenceBoard.replaceChildren();
  if (!words.length) {
    const empty = document.createElement("p");
    empty.className = "empty-sentence";
    empty.textContent = "Your signs will appear here as words.";
    elements.sentenceBoard.append(empty);
  } else {
    for (const word of words) {
      const chip = document.createElement("span");
      chip.className = "word-chip";
      chip.textContent = humanize(word);
      elements.sentenceBoard.append(chip);
    }
  }
  elements.wordCount.textContent = `${words.length} ${words.length === 1 ? "WORD" : "WORDS"}`;
  elements.delete.disabled = !words.length;
  elements.clear.disabled = !words.length;
  elements.copy.disabled = !words.length;
  elements.speak.disabled = !words.length;
}

function renderHistory(items) {
  elements.historyCount.textContent = String(items.length);
  elements.history.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = "Recognized signs will be logged here.";
    elements.history.append(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("li");
    row.className = "history-item";
    const label = document.createElement("span");
    label.textContent = item.text || humanize(item.label);
    const time = document.createElement("time");
    time.textContent = item.time;
    row.append(label, time);
    elements.history.append(row);
  }
}

function renderSpeech(status, error) {
  status = browserSpeechStatus || status;
  elements.stopSpeech.disabled = status !== "speaking";
  if (error) elements.speechNote.textContent = error;
  else if (status === "speaking") elements.speechNote.textContent = "Speaking sentence aloud…";
  else if (status === "stopped") elements.speechNote.textContent = "Speech stopped.";
  else elements.speechNote.textContent = "Offline voice output · device voices";
}

async function poll() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    elements.cameraError.hidden = false;
    elements.cameraError.textContent = `Unable to reach the local service: ${error.message}`;
  }
}

function beginPolling() {
  clearInterval(pollHandle);
  pollHandle = setInterval(poll, 350);
  poll();
}

async function startCamera() {
  cameraRequestPending = true;
  clientCameraError = null;
  elements.cameraError.hidden = true;
  elements.start.disabled = true;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access requires HTTPS or localhost in a supported browser.");
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    elements.cameraSource.srcObject = mediaStream;
    await elements.cameraSource.play();
    const state = await api("/api/camera/start", {});
    cameraRequestPending = false;
    elements.cameraFeed.src = `/video_feed?t=${Date.now()}`;
    render(state);
    frameHandle = setInterval(sendFrame, 125);
    mediaStream.getVideoTracks().forEach((track) => {
      track.addEventListener("ended", stopCamera, { once: true });
    });
  } catch (error) {
    releaseCamera();
    cameraRequestPending = false;
    clientCameraError = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser settings, then retry."
      : error.name === "NotFoundError"
        ? "No camera was found on this device. Connect a webcam, then retry."
        : error.message;
    elements.start.disabled = false;
    elements.cameraError.hidden = false;
    elements.cameraError.textContent = clientCameraError;
  }
}

function releaseCamera() {
  clearInterval(frameHandle);
  frameHandle = null;
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.cameraSource.srcObject = null;
  frameInFlight = false;
}

function sendFrame() {
  if (frameInFlight || !mediaStream || elements.cameraSource.readyState < 2) return;
  const video = elements.cameraSource;
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  frameCanvas.width = Math.round(video.videoWidth * scale);
  frameCanvas.height = Math.round(video.videoHeight * scale);
  frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  frameInFlight = true;
  frameCanvas.toBlob(async (blob) => {
    if (!blob) {
      frameInFlight = false;
      return;
    }
    try {
      const response = await fetch("/api/frame", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The camera frame was rejected.");
      render(result.state);
    } catch (error) {
      clientCameraError = error.message;
      elements.cameraError.hidden = false;
      elements.cameraError.textContent = clientCameraError;
    } finally {
      frameInFlight = false;
    }
  }, "image/jpeg", 0.72);
}

async function stopCamera() {
  elements.stop.disabled = true;
  releaseCamera();
  try {
    const result = await api("/api/camera/stop", {});
    elements.cameraFeed.removeAttribute("src");
    consumedEventId = null;
    render(result);
  } catch (error) {
    toast(error.message);
  }
}

elements.start.addEventListener("click", startCamera);
elements.stop.addEventListener("click", stopCamera);
elements.add.addEventListener("click", async () => {
  if (lastEventId === null) return;
  try {
    const result = await api("/api/sentence/add", { event_id: lastEventId });
    consumedEventId = lastEventId;
    render(result.state);
    toast(`${result.added} added`);
  } catch (error) {
    toast(error.message);
    await poll();
  }
});
elements.delete.addEventListener("click", async () => {
  try {
    const result = await api("/api/sentence/delete-last", {});
    render(result.state);
    if (result.removed) toast(`${result.removed} removed`);
  } catch (error) { toast(error.message); }
});
elements.clear.addEventListener("click", async () => {
  try { render((await api("/api/sentence/clear", {})).state); }
  catch (error) { toast(error.message); }
});
elements.copy.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(sentenceText); toast("Sentence copied"); }
  catch { toast("Clipboard access is unavailable in this browser"); }
});
elements.speak.addEventListener("click", async () => {
  if ("speechSynthesis" in window && window.SpeechSynthesisUtterance) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sentenceText);
    browserSpeechStatus = "speaking";
    renderSpeech(browserSpeechStatus, null);
    utterance.onend = () => { browserSpeechStatus = null; poll(); };
    utterance.onerror = () => {
      browserSpeechStatus = null;
      elements.speechNote.textContent = "This device could not start speech output.";
    };
    window.speechSynthesis.speak(utterance);
    return;
  }
  try { render((await api("/api/speech/speak", {})).state); }
  catch (error) { elements.speechNote.textContent = error.message; }
});
elements.stopSpeech.addEventListener("click", async () => {
  if (browserSpeechStatus) {
    window.speechSynthesis.cancel();
    browserSpeechStatus = "stopped";
    renderSpeech(browserSpeechStatus, null);
    return;
  }
  try { render((await api("/api/speech/stop", {})).state); }
  catch (error) { toast(error.message); }
});

const savedTheme = localStorage.getItem("signal-theme");
if (savedTheme === "dark") document.documentElement.dataset.theme = "dark";
$("#theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  if (dark) document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  localStorage.setItem("signal-theme", dark ? "dark" : "light");
});

beginPolling();
window.addEventListener("beforeunload", () => {
  clearInterval(pollHandle);
  releaseCamera();
  window.speechSynthesis?.cancel();
});
