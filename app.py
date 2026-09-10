"""
Pothole Detection - Live Camera Streamlit App
AI for Future Workforce - Capstone Project

Uses the browser's webcam (via streamlit-webrtc) to run a trained CNN on
live video frames and raise an on-screen + audio alert whenever a pothole
is detected on a paved road.

This version uses a 3-class model (Plain / Pothole / NotRoad) instead of
a binary Plain-vs-Pothole model. The NotRoad class was added because a
binary model, having only ever seen road images, would confidently but
wrongly call any unrelated scene (a face, a room) "Plain" or "Pothole" --
it had no way to say "this isn't a road at all". Training on real
NotRoad images (captured from the same webcam/room this app runs in via
collect_data.py) lets the model actually recognize non-road scenes,
instead of relying on a hand-tuned classical-CV heuristic.
"""

import streamlit as st
import numpy as np
import av
import time
import cv2
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

st.set_page_config(page_title="Pothole Detector", page_icon="🕳️", layout="wide")

WEIGHTS_PATH = "pothole_classifier.weights.h5"
IMG_SIZE = (64, 64)
CLASS_NAMES = {0: "NotRoad", 1: "Plain", 2: "Pothole"}  # must match training class_indices

# ---------------------------------------------------------
# Load model once (cached across reruns)
#
# NOTE: We rebuild the architecture in code and load only the *weights*
# (.weights.h5), instead of using load_model() on a full .keras file.
# Full .keras files embed a version-specific layer/initializer config,
# which breaks when the Keras version used to save the model differs
# from the Keras version used to load it (e.g. "GlorotUniform got an
# unexpected keyword argument 'input_axes'"). Weights-only loading avoids
# this entirely since it just loads raw numeric arrays into a model we
# define ourselves here, matching the exact training architecture.
# ---------------------------------------------------------
def build_architecture():
    m = Sequential()
    m.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(units=128, activation='relu'))
    m.add(Dropout(0.4))
    m.add(Dense(units=3, activation='softmax'))  # NotRoad / Plain / Pothole
    return m


@st.cache_resource
def get_model():
    m = build_architecture()
    m.load_weights(WEIGHTS_PATH)
    return m

model = get_model()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")
frame_skip = st.sidebar.slider("Check every Nth frame", 1, 15, 5,
                                help="Higher = faster / less CPU, but slightly less responsive")
cooldown_frames = st.sidebar.slider("Cooldown frames (avoid double count)", 1, 10, 3)
sound_alert = st.sidebar.checkbox("🔊 Play sound alert on detection", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Scope note:** This model is trained on paved (asphalt/concrete) "
    "roads only. Unpaved/kaccha road surfaces are outside its scope."
)
st.sidebar.markdown(
    "**Not-Road detection:** The model has a dedicated third class trained "
    "on real photos of non-road scenes (captured via `collect_data.py`), "
    "so it directly predicts 'NotRoad' when the camera isn't pointed at a "
    "road, instead of relying on a hand-tuned heuristic."
)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["📹 Live Camera", "📁 Upload Video"])

# ===========================================================
# Shared helper: run one frame through the model and update
# the running pothole-count state (used by both modes)
# ===========================================================
def classify_frame(img, state):
    """state is a dict with keys: pothole_in_view, no_detect_streak,
    total_potholes, last_status, new_event. Mutated in place."""
    resized = cv2.resize(img, IMG_SIZE)
    arr = np.expand_dims(resized / 255.0, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    pred_class = CLASS_NAMES[int(np.argmax(probs))]
    state["last_status"] = pred_class

    state["new_event"] = False
    if pred_class == "Pothole":
        state["no_detect_streak"] = 0
        if not state["pothole_in_view"]:
            state["total_potholes"] += 1
            state["pothole_in_view"] = True
            state["new_event"] = True
    else:
        state["no_detect_streak"] += 1
        if state["no_detect_streak"] >= cooldown_frames:
            state["pothole_in_view"] = False
    return state


def draw_overlay(img, state):
    if state["last_status"] == "NotRoad":
        label, color = "Not a road scene", (0, 200, 200)
    elif state["pothole_in_view"]:
        label, color = "POTHOLE DETECTED!", (0, 0, 255)
    else:
        label, color = "Road Clear", (0, 200, 0)

    cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(img, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(img, f"Count: {state['total_potholes']}",
                (img.shape[1] - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if state["pothole_in_view"]:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 10)
    return img

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🕳️ Real-Time Pothole Detection")

if mode == "📹 Live Camera":
    st.caption("Point your camera at the road ahead — the model checks live frames and alerts you when it sees a pothole.")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### 📊 Live Stats")
        status_placeholder = st.empty()
        count_placeholder = st.empty()
        alert_placeholder = st.empty()

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class PotholeDetector(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.state = {
                "pothole_in_view": False, "no_detect_streak": 0,
                "total_potholes": 0, "last_status": "NotRoad", "new_event": False,
            }

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % frame_skip == 0:
                classify_frame(img, self.state)

            img = draw_overlay(img, self.state)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with col1:
        ctx = webrtc_streamer(
            key="pothole-detection",
            video_processor_factory=PotholeDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # -------------------------------------------------------
    # Live stats panel (polls the processor state)
    # -------------------------------------------------------
    if ctx.video_processor:
        placeholder_loop_count = 0
        while ctx.state.playing:
            proc = ctx.video_processor
            if proc is None:
                break
            s = proc.state

            if s["last_status"] == "NotRoad":
                status_placeholder.info("🟡 Not a road scene — pothole check paused")
            elif s["pothole_in_view"]:
                status_placeholder.error("🔴 **POTHOLE DETECTED**")
            else:
                status_placeholder.success("🟢 Road Clear")

            count_placeholder.metric("Total Potholes Detected", s["total_potholes"])

            if s["new_event"]:
                alert_placeholder.warning(f"⚠️ New pothole spotted! (#{s['total_potholes']})")
                if sound_alert:
                    st.markdown(
                        """
                        <audio autoplay>
                        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                        </audio>
                        """,
                        unsafe_allow_html=True,
                    )
                s["new_event"] = False

            time.sleep(0.3)
            placeholder_loop_count += 1
            if placeholder_loop_count > 100000:
                break
    else:
        status_placeholder.info("Click **START** above to begin live detection.")

    st.markdown("---")
    st.caption(
        "Note: since this model classifies single frames rather than localizing "
        "objects, a stateful cooldown is used so a single pothole in view for "
        "several frames is counted only once — it resets once the road is clear "
        "(or the camera moves off the road) for a few frames."
    )

else:  # 📁 Upload Video
    st.caption("Upload a road video — the model scans it frame by frame and produces an annotated output with a pothole count.")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️ Run Pothole Detection on this Video", use_container_width=True):
            # Save upload to a temp file (cv2 needs a real file path)
            in_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(in_path, "wb") as f:
                f.write(uploaded_video.read())

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(in_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            state = {
                "pothole_in_view": False, "no_detect_streak": 0,
                "total_potholes": 0, "last_status": "NotRoad", "new_event": False,
            }

            progress_bar = st.progress(0, text="Processing video...")
            count_display = st.empty()
            event_log = st.container()

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % frame_skip == 0:
                    classify_frame(frame, state)
                    if state["new_event"]:
                        timestamp = frame_idx / fps
                        with event_log:
                            st.write(f"⚠️ Pothole #{state['total_potholes']} detected at t={timestamp:.1f}s")

                frame = draw_overlay(frame, state)
                out.write(frame)

                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing video... frame {frame_idx}/{total_frames}"
                    )
                count_display.metric("Total Potholes Detected So Far", state["total_potholes"])

            cap.release()
            out.release()
            progress_bar.progress(1.0, text="Done!")

            st.success(f"✅ Processing complete — {state['total_potholes']} unique pothole(s) detected.")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.markdown("### 🎬 Annotated Output")
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="pothole_detection_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            os.remove(in_path)
            os.remove(out_path)

    st.markdown("---")
    st.caption(
        "Note: as with live detection, a stateful cooldown avoids counting the "
        "same physical pothole multiple times just because it stays in view "
        "across several consecutive frames."
    )
