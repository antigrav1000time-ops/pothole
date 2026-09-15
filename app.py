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

UI note: all detection/model logic below is unchanged from the previous
version. Only the presentation layer (CSS, layout, cards, tabs) was
redesigned -- see inject_custom_css() and the render_* helpers.
"""

import streamlit as st
import numpy as np
import av
import time
import cv2
import tempfile
import os
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import road_router

st.set_page_config(page_title="Pothole Detector", page_icon="🕳️", layout="wide")

WEIGHTS_PATH = "pothole_classifier.weights.h5"
IMG_SIZE = (64, 64)
CLASS_NAMES = {0: "NotRoad", 1: "Plain", 2: "Pothole"}  # must match training class_indices

# Road-hazard palette (kept in sync with .streamlit/config.toml)
CHARCOAL = "#23272B"
CARD_BG = "#2E3338"
CARD_BORDER = "#3A3F44"
HAZARD = "#F2A900"
SAFETY_ORANGE = "#E85D26"
TEXT_MAIN = "#F5F5F3"
TEXT_MUTED = "#9AA0A6"
GREEN = "#3FB950"
RED = "#F85149"
AMBER = "#F2C94C"


# ============================================================
# UI helpers -- custom CSS + small HTML component renderers.
# None of this touches the detection logic below.
# ============================================================
def inject_custom_css():
    st.markdown(f"""
    <style>
        /* ---- Hero header ---- */
        .hero-wrap {{
            padding: 0.4rem 0 1.2rem 0;
            border-bottom: 3px solid {HAZARD};
            margin-bottom: 1.6rem;
        }}
        .hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            color: {TEXT_MAIN};
            margin-bottom: 0.15rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .hero-tagline {{
            font-size: 1.05rem;
            color: {TEXT_MUTED};
            font-style: italic;
        }}

        /* ---- Status banner (Road Clear / Pothole / Not a Road) ---- */
        .status-banner {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
            animation: fadeIn 0.35s ease-in-out;
            border: 1px solid transparent;
        }}
        .status-clear {{
            background: rgba(63, 185, 80, 0.14);
            border-color: rgba(63, 185, 80, 0.4);
            color: {GREEN};
        }}
        .status-alert {{
            background: rgba(248, 81, 73, 0.16);
            border-color: rgba(248, 81, 73, 0.5);
            color: {RED};
            animation: fadeIn 0.35s ease-in-out, pulse 1.4s ease-in-out infinite;
        }}
        .status-notroad {{
            background: rgba(242, 201, 76, 0.14);
            border-color: rgba(242, 201, 76, 0.4);
            color: {AMBER};
        }}
        .status-idle {{
            background: {CARD_BG};
            border-color: {CARD_BORDER};
            color: {TEXT_MUTED};
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-4px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.35); }}
            50%      {{ box-shadow: 0 0 0 8px rgba(248, 81, 73, 0); }}
        }}

        /* ---- Metric / stat cards ---- */
        .metric-card {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 1.1rem 0.8rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.1rem;
            font-weight: 800;
            color: {HAZARD};
            line-height: 1.15;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: {TEXT_MUTED};
            margin-top: 0.2rem;
        }}
        .stats-row {{ display: flex; gap: 0.7rem; margin-bottom: 0.8rem; }}
        .stats-row .metric-card {{ flex: 1; }}

        /* ---- New-event alert chip ---- */
        .event-chip {{
            background: rgba(242, 169, 0, 0.14);
            border: 1px solid rgba(242, 169, 0, 0.4);
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            color: {HAZARD};
            font-weight: 600;
            font-size: 0.95rem;
            animation: fadeIn 0.3s ease-in-out;
        }}

        /* ---- Sidebar section headers ---- */
        .sidebar-section-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {HAZARD};
            margin-top: 0.6rem;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        /* ---- Footer ---- */
        .app-footer {{
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid {CARD_BORDER};
            text-align: center;
            color: {TEXT_MUTED};
            font-size: 0.85rem;
        }}

        /* Tabs: a bit more breathing room + bolder active tab */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-title">🕳️ Pothole Detection on Paved Roads</div>
        <div class="hero-tagline">Real-time AI road-safety monitoring — live camera and video analysis</div>
    </div>
    """, unsafe_allow_html=True)


def status_banner_html(status_key, count=None):
    """status_key: 'clear' | 'alert' | 'notroad' | 'verifying' | 'idle'"""
    variants = {
        "clear":     ("🟢", "Road Clear", "status-clear"),
        "alert":     ("🔴", "POTHOLE DETECTED", "status-alert"),
        "notroad":   ("🟡", "Not a Road Scene — check paused", "status-notroad"),
        "verifying": ("🤖", "Verifying it's a road (AI check in progress)...", "status-notroad"),
        "idle":      ("⏳", "Click START below to begin live detection", "status-idle"),
    }
    icon, text, css_class = variants[status_key]
    suffix = f"&nbsp;&nbsp;·&nbsp;&nbsp;Count: {count}" if count is not None else ""
    return f"""<div class="status-banner {css_class}"><span>{icon}</span><span>{text}{suffix}</span></div>"""


def metric_card_html(value, label):
    return f"""<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>"""


def stats_row_html(cards):
    inner = "".join(metric_card_html(v, l) for v, l in cards)
    return f'<div class="stats-row">{inner}</div>'


def render_footer():
    st.markdown(f"""
    <div class="app-footer">
        Built with a 3-class CNN (Plain / Pothole / NotRoad) &nbsp;·&nbsp;
        AI Associate Developer Program — Capstone Project
    </div>
    """, unsafe_allow_html=True)


inject_custom_css()

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
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown('<div class="sidebar-section-title">Detection Sensitivity</div>', unsafe_allow_html=True)
    frame_skip = st.slider("Check every Nth frame", 1, 15, 5,
                            help="Higher = faster / less CPU, but slightly less responsive")
    cooldown_frames = st.slider("Cooldown frames (avoid double count)", 1, 10, 3,
                                 help="How many clear frames in a row before a new pothole can be counted again")

    st.markdown('<div class="sidebar-section-title">Alerts</div>', unsafe_allow_html=True)
    sound_alert = st.checkbox("🔊 Play sound alert on detection", value=True)

    st.divider()

    with st.expander("📍 Scope note"):
        st.markdown(
            "This model is trained on **paved (asphalt/concrete) roads only**. "
            "Unpaved/kaccha road surfaces are outside its scope."
        )
    with st.expander("🧠 How 'Not-Road' detection works"):
        st.markdown(
            "The model has a dedicated third class trained on real photos of "
            "non-road scenes (captured via `collect_data.py`), so it directly "
            "predicts **NotRoad** when the camera isn't pointed at a road, "
            "instead of relying on a hand-tuned heuristic."
        )

    st.divider()
    st.markdown('<div class="sidebar-section-title">🤖 AI Road Verification (optional)</div>', unsafe_allow_html=True)
    st.caption(
        "Adds a vision-LLM (Groq) as a second, more general check for "
        "'is this actually a road?' — periodically, in the background, "
        "on top of the local CNN's own NotRoad class."
    )
    use_llm_gate = st.checkbox("Enable AI road verification", value=False)
    llm_check_interval = 3
    if use_llm_gate:
        llm_check_interval = st.slider(
            "Check interval (seconds)", 2, 10, 3,
            help="Lower = catches scene changes faster, but more API calls."
        )
        _env_key = road_router._get_api_key()
        if _env_key:
            st.success("GROQ_API_KEY found — AI verification ready.", icon="✅")
        else:
            st.warning(
                "No GROQ_API_KEY found in environment/secrets. "
                "Set it before running the app — never paste a key directly "
                "into this box or into source code.",
                icon="⚠️",
            )
            with st.expander("🔍 Diagnose why it's not found"):
                st.markdown(road_router.diagnose_api_key_setup())

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
    if state["last_status"] == "Verifying":
        label, color = "Verifying road (AI)...", (0, 255, 255)
    elif state["last_status"] == "NotRoad":
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
render_hero()

tab_live, tab_upload = st.tabs(["📹  Live Camera", "📁  Upload Video"])

# ============================================================
# TAB 1 — Live Camera
# ============================================================
with tab_live:
    st.caption("Point your camera at the road ahead — the model checks live frames and alerts you when it sees a pothole.")

    col1, col2 = st.columns([3, 1], gap="medium")

    with col2:
        st.markdown("#### 📊 Live Stats")
        status_placeholder = st.empty()
        stats_placeholder = st.empty()
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
            self.latest_frame = None
            self.llm_says_road = None  # None = not checked yet / unknown -> trust local CNN
            self._stop_llm = False
            if use_llm_gate:
                self._llm_thread = threading.Thread(target=self._llm_loop, daemon=True)
                self._llm_thread.start()

        def _llm_loop(self):
            # Runs in the background so the video feed never waits on a
            # network call. Reads whatever frame was most recently seen by
            # recv() and periodically asks the LLM road-router about it.
            while not self._stop_llm:
                if self.latest_frame is not None:
                    result = road_router.check_is_road(self.latest_frame)
                    if result is not None:
                        self.llm_says_road = result
                time.sleep(llm_check_interval)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            self.latest_frame = img.copy()

            if self.frame_count % frame_skip == 0:
                if use_llm_gate:
                    # Strict gate: pothole detection only runs once the LLM
                    # has explicitly confirmed this is a road. Until then
                    # (still checking, or confirmed not-a-road), we do NOT
                    # trust the local CNN's own Plain/Pothole/NotRoad guess --
                    # this avoids the CNN's small NotRoad class producing
                    # false "Pothole" calls on non-road scenes.
                    if self.llm_says_road is True:
                        classify_frame(img, self.state)
                    elif self.llm_says_road is False:
                        self.state["last_status"] = "NotRoad"
                        self.state["pothole_in_view"] = False
                        self.state["new_event"] = False
                    else:
                        self.state["last_status"] = "Verifying"
                        self.state["pothole_in_view"] = False
                        self.state["new_event"] = False
                else:
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

            if s["last_status"] == "Verifying":
                status_placeholder.markdown(status_banner_html("verifying"), unsafe_allow_html=True)
            elif s["last_status"] == "NotRoad":
                status_placeholder.markdown(status_banner_html("notroad"), unsafe_allow_html=True)
            elif s["pothole_in_view"]:
                status_placeholder.markdown(status_banner_html("alert"), unsafe_allow_html=True)
            else:
                status_placeholder.markdown(status_banner_html("clear"), unsafe_allow_html=True)

            stats_placeholder.markdown(
                stats_row_html([(s["total_potholes"], "Potholes Detected")]),
                unsafe_allow_html=True,
            )

            if s["new_event"]:
                alert_placeholder.markdown(
                    f'<div class="event-chip">⚠️ New pothole spotted — #{s["total_potholes"]}</div>',
                    unsafe_allow_html=True,
                )
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
            elif use_llm_gate:
                llm_status = proc.llm_says_road
                if llm_status is True:
                    alert_placeholder.caption("🤖 AI verification: confirmed road ✅")
                elif llm_status is False:
                    alert_placeholder.caption("🤖 AI verification: not a road ⛔")
                else:
                    alert_placeholder.caption("🤖 AI verification: checking…")
            else:
                alert_placeholder.empty()

            time.sleep(0.3)
            placeholder_loop_count += 1
            if placeholder_loop_count > 100000:
                break
    else:
        status_placeholder.markdown(status_banner_html("idle"), unsafe_allow_html=True)
        stats_placeholder.markdown(stats_row_html([(0, "Potholes Detected")]), unsafe_allow_html=True)

    st.caption(
        "Note: since this model classifies single frames rather than localizing "
        "objects, a stateful cooldown is used so a single pothole in view for "
        "several frames is counted only once — it resets once the road is clear "
        "(or the camera moves off the road) for a few frames."
    )

# ============================================================
# TAB 2 — Upload Video
# ============================================================
with tab_upload:
    st.caption("Upload a road video — the model scans it frame by frame and produces an annotated output with a pothole count.")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if use_llm_gate:
        st.caption("🤖 AI road verification is ON — processing will be slower (one API call roughly every second of footage).")

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️  Run Pothole Detection on this Video", use_container_width=True, type="primary"):
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
            live_stats_placeholder = st.empty()
            event_log = st.container()

            llm_says_road = None  # None = unknown yet -> trust local CNN
            llm_check_every_n_frames = max(int(fps), 1)  # roughly once per second of footage

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if use_llm_gate and frame_idx % llm_check_every_n_frames == 0:
                    result = road_router.check_is_road(frame)
                    if result is not None:
                        llm_says_road = result

                if frame_idx % frame_skip == 0:
                    if use_llm_gate:
                        # Strict gate: only trust the CNN's Plain/Pothole
                        # call once the LLM has explicitly confirmed this
                        # segment of the video is a road.
                        if llm_says_road is True:
                            classify_frame(frame, state)
                        elif llm_says_road is False:
                            state["last_status"] = "NotRoad"
                            state["pothole_in_view"] = False
                            state["new_event"] = False
                        else:
                            state["last_status"] = "Verifying"
                            state["pothole_in_view"] = False
                            state["new_event"] = False
                    else:
                        classify_frame(frame, state)

                    if state["new_event"]:
                        timestamp = frame_idx / fps
                        with event_log:
                            st.markdown(
                                f'<div class="event-chip">⚠️ Pothole #{state["total_potholes"]} detected at t={timestamp:.1f}s</div>',
                                unsafe_allow_html=True,
                            )

                frame = draw_overlay(frame, state)
                out.write(frame)

                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing video... frame {frame_idx}/{total_frames}"
                    )
                live_stats_placeholder.markdown(
                    stats_row_html([(state["total_potholes"], "Potholes So Far")]),
                    unsafe_allow_html=True,
                )

            cap.release()
            out.release()
            progress_bar.progress(1.0, text="Done!")

            duration_sec = total_frames / fps if fps else 0

            st.markdown(status_banner_html("alert" if state["total_potholes"] > 0 else "clear",
                                            count=None), unsafe_allow_html=True)

            st.markdown("#### 📋 Summary")
            st.markdown(
                stats_row_html([
                    (state["total_potholes"], "Potholes Detected"),
                    (f"{duration_sec:.0f}s", "Video Duration"),
                    (f"{total_frames}", "Frames Processed"),
                ]),
                unsafe_allow_html=True,
            )

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.markdown("#### 🎬 Annotated Output")
            st.video(video_bytes)
            st.download_button(
                "⬇️  Download Annotated Video",
                data=video_bytes,
                file_name="pothole_detection_output.mp4",
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )

            os.remove(in_path)
            os.remove(out_path)

    st.caption(
        "Note: as with live detection, a stateful cooldown avoids counting the "
        "same physical pothole multiple times just because it stays in view "
        "across several consecutive frames."
    )

render_footer()
