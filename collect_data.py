"""
Non-Road Image Collector - Streamlit App
AI for Future Workforce - Capstone Project

Run this alongside your webcam to quickly collect "Not-Road" training
images (faces, rooms, objects, anything that isn't a road). These will be
used to train a proper 3-class model (Plain / Pothole / Not-Road) so the
main app stops misfiring on non-road scenes.

Usage:
    streamlit run collect_data.py

Instructions:
    1. Click START to begin the webcam.
    2. Move around, show your face, hands, the room, objects on your desk,
       different walls/backgrounds, different lighting/distances.
    3. Either click "Capture This Frame" repeatedly, or turn on
       "Auto-capture" to grab a frame automatically every second.
    4. Aim for at least 50-80 varied images.
    5. Click "Download ZIP" and send that zip file back in the chat.
"""

import streamlit as st
import numpy as np
import av
import cv2
import io
import time
import zipfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Collect Non-Road Images", page_icon="📸", layout="wide")

st.title("📸 Collect 'Not-Road' Training Images")
st.markdown(
    "Move around and show **anything that is NOT a road** — your face, hands, "
    "the room, objects on your desk, walls, ceiling, different lighting. "
    "The more varied, the better the final model will be at telling "
    "'road' apart from 'everything else'."
)

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []  # list of jpg-encoded bytes

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class PassthroughProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")


col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="data-collector",
        video_processor_factory=PassthroughProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### Controls")
    capture_clicked = st.button("📸 Capture This Frame", use_container_width=True)
    auto_capture = st.checkbox("Auto-capture every 1s")
    count_placeholder = st.empty()
    count_placeholder.metric("Images captured", len(st.session_state.captured_images))

    st.markdown("---")
    if st.session_state.captured_images:
        # Build zip in-memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img_bytes in enumerate(st.session_state.captured_images):
                zf.writestr(f"not_road_{i:04d}.jpg", img_bytes)
        zip_buffer.seek(0)

        st.download_button(
            "⬇️ Download ZIP",
            data=zip_buffer,
            file_name="not_road_images.zip",
            mime="application/zip",
            use_container_width=True,
        )

        if st.button("🗑️ Clear all captured images", use_container_width=True):
            st.session_state.captured_images = []
            st.rerun()


def capture_frame():
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            st.session_state.captured_images.append(buf.tobytes())


if capture_clicked:
    capture_frame()
    count_placeholder.metric("Images captured", len(st.session_state.captured_images))

if auto_capture and ctx.state.playing:
    last_capture_time = time.time()
    loop_guard = 0
    while ctx.state.playing and auto_capture:
        now = time.time()
        if now - last_capture_time >= 1.0:
            capture_frame()
            last_capture_time = now
            count_placeholder.metric("Images captured", len(st.session_state.captured_images))
        time.sleep(0.1)
        loop_guard += 1
        if loop_guard > 100000:
            break

st.markdown("---")
st.caption(
    "Tip: for the best results, capture in the same lighting/room where "
    "you'll actually use the pothole detector — this helps the model learn "
    "to tell your real background apart from an actual road."
)
