"""
Road-scene gate using a vision-capable LLM (Groq).

This module answers one yes/no question about a frame -- "is this a paved
road?" -- as a smarter, more general alternative/supplement to the local
CNN's own NotRoad class.

Why: the CNN's NotRoad class was trained on only ~68 photos captured from a
single room/setup, so it reliably rejects that specific environment but
does not generalize to arbitrary non-road scenes. A general vision LLM has
seen a vast, diverse range of real-world images, so it recognizes "this is
/ isn't a road" far more robustly -- at the cost of network latency and
per-call cost, which is why it's used as a periodic background check
rather than on every single frame (see app.py).

SECURITY NOTE: the API key is read from the GROQ_API_KEY environment
variable (or Streamlit secrets) -- NEVER hardcode an API key directly in
source code. A key pasted into a shared file is effectively public; if
that ever happens, revoke/rotate the key immediately.
"""

import os
import base64
from typing import Optional

import cv2
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

GROQ_MODEL = "qwen/qwen3.8-27b"  # Groq vision-capable model (see console.groq.com/docs/vision)


class RoadCheck(BaseModel):
    is_road: bool = Field(
        description="True if the image shows a paved (asphalt/concrete) road surface, as seen from a dashcam or road-inspection camera. False for anything else."
    )
    reason: str = Field(description="One short phrase explaining the decision.")


def _get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets if running inside Streamlit with a configured
    # secrets.toml; fall back to a plain environment variable. Either way,
    # nothing is ever hardcoded in source.
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def _get_llm():
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it as an environment variable "
            "(export GROQ_API_KEY=...) or in .streamlit/secrets.toml -- "
            "never hardcode it in source code."
        )
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, max_tokens=200, temperature=0)
    return llm.with_structured_output(RoadCheck)


def _frame_to_data_uri(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise ValueError("Could not encode frame to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def check_is_road(frame_bgr) -> Optional[bool]:
    """Returns True/False, or None if the check couldn't be completed
    (missing API key, network error, etc). Callers should treat None as
    'unknown -- fall back to the local CNN's own judgement'."""
    try:
        llm = _get_llm()
        data_uri = _frame_to_data_uri(frame_bgr)
        message = HumanMessage(content=[
            {"type": "text", "text": (
                "Look at this image. Is it a photo of a paved (asphalt or "
                "concrete) road surface -- the kind of thing a pothole-"
                "detection dashcam or road-inspection camera would see? "
                "Answer false for anything else, including people, rooms, "
                "screenshots, unpaved/dirt roads, or unrelated objects."
            )},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ])
        result = llm.invoke([message])
        return result.is_road
    except Exception as e:
        print(f"[road_router] check failed, falling back to local CNN: {e}")
        return None
