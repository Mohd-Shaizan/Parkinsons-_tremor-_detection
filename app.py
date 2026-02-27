import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
from collections import deque
from scipy.fft import fft

# =============================
# CONFIG
# =============================
st.set_page_config(layout="centered")

# =============================
# GLOBAL VARIABLES
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

position_buffer = deque(maxlen=150)  # 5 sec @30fps
start_time = time.time()

# =============================
# RISK CALCULATION (FIXED)
# =============================
def calculate_risk(freq, amplitude, stability):

    if freq is None:
        return 0

    risk = 0

    # Parkinson's range 4–6 Hz
    if 4 <= freq <= 6:
        risk += 40
    elif 3.5 <= freq <= 7:
        risk += 20

    if amplitude > 15:
        risk += 30

    risk += (1 - stability) * 30

    return min(100, max(0, risk))


# =============================
# VIDEO PROCESSOR
# =============================
class TremorProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        freq = None
        amplitude = 0
        stability = 1
        risk = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            # Index fingertip (landmark 8)
            lm = hand.landmark[8]
            x = int(lm.x * w)
            y = int(lm.y * h)

            position_buffer.append(y)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                img, hand, mp_hands.HAND_CONNECTIONS
            )

            # =============================
            # FFT Tremor Detection
            # =============================
            if len(position_buffer) > 60:

                signal = np.array(position_buffer)
                signal = signal - np.mean(signal)

                fft_values = np.abs(fft(signal))
                freqs = np.fft.fftfreq(len(signal), d=1/30)

                positive = freqs > 0
                freqs = freqs[positive]
                fft_values = fft_values[positive]

                if len(fft_values) > 0:
                    idx = np.argmax(fft_values)
                    freq = freqs[idx]
                    amplitude = fft_values[idx]

                    stability = 1 - (np.std(signal) / 50)
                    stability = np.clip(stability, 0, 1)

                    risk = calculate_risk(freq, amplitude, stability)

        # =============================
        # UI Overlay
        # =============================
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        neon = (0, 255, 200)

        cv2.putText(img, "NEUROSCAN AI", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)

        cv2.putText(img, f"Frequency: {freq:.2f} Hz" if freq else "Frequency: --",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon, 1)

        cv2.putText(img, f"Risk Index: {risk:.1f}%",
                    (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, neon, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================
# STREAMLIT UI
# =============================

st.title("NeuroScan AI")
st.caption("Research Prototype — Not a Medical Diagnosis Tool")

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(
        key="camera",
        video_processor_factory=TremorProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
