import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import time
import gc
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="wide")

st.title("🧠 NeuroScan Lite – Optimized Version")

# -------------------------
# Risk Function
# -------------------------
def calculate_risk(freq, amplitude):

    if not (4 <= freq <= 6):
        return 0

    if amplitude < 3:
        return 0

    return min(100, 50 + amplitude * 5)

# -------------------------
# Video Processor
# -------------------------
class TremorProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,  # lighter model
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.prev_x = None
        self.alpha = 0.3

        self.x_buffer = deque(maxlen=90)  # reduced buffer
        self.fps = 20  # lower fps
        self.freq = 0
        self.amplitude = 0
        self.risk = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # 🔥 Reduce resolution to save memory
        img = cv2.resize(img, (480, 360))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = img.shape
            index_tip = results.multi_hand_landmarks[0].landmark[8]
            x = int(index_tip.x * w)

            if self.prev_x is None:
                smooth_x = x
            else:
                smooth_x = self.alpha * x + (1 - self.alpha) * self.prev_x

            self.prev_x = smooth_x
            self.x_buffer.append(smooth_x)

        # ---- SIGNAL ----
        if len(self.x_buffer) > 50:
            signal = np.array(self.x_buffer)
            signal -= np.mean(signal)

            self.amplitude = np.std(signal)

            fft_vals = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), d=1/self.fps)

            band_mask = (freqs >= 4) & (freqs <= 6)

            if np.any(band_mask):
                band_freqs = freqs[band_mask]
                band_fft = fft_vals[band_mask]

                if len(band_fft) > 0:
                    self.freq = band_freqs[np.argmax(band_fft)]
                else:
                    self.freq = 0

            self.risk = calculate_risk(self.freq, self.amplitude)

        # ---- OVERLAY ----
        cv2.putText(img, f"Freq: {self.freq:.2f} Hz",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 200), 2)

        cv2.putText(img, f"Amp: {self.amplitude:.2f}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 200), 2)

        cv2.putText(img, f"Risk: {self.risk:.1f}%",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 200), 2)

        gc.collect()  # 🔥 force memory cleanup

        return img


RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

webrtc_streamer(
    key="neuro-lite",
    video_processor_factory=TremorProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": 480,
            "height": 360,
            "frameRate": 20
        },
        "audio": False,
    },
)
