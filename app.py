import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="NeuroScan AI", layout="wide")

st.markdown("""
<style>
video {
    max-width: 750px !important;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroScan AI – Tremor Pattern Analyzer")
st.caption("⚠ Research Prototype – Not a Medical Diagnostic Tool")

# ---------------------------
# RISK FUNCTION
# ---------------------------
def calculate_risk(freq, amplitude, stability):

    AMP_MIN = 2.5
    AMP_HIGH = 8.0

    # Must be in Parkinson tremor band
    if not (4 <= freq <= 6):
        return 0

    risk = 50  # base if in band

    if amplitude > AMP_HIGH:
        risk += 30
    elif amplitude > AMP_MIN:
        risk += 15
    else:
        return 0

    risk += (1 - stability) * 20

    return min(100, risk)

# ---------------------------
# VIDEO PROCESSOR
# ---------------------------
class TremorProcessor(VideoProcessorBase):

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.prev_x = None
        self.alpha = 0.2  # smoothing factor

        self.x_buffer = deque(maxlen=150)  # ~5 seconds @30fps
        self.fps = 30
        self.last_time = time.time()

        self.freq = 0
        self.amplitude = 0
        self.stability = 1
        self.risk = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            index_tip = hand.landmark[8]
            x = int(index_tip.x * w)

            # Exponential smoothing
            if self.prev_x is None:
                smooth_x = x
            else:
                smooth_x = self.alpha * x + (1 - self.alpha) * self.prev_x

            self.prev_x = smooth_x
            self.x_buffer.append(smooth_x)

            # Draw landmarks
            for lm in hand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 255, 200), -1)

        # --- SIGNAL PROCESSING ---
        if len(self.x_buffer) > 60:
            signal = np.array(self.x_buffer)

            # Remove DC drift
            signal = signal - np.mean(signal)

            # Amplitude (std deviation)
            self.amplitude = np.std(signal)

            # Stability (variance consistency)
            self.stability = 1 / (1 + np.var(signal))

            # FFT
            fft_vals = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), d=1/self.fps)

            # Parkinson band mask
            band_mask = (freqs >= 4) & (freqs <= 6)

            if np.any(band_mask):
                band_freqs = freqs[band_mask]
                band_fft = fft_vals[band_mask]

                if len(band_fft) > 0:
                    self.freq = band_freqs[np.argmax(band_fft)]
                else:
                    self.freq = 0
            else:
                self.freq = 0

            self.risk = calculate_risk(self.freq, self.amplitude, self.stability)

        # --- UI OVERLAY ---
        overlay = img.copy()
        cv2.rectangle(overlay, (30, 30), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        neon = (0, 255, 200)

        cv2.putText(img, "NEUROSCAN AI", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)

        cv2.putText(img, f"Frequency: {self.freq:.2f} Hz", (50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon, 2)

        cv2.putText(img, f"Amplitude: {self.amplitude:.2f}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon, 2)

        cv2.putText(img, f"Stability: {self.stability:.3f}", (50, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon, 2)

        cv2.putText(img, f"Risk Index: {self.risk:.1f}%", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 3)

        return img


# ---------------------------
# WEBRTC CONFIG
# ---------------------------
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
}

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="neuroscan",
        video_processor_factory=TremorProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.markdown("""
### 🧪 How It Works
- Tracks index fingertip motion
- Removes camera jitter
- Applies FFT
- Detects 4–6 Hz tremor band
- Estimates tremor pattern strength

Stable hand → 0% risk  
Intentional 4–5 Hz shaking → higher risk
""")
