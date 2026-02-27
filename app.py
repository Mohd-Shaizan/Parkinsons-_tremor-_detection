import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.fft import fft, fftfreq
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ==============================
# CONFIG & REFINED LOGIC CONSTANTS
# ==============================
WINDOW_DURATION = 4
MIN_FREQ = 3.5
MAX_FREQ = 7.5
AMP_THRESHOLD = 8.0  # Increased to ignore minor sensor jitter
POWER_THRESHOLD = 500 # New: Ignore FFT peaks if they don't have enough "energy"

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="NeuroScan AI Pro", layout="wide")

# CSS to keep the camera container centered and sized correctly
st.markdown("""
    <style>
    .element-container iframe {
        width: 700px !important;
        height: 500px !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuroScan AI - Full Diagnostic View")

# --- Control Panel ---
run = st.checkbox("Toggle System Power (Start/Stop)", value=True)

class TremorProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.timestamps = deque()
        self.positions = deque()
        self.freq_history = deque(maxlen=10)

    def compute_refined_metrics(self):
        if len(self.positions) < 40:
            return 0, 0, 0

        signal = np.array(self.positions)
        time_array = np.array(self.timestamps)
        
        # Remove DC Offset and apply basic smoothing
        signal = signal - np.mean(signal)
        
        # Noise Filter: If movement is microscopic, it's just camera noise
        if np.std(signal) < 2.0:
            return 0, 0, 0

        dt = np.mean(np.diff(time_array))
        if dt <= 0: return 0, 0, 0

        yf = fft(signal)
        xf = fftfreq(len(signal), dt)

        pos_mask = (xf[:len(xf)//2] >= MIN_FREQ) & (xf[:len(xf)//2] <= MAX_FREQ)
        band_freqs = xf[:len(xf)//2][pos_mask]
        band_mag = np.abs(yf[:len(yf)//2])[pos_mask]

        if len(band_mag) == 0 or np.max(band_mag) < POWER_THRESHOLD:
            return 0, 0, 0

        dom_idx = np.argmax(band_mag)
        return band_freqs[dom_idx], band_mag[dom_idx], np.std(self.freq_history) if self.freq_history else 1.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not run: # Stop processing if checkbox is off
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        current_time = time.time()

        # RESTORED: Edges and Lines
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Draw the Full Hand Skeleton
                self.mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                )

                # 2. Track Index Tip
                tip = hand_landmarks.landmark[8]
                self.positions.append(tip.y * h)
                self.timestamps.append(current_time)
                cv2.circle(img, (int(tip.x*w), int(tip.y*h)), 10, (0, 255, 255), -1)

        # Window maintenance
        while self.timestamps and (current_time - self.timestamps[0]) > WINDOW_DURATION:
            self.timestamps.popleft()
            self.positions.popleft()

        freq, amp, dev = self.compute_refined_metrics()
        if freq > 0: self.freq_history.append(freq)

        # Logic: High risk only if rhythmic (low dev) AND strong (high amp)
        stability = max(0, 1 - (dev / 2.0)) if freq > 0 else 0
        risk = 0
        if freq > 0 and amp > POWER_THRESHOLD:
            # Weighting: 40% Freq Match, 40% Amplitude Power, 20% Rhythmic Stability
            freq_score = 40 if (4 <= freq <= 6) else 15
            amp_score = min((amp / 2000) * 40, 40)
            risk = freq_score + amp_score + (stability * 20)

        # Visual UI
        cv2.rectangle(img, (20, 20), (350, 180), (10, 10, 10), -1)
        neon = (0, 255, 200)
        cv2.putText(img, "NEUROSCAN ACTIVE", (40, 50), 1, 1.2, neon, 2)
        cv2.putText(img, f"Freq: {freq:.1f} Hz", (40, 90), 1, 1, (255,255,255), 1)
        cv2.putText(img, f"Stability: {stability:.2f}", (40, 120), 1, 1, (255,255,255), 1)
        
        risk_color = (0, 255, 0) if risk < 35 else (0, 150, 255) if risk < 70 else (0, 0, 255)
        cv2.putText(img, f"RISK: {int(risk)}%", (40, 160), 1, 1.5, risk_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="neuroscan-full",
    video_processor_factory=TremorProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
