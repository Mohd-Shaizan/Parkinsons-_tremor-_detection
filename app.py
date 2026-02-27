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
# CONFIG
# ==============================
WINDOW_DURATION = 4      # Slightly shorter window for faster response
MIN_FREQ = 3.5           # Pathological tremors usually start above 3.5Hz
MAX_FREQ = 7.5
STABILITY_HISTORY = 10   # Longer history for better stability average
AMP_THRESHOLD = 5.0      # CRITICAL: Ignore anything below this movement magnitude

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="NeuroScan AI Pro", layout="wide")

# CSS to control the camera container size
st.markdown("""
    <style>
    .element-container iframe {
        width: 640px !important;
        height: 480px !important;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuroScan AI - Refined Tremor Analysis")

class TremorProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8, # Higher confidence for better accuracy
            min_tracking_confidence=0.8
        )
        self.timestamps = deque()
        self.positions = deque()
        self.freq_history = deque(maxlen=STABILITY_HISTORY)

    def compute_tremor_frequency(self):
        if len(self.positions) < 40: # Need more points for accurate FFT
            return None, None

        time_array = np.array(self.timestamps)
        signal = np.array(self.positions)
        
        # Simple noise reduction: Moving Average
        signal = np.convolve(signal, np.ones(3)/3, mode='valid')
        time_array = time_array[2:] 

        # Remove DC Offset
        signal = signal - np.mean(signal)

        # Standard deviation check: if the hand is very still, don't run FFT
        if np.std(signal) < 1.5: 
            return 0, 0

        dt = np.mean(np.diff(time_array))
        if dt <= 0: return None, None

        yf = fft(signal)
        xf = fftfreq(len(signal), dt)

        positive_freqs = xf[:len(xf)//2]
        magnitude = np.abs(yf[:len(yf)//2])

        band_mask = (positive_freqs >= MIN_FREQ) & (positive_freqs <= MAX_FREQ)
        if not np.any(band_mask): return 0, 0

        band_freqs = positive_freqs[band_mask]
        band_magnitude = magnitude[band_mask]

        dom_idx = np.argmax(band_magnitude)
        dominant_freq = band_freqs[dom_idx]
        amplitude = band_magnitude[dom_idx]

        return dominant_freq, amplitude

    def compute_risk(self, freq, amplitude, stability):
        # LOGIC REFINEMENT:
        # 1. If amplitude is low, risk is 0.
        if amplitude < AMP_THRESHOLD or freq == 0:
            return 0.0
        
        # 2. Pathological tremors are very "Stable" (rhythmic). 
        # If stability is low, it's likely just random voluntary movement.
        if stability < 0.6:
            return round((amplitude / 100) * 10, 1) # Low risk for erratic movement

        # 3. Frequency weighting (Parkinsonian rest tremors are typically 4-6Hz)
        freq_weight = 1.0 if 4.0 <= freq <= 6.0 else 0.4
        
        risk = (freq_weight * 40) + (min(amplitude/2, 40)) + (stability * 20)
        return round(min(risk, 99.9), 1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        current_time = time.time()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            # Convert normalized to pixel coordinates
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            
            self.positions.append(iy) # Track vertical tremor (usually more distinct)
            self.timestamps.append(current_time)
            
            # Draw tracking dot
            cv2.circle(img, (ix, iy), 8, (0, 255, 255), -1)

        # Window maintenance
        while self.timestamps and (current_time - self.timestamps[0]) > WINDOW_DURATION:
            self.timestamps.popleft()
            self.positions.popleft()

        freq, amp = self.compute_tremor_frequency()
        if freq and freq > 0: self.freq_history.append(freq)
        
        # Stability Calculation
        stability = 0
        if len(self.freq_history) >= 5:
            std = np.std(self.freq_history)
            stability = max(0, 1 - (std / 1.5))

        risk = self.compute_risk(freq, amp if amp else 0, stability)

        # UI Panel
        cv2.rectangle(img, (10, 10), (320, 160), (30, 30, 30), -1)
        color = (0, 255, 0) if risk < 30 else (0, 165, 255) if risk < 70 else (0, 0, 255)
        
        cv2.putText(img, f"Freq: {freq:.1f}Hz" if freq else "Scanning...", (20, 40), 1, 1.2, (255,255,255), 2)
        cv2.putText(img, f"Stability: {stability:.2f}", (20, 75), 1, 1.1, (255,255,255), 1)
        cv2.putText(img, f"RISK: {risk}%", (20, 130), 1, 1.8, color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="neuroscan-refined",
    video_processor_factory=TremorProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
