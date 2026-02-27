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



WINDOW_DURATION = 5

MIN_FREQ = 3

MAX_FREQ = 8

STABILITY_HISTORY = 5
MOTION_THRESHOLD = 2.0       # minimum pixel motion
POWER_THRESHOLD = 15         # minimum FFT power
SMOOTHING_WINDOW = 5
MIN_SIGNAL_DURATION = 3      # seconds



rtc_configuration = RTCConfiguration(

    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

)



# ==============================

# STREAMLIT UI

# ==============================



st.set_page_config(page_title="NeuroScan AI", layout="wide")

st.title("🧠 NeuroScan AI - Micro Tremor Quantification System")



st.markdown("""

This system tracks fingertip micro-movements in real time  

and analyzes tremor frequency & rhythmic stability.



⚠ Research Prototype — Not a Medical Diagnosis Tool.

""")



# ==============================

# VIDEO PROCESSOR

# ==============================



class TremorProcessor(VideoProcessorBase):



    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(

            max_num_hands=1,

            min_detection_confidence=0.7,

            min_tracking_confidence=0.7

        )

        self.timestamps = deque()

        self.positions = deque()
        self.magnitudes = deque()

        self.freq_history = deque(maxlen=STABILITY_HISTORY)



    def compute_tremor_frequency(self):

    if len(self.magnitudes) < 30:
        return None, None

    signal = np.array(self.magnitudes)

    # Smooth signal
    if len(signal) > SMOOTHING_WINDOW:
        kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
        signal = np.convolve(signal, kernel, mode='valid')

    # Motion gate
    avg_motion = np.mean(signal)
    if avg_motion < MOTION_THRESHOLD:
        return None, None

    time_array = np.array(self.timestamps)[-len(signal):]
    dt = np.mean(np.diff(time_array))
    if dt <= 0:
        return None, None

    signal = signal - np.mean(signal)

    yf = fft(signal)
    xf = fftfreq(len(signal), dt)

    positive_freqs = xf[:len(xf)//2]
    magnitude = np.abs(yf[:len(yf)//2])

    band_mask = (positive_freqs >= MIN_FREQ) & (positive_freqs <= MAX_FREQ)
    if not np.any(band_mask):
        return None, None

    band_freqs = positive_freqs[band_mask]
    band_magnitude = magnitude[band_mask]

    dominant_index = np.argmax(band_magnitude)
    dominant_freq = band_freqs[dominant_index]
    power = band_magnitude[dominant_index]

    if power < POWER_THRESHOLD:
        return None, None

    return dominant_freq, power

    
    def compute_stability(self):

        if len(self.freq_history) < STABILITY_HISTORY:

            return 0

        std_dev = np.std(self.freq_history)

        return max(0, 1 - (std_dev / 2))



    def compute_risk(self, freq, power, stability):

    if freq is None:
        return 0

    # Parkinson tremor typical band: 4–6 Hz
    freq_score = 1 if 4 <= freq <= 6 else 0.3

    # Normalize power
    power_score = min(power / 80, 1)

    total_score = (
        (freq_score * 0.5) +
        (power_score * 0.3) +
        (stability * 0.2)
    )

    risk = round(total_score * 100, 1)

    # Final safety gate
    if risk < 40:
        return 0

    return risk


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        results = self.hands.process(rgb)

        current_time = time.time()



        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                h, w, _ = img.shape



                # Draw edges

                for connection in self.mp_hands.HAND_CONNECTIONS:

                    start_idx, end_idx = connection

                    x1 = int(hand_landmarks.landmark[start_idx].x * w)

                    y1 = int(hand_landmarks.landmark[start_idx].y * h)

                    x2 = int(hand_landmarks.landmark[end_idx].x * w)

                    y2 = int(hand_landmarks.landmark[end_idx].y * h)

                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 180), 2)



                # Draw vertices

                for idx, lm in enumerate(hand_landmarks.landmark):

                    x = int(lm.x * w)

                    y = int(lm.y * h)

                    cv2.circle(img, (x, y), 5, (0, 255, 255), -1)



                # Track index fingertip

                index_tip = hand_landmarks.landmark[8]

                x = int(index_tip.x * w)

                y = int(index_tip.y * h)

                cv2.circle(img, (x, y), 10, (255, 255, 0), -1)



                self.positions.append((x, y))

                if len(self.positions) > 1:
                    dx = self.positions[-1][0] - self.positions[-2][0]
                    dy = self.positions[-1][1] - self.positions[-2][1]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    self.magnitudes.append(magnitude)

                self.timestamps.append(current_time)



        # Maintain sliding window

        while self.timestamps and (current_time - self.timestamps[0]) > WINDOW_DURATION:

            self.timestamps.popleft()

            self.positions.popleft()



        freq, amplitude = self.compute_tremor_frequency()



        if freq:

            self.freq_history.append(freq)



        stability = self.compute_stability()

        risk = self.compute_risk(freq, amplitude if amplitude else 0, stability)



        # Futuristic Panel

        overlay = img.copy()

        cv2.rectangle(overlay, (20, 20), (480, 240), (20, 20, 20), -1)

        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)



        neon = (0, 255, 200)

        cv2.putText(img, "NEUROSCAN AI", (40, 50),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)



        freq_text = f"Frequency: {freq:.2f} Hz" if freq else "Frequency: --"

        amp_text = f"Amplitude: {amplitude:.2f}" if amplitude else "Amplitude: --"



        cv2.putText(img, freq_text, (40, 100),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)

        cv2.putText(img, amp_text, (40, 130),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)

        cv2.putText(img, f"Stability: {round(stability,2)}", (40, 160),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)

        cv2.putText(img, f"Risk Index: {risk}%", (40, 200),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)



        return av.VideoFrame.from_ndarray(img, format="bgr24")





# ==============================

# WEBRTC STREAM

# ==============================



webrtc_streamer(

    key="neuroscan",

    video_processor_factory=TremorProcessor,

    rtc_configuration=rtc_configuration,

)
