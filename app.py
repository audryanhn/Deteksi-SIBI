import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
from Utils.hand_utils import get_hand_landmarks, mp_draw
import json

# Load model dan label
@st.cache_resource

def load_model_and_labels():
    try :
        model = load_model('Models/model_landmarksV3.4.h5')
        with open("Data/labels.json", "r") as f :
            labels = json.load(f)
        return model, labels
    except Exception as e :
        st.error(f"Gagal Memuat kode atau label: {e}")
        raise e

model, labels = load_model_and_labels()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

st.title("Deteksi Bahasa Isyarat SIBI dengan MediaPipe")
st.markdown("Klik tombol di bawah ini untuk mengaktifkan/menonaktifkan kamera")


if st.button("🔁 Mulai / Matikan Kamera"):
    st.session_state.camera_on = not st.session_state.camera_on

if st.session_state.camera_on:
    st.success("Kamera Aktif ✅")
else:
    st.warning("Kamera Non-Aktif ❌")
# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks).reshape(1, -1)

                try : 
                    prediction = model.predict(landmarks)
                    class_id = np.argmax(prediction)
                    label = labels[class_id]
                    cv2.putText(img, f"Huruf: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                except Exception as e :
                    print(f"Prediction Error, {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Aktifkan stream jika kamera aktif
if st.session_state.camera_on:
    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )