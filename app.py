import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Utils.hand_utils import get_hand_landmarks, mp_draw
import tempfile
import mediapipe as mp
import os
import time

# Load model dan label
@st.cache_resource

def load_model_and_labels():
    model = load_model('Models/model_landmarksV3.4.h5')
    labels = sorted(os.listdir('Dataset-jari/SIBI'))
    return model, labels

model, labels = load_model_and_labels()

st.title("Deteksi Bahasa Isyarat SIBI dengan MediaPipe")
st.markdown("Klik tombol di bawah ini untuk mengaktifkan/menonaktifkan kamera")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "camera" not in st.session_state:
    st.session_state.camera = None


col1,col2 = st.columns([1,4])

# Tombol Toggle kamera
with col1: 
    if st.button("Mulai / Matikan kamera") :
        if st.session_state.camera_on:
            if st.session_state.camera :
                st.session_state.camera.release()
                st.session_state.camera = None
            st.session_state.camera_on = False
        else :
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera_on = True


# Status Camera
with col2 :
    if st.session_state.camera_on :
        st.success("Kamera Aktif ✅")
    else:
        st.warning("Kamera Non-Aktif ❌")

# Akses webcam
FRAME_WINDOW = st.empty()

if st.session_state.camera_on and st.session_state.camera:
    while st.session_state.camera_on:

        ret, frame = st.session_state.camera.read()

        if not ret :
            st.error("Gagal Membuka Kamera")
            break

        landmarks, raw_landmarks = get_hand_landmarks(frame)

        if landmarks is not None:
            prediction = model.predict(landmarks)
            label_idx = np.argmax(prediction)
            pred_label = labels[label_idx]
            confidence = np.max(prediction)

            # cv2.putText(frame, f'{pred_label} ({confidence:.2f})', (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Huruf : {pred_label} ' , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            

            mp_draw.draw_landmarks(frame, raw_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        time.sleep(0.03)

    FRAME_WINDOW.empty()

   