import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Muat model yang sudah dilatih
model = load_model('Models/model_landmarksV3.3.h5')


# Muat scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Daftar label yang digunakan
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar landmark di tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ekstrak landmark
            landmark_coords = []
            for landmark in hand_landmarks.landmark:
                landmark_coords.extend([landmark.x, landmark.y, landmark.z])

            # Normalisasi menggunakan scaler yang sudah dilatih
            landmark_coords = np.array(landmark_coords).reshape(1, -1)
            landmark_coords = scaler.transform(landmark_coords)

            # Prediksi menggunakan model
            prediction = model.predict(landmark_coords)
            predicted_label = labels[np.argmax(prediction)]

            # Tampilkan hasil prediksi
            cv2.putText(frame, f'Prediksi: {predicted_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Pendeteksian Real-Time Bahasa Isyarat', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
