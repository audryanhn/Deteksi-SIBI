import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Muat model yang sudah dilatih
model = load_model('Models/model_landmarksV3.4.h5')

# Muat label
label_names = os.listdir('Dataset-jari/SIBI')

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            # Prediksi menggunakan model
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            label = label_names[class_id]

            # Tampilkan hasil prediksi di layar
            cv2.putText(frame, f'Huruf : {label} ' , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Gambar landmark di tangan
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('SIBI Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()