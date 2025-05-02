import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Path ke folder dataset
DATASET_PATH = 'Dataset-jari/SIBI/'

# File output
OUTPUT_FILE = 'landmarks.npy'
LABELS_FILE = 'labels.npy'

IMG_SIZE = 128

all_landmarks = []
all_labels = []
label_names = os.listdir(DATASET_PATH)

failed_counts ={}

for label_index, label_name in enumerate(label_names):
    folder_path = os.path.join(DATASET_PATH, label_name)
    if not os.path.isdir(folder_path):
        continue

    fail_count = 0

    for img_name in tqdm(os.listdir(folder_path), desc=f'Processing {label_name}'):  
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None :
                # print(f"failed to read : {img_path}")
                fail_count+=1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            if results.multi_hand_landmarks:  
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []

                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                all_landmarks.append(landmarks)
                all_labels.append(label_index)
            else:
                # print(f"no hand detected : {img_path}")
                fail_count+=1
        except Exception as e:
            print(f'Error processing {img_name}: {e}')
            fail_count+=1
    failed_counts[label_name] =fail_count

hands.close()

# Simpan landmarks dan labels
np.save(os.path.join("Data/", OUTPUT_FILE), np.array(all_landmarks))
np.save(os.path.join("Data/", LABELS_FILE), np.array(all_labels))

print("\nRingkasan gambar gagal ekstrak : ")
for label, count in failed_counts.items():
    print(f"{label} : {count} gambar gagal")

print('Ekstraksi landmarks selesai dan berhasil disimpan!')
