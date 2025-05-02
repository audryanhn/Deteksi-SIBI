import numpy as np
import pandas as pd
import os

# Load file .npy
landmarks = np.load('Data/landmarks.npy')
labels = np.load('Data/labels.npy')

# Gabungkan landmarks dan labels
# Misal: kolom terakhir adalah label
combined = np.hstack([landmarks, labels.reshape(-1, 1)])

# Buat nama kolom
num_landmark_features = landmarks.shape[1]  # biasanya 63 untuk 21 titik (x,y,z)
columns = [f'x{i//3+1}' if i%3==0 else f'y{i//3+1}' if i%3==1 else f'z{i//3+1}' for i in range(num_landmark_features)]
columns.append('label')

# Buat dataframe
df = pd.DataFrame(combined, columns=columns)

# Simpan ke CSV
csv_filename = 'landmarks_with_labels.csv'
df.to_csv(os.path.join("Data/", csv_filename), index=False)

print(f'Data berhasil disimpan ke {csv_filename}!')
