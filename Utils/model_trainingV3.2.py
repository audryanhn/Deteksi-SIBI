import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os


# Load data
landmarks = np.load('Data/landmarks.npy')
labels = np.load('Data/labels.npy')

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(landmarks, labels, test_size=0.2, random_state=42)

# Normalisasi data
X_train = np.array(X_train)
X_test = np.array(X_test)

# Bangun model neural network sederhana
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.2),    

    Dense(len(np.unique(labels)), activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Latih model
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
# history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# Simpan model
modelName = 'model_landmarksV3.5.h5'
model.save(os.path.join("Models/" ,modelName) )

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model berhasil disimpan sebagai {modelName}')
print(f'Akurasi model: {accuracy * 100:.2f}%')
