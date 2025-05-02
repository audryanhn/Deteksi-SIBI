import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import os

print("Current working directory:", os.getcwd())
print("Files in Models/:", os.listdir('Models'))

# Load model
modelName = "model_landmarksV3.3.h5"
folderName = "Models/"
MODEL_PATH = os.path.join(folderName, modelName)
model = load_model(MODEL_PATH)

# Load data uji
X_test = np.load('landmarks.npy')
y_test = np.load('labels.npy')

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Prediksi
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Laporan klasifikasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
