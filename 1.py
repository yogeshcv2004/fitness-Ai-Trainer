import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mediapipe as mp
import time
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset_path = "C:/Users/chara/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting/final_health_exercise_dataset_modified.csv"

df = pd.read_csv(dataset_path)

df.columns = df.columns.str.strip().str.lower()
required_columns = ['elbow_angle', 'knee_angle', 'hip_angle', 'x_position', 'y_position', 'exercise_label']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

X = df[['elbow_angle', 'knee_angle', 'hip_angle', 'x_position', 'y_position']].values
y = df['exercise_label'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = X.reshape((X.shape[0], 1, X.shape[1]))
X = X / np.max(X, axis=0)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, 5)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(len(set(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=3, batch_size=16, verbose=1)

movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

def detect_pose(image):
    img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_tensor = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures["serving_default"](input_tensor)
    keypoints = outputs['output_0'].numpy().reshape(-1, 3)
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def estimate_calories(exercise, reps):
    MET_values = {"Squats": 5, "Push-ups": 8, "Lunges": 6}
    weight_kg = 70  
    duration_min = reps * 0.1  
    return (MET_values.get(exercise, 5) * weight_kg * 3.5) / 200 * duration_min

def posture_correction(elbow_angle, knee_angle):
    feedback = []
    if elbow_angle < 30:
        feedback.append("Extend arms fully")
    if knee_angle > 160:
        feedback.append("Keep knees slightly bent")
    return ", ".join(feedback) if feedback else "Good Form!"

plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
elbow_angles, knee_angles, hip_angles = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
exercise_counts, classifications = deque(maxlen=50), deque(maxlen=50)
count = 0
start_time = time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = detect_pose(frame)
    shoulder, elbow, wrist = keypoints[5][:2], keypoints[7][:2], keypoints[9][:2]
    hip, knee, ankle = keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)

    elbow_angles.append(elbow_angle)
    knee_angles.append(knee_angle)
    hip_angles.append(hip_angle)

    input_data = np.array([[elbow_angle, knee_angle, hip_angle, keypoints[0][0], keypoints[0][1]]]).reshape(1, 1, 5)
    prediction = np.argmax(model.predict(input_data), axis=1)[0]
    exercise_name = label_encoder.inverse_transform([prediction])[0]
    classifications.append(prediction)

    if knee_angle < 90:
        count += 1
    exercise_counts.append(count)

    calories = estimate_calories(exercise_name, count)
    feedback = posture_correction(elbow_angle, knee_angle)

    axes[0].cla()
    axes[0].plot(elbow_angles, label="Elbow Angle")
    axes[0].plot(knee_angles, label="Knee Angle")
    axes[0].plot(hip_angles, label="Hip Angle")
    axes[0].set_title("Joint Angles Over Time")
    axes[0].legend()

    axes[1].cla()
    axes[1].bar(label_encoder.classes_, model.predict(input_data)[0])
    axes[1].set_title("Exercise Classification")

    axes[2].cla()
    axes[2].plot(exercise_counts, label="Repetitions")
    axes[2].set_title("Exercise Repetitions")
    axes[2].legend()

    plt.pause(0.01)

    cv2.putText(frame, f'Elbow: {int(elbow_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Knee: {int(knee_angle)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Exercise: {exercise_name}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Calories: {calories:.2f}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pose Estimation - Fitness App', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.show(block=True)
plt.ioff()