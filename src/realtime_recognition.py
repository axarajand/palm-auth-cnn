import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os

# ===== CONFIG =====
MODEL_PATH = "model/palm_model.h5"
LABEL_PATH = "metadata/labels.json"
CONF_THRESHOLD = 0.5

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== LOAD LABEL =====
with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

# ===== CLASS NAME (WAJIB SORT SESUAI TRAINING) =====
class_names = sorted(label_map.keys())

# ===== DEBUG INFO =====
print("==== DEBUG INFO ====")
print("Jumlah class model :", model.output_shape[-1])
print("Jumlah label       :", len(class_names))
print("Class names        :", class_names)
print("====================")

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            padding = 20
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:
                input_img = preprocess(hand_img)

                preds = model.predict(input_img, verbose=0)
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

                # ===== ANTI CRASH =====
                if class_id >= len(class_names):
                    label = "Unknown"
                    print("[WARNING] class_id out of range:", class_id)
                else:
                    folder_name = class_names[class_id]
                    user_name = label_map[folder_name]

                    # ===== FILTER CONFIDENCE =====
                    if confidence < CONF_THRESHOLD:
                        label = "Unknown"
                    else:
                        label = f"{user_name} ({confidence:.2f})"

                # ===== DISPLAY =====
                cv2.putText(frame, label, (xmin, ymin-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.imshow("ROI", hand_img)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    cv2.imshow("Camera", frame)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()