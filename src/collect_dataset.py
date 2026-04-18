import cv2
import mediapipe as mp
import os, time, random, string
import math
import json

# ===== CONFIG =====
DATASET_PATH = "dataset"
METADATA_PATH = "metadata"
MAX_IMAGES = 100
DELAY = 0.5

# ===== GENERATE ID =====
def generate_id(n=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# ===== INPUT USER =====
user_name = input("Masukkan nama user: ")

labels_path = os.path.join(METADATA_PATH, "labels.json")

# ===== LOAD LABELS =====
try:
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = json.load(f)
    else:
        labels = {}
except:
    print("[WARNING] labels.json rusak, reset ulang")
    labels = {}

# ===== GENERATE ID =====
user_id = generate_id()
while user_id in labels:
    user_id = generate_id()

save_path = os.path.join(DATASET_PATH, user_id)
os.makedirs(save_path, exist_ok=True)

# ===== UPDATE LABEL =====
labels[user_id] = user_name

with open(labels_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"[INFO] User: {user_name}")
print(f"[INFO] ID: {user_id}")

# ===== INSTRUKSI =====
instructions = [
    "Normal",
    "Miring kiri",
    "Miring kanan",
    "Dekatkan",
    "Jauhkan"
]

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(1)

count = 0
last_capture = time.time()

# ===== FUNGSI =====
def angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def is_fingers_open(lm):
    return (
        lm[8].y < lm[6].y and
        lm[12].y < lm[10].y and
        lm[16].y < lm[14].y and
        lm[20].y < lm[18].y
    )

def is_thumb_open(lm):
    palm_width = abs(lm[5].x - lm[17].x)
    thumb_dist = abs(lm[4].x - lm[2].x)
    return thumb_dist > (palm_width * 0.3)

def is_palm_facing_camera(lm):
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    v1 = [index_mcp.x - wrist.x, index_mcp.y - wrist.y]
    v2 = [pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y]

    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return cross > 0

# ===== LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    now = time.time()
    valid = False
    reason = ""

    if result.multi_hand_landmarks is None:
        reason = "Tidak ada tangan"

    elif len(result.multi_hand_landmarks) > 1:
        reason = "Hanya 1 tangan!"

    else:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        h, w, _ = frame.shape
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        padding = 20
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(w, xmax + padding)
        ymax = min(h, ymax + padding)

        hand_img = frame[ymin:ymax, xmin:xmax]
        area = (xmax - xmin) * (ymax - ymin)

        p_wrist = (int(lm[0].x * w), int(lm[0].y * h))
        p_mid = (int(lm[12].x * w), int(lm[12].y * h))
        ang = angle_deg(p_wrist, p_mid)

        instruction = instructions[(count // 20) % len(instructions)]

        if not is_palm_facing_camera(lm):
            reason = "Bukan telapak"
        elif not is_thumb_open(lm):
            reason = "Ibu jari tertutup"
        elif not is_fingers_open(lm):
            reason = "Jari tidak terbuka"
        else:
            if instruction == "Normal" and -110 <= ang <= -70:
                valid = True
            elif instruction == "Miring kiri" and ang < -110:
                valid = True
            elif instruction == "Miring kanan" and ang > -70:
                valid = True
            elif instruction == "Dekatkan" and area > 80000:
                valid = True
            elif instruction == "Jauhkan" and area < 40000:
                valid = True
            else:
                reason = "Posisi tidak sesuai"

        if hand_img.size != 0:
            hand_img = cv2.resize(hand_img, (224, 224))

            if valid and (now - last_capture > DELAY):
                cv2.imwrite(f"{save_path}/{count}.jpg", hand_img)
                count += 1
                last_capture = now

            cv2.imshow("ROI", hand_img)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    instruction = instructions[(count // 20) % len(instructions)]
    status = "VALID" if valid else f"TIDAK VALID ({reason})"
    color = (0,255,0) if valid else (0,0,255)

    cv2.putText(frame, f"Instruksi: {instruction}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Status: {status}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"{count}/{MAX_IMAGES}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Camera", frame)

    if count >= MAX_IMAGES:
        print("[INFO] Selesai")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()