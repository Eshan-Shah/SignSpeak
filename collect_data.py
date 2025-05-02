import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# === Config ===
WORDS = ["Hello", "Bye", "Thanks", "Yes", "No"]
SAMPLES_PER_WORD = 30
DATA_PATH = "data/gestures.csv"
os.makedirs("data", exist_ok=True)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# === Webcam ===
cap = cv2.VideoCapture(0)
all_data = []

def draw_text(frame, text, y=30, color=(255, 255, 255)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# === Data Collection Loop ===
for word in WORDS:
    print(f"✋ Prepare to sign: {word}")
    for i in range(SAMPLES_PER_WORD):
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Countdown before capture
        for countdown in [3, 2, 1]:
            temp_frame = frame.copy()
            draw_text(temp_frame, f"Get ready to sign: {word}", 30)
            draw_text(temp_frame, f"Capturing in... {countdown}", 80, (0, 255, 255))
            cv2.imshow("SignSpeak - Data Collection", temp_frame)
            cv2.waitKey(1000)  # wait 1 second

        # Final capture
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            features = []
            for hand in result.multi_hand_landmarks:
                for lm in hand.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            features.append(word)
            all_data.append(features)
            print(f"✅ Captured {i+1}/{SAMPLES_PER_WORD} for '{word}'")
        else:
            print(f"⚠️ Could not detect both hands — skipping frame.")

        # Short pause between samples
        time.sleep(0.3)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Save Data ===
if all_data:
    num_features = len(all_data[0]) - 1
    columns = [f"f{i}" for i in range(num_features)] + ["label"]
    df = pd.DataFrame(all_data, columns=columns)
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)
    print(f"📁 Saved {len(all_data)} samples to {DATA_PATH}")
else:
    print("⚠️ No data collected.")
