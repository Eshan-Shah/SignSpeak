import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# === Config ===
WORDS = {
    "Hello": "right",
    "Bye": "right",
    "Thanks": "right",
    "Yes": "both",
    "No": "both"
}
SAMPLES_PER_WORD = 30
DATA_PATH = "data/gestures.csv"
os.makedirs("data", exist_ok=True)

# === MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
all_data = []

def countdown_screen(word, hand_usage):
    for count in [3, 2, 1]:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Get ready to sign: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Using {hand_usage} hand(s)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(frame, f"Capturing in... {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("SignSpeak - Data Collection", frame)
        cv2.waitKey(1000)

for word, hand_usage in WORDS.items():
    print(f"✋ Prepare to sign: {word} ({hand_usage} hand)")
    countdown_screen(word, hand_usage)
    collected = 0

    while collected < SAMPLES_PER_WORD:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        display = frame.copy()
        cv2.putText(display, f"Sign: {word} ({hand_usage})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Samples collected: {collected}/{SAMPLES_PER_WORD}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

        if result.multi_hand_landmarks and result.multi_handedness:
            landmarks = []
            hands_detected = {
                result.multi_handedness[i].classification[0].label.lower(): result.multi_hand_landmarks[i]
                for i in range(len(result.multi_hand_landmarks))
            }

            if hand_usage == "both":
                if "left" in hands_detected and "right" in hands_detected:
                    for label in ["left", "right"]:
                        for lm in hands_detected[label].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    cv2.putText(display, "⚠️ Both hands required", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("SignSpeak - Data Collection", display)
                    cv2.waitKey(1)
                    continue
            else:
                hand_key = hand_usage if hand_usage in ["left", "right"] else next(iter(hands_detected))
                if hand_key in hands_detected:
                    for lm in hands_detected[hand_key].landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks += [0.0] * (21 * 3)  # pad for 2nd hand
                else:
                    cv2.putText(display, f"⚠️ {hand_key.title()} hand not detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("SignSpeak - Data Collection", display)
                    cv2.waitKey(1)
                    continue

            landmarks.append(word)
            all_data.append(landmarks)
            collected += 1
            print(f"✅ Collected {collected}/{SAMPLES_PER_WORD} for '{word}'")

        else:
            cv2.putText(display, "⚠️ No hands detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("SignSpeak - Data Collection", display)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

# === Save ===
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
