import cv2
import mediapipe as mp
import pandas as pd
import os

# === Settings ===
OUTPUT_CSV = "data/gestures.csv"
SIGN_LABEL = "hello"  # <- change this for each new sign
NUM_SAMPLES = 200     # how many frames to record

# === Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)
sample_count = 0
data = []

print(f"✋ Get ready to record sign: {SIGN_LABEL}")

while cap.isOpened() and sample_count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        frame_data = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])
        
        if len(frame_data) == 126:
            frame_data.append(SIGN_LABEL)
            data.append(frame_data)
            sample_count += 1
            print(f"Saved sample {sample_count}/{NUM_SAMPLES}")

    # Show preview
    cv2.imshow("Data Collection - Both Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# === Save to CSV ===
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
print(f"✅ Done. {sample_count} samples saved to {OUTPUT_CSV}")
