import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for i in range(len(result.multi_hand_landmarks)):
            hand_landmarks = result.multi_hand_landmarks[i]
            hand_label = result.multi_handedness[i].classification[0].label  # 'Left' or 'Right'

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Label the hand on screen
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f'{hand_label} Hand', (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Optional: extract landmarks for later use
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            print(f"{hand_label} hand landmarks: {landmarks}")

    cv2.imshow("Sign Speak - Both Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
