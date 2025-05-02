def trackHands():

    import cv2
    import mediapipe as mp
    import numpy as np
    import tensorflow as tf

    # === Load TFLite Model + Labels ===
    interpreter = tf.lite.Interpreter(model_path="model/gestures.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = np.load("model/labels.npy", allow_pickle=True)

    # === MediaPipe Init ===
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,    
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # === Webcam Capture ===
    cap = cv2.VideoCapture(0)

    def extract_hand_landmarks(results):
        # Map hands by handedness label
        hand_map = {}
        for i, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label.lower()
            hand_map[label] = results.multi_hand_landmarks[i]

        feature_vector = []
        for label in ['right', 'left']:  # order matters!
            if label in hand_map:
                for lm in hand_map[label].landmark:
                    feature_vector.extend([lm.x, lm.y, lm.z])
            else:
                # Pad with zeros if hand not present
                feature_vector.extend([0.0] * 63)

        return np.array(feature_vector, dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_label = None

        if result.multi_hand_landmarks and result.multi_handedness:
            # Draw hands
            for i in range(len(result.multi_hand_landmarks)):
                mp_drawing.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[i],
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            # === Extract landmarks and predict ===
            landmarks_vector = extract_hand_landmarks(result)
            input_data = np.expand_dims(landmarks_vector, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = labels[np.argmax(output)]

            # Display prediction
            cv2.putText(frame, f"Sign: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Sign Speak - Realtime Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

trackHands()