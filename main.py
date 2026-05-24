import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Fingertip IDs
tip_ids = [4, 8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            landmarks = []

            for id, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmarks.append((id, cx, cy))

            fingers = []

          # Thumb
            if landmarks[4][1] < landmarks[3][1]:
             fingers.append(1)
            else:
             fingers.append(0)

            # Other fingers
            for i in range(1, 5):
                if landmarks[tip_ids[i]][2] < landmarks[tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Count open fingers
            total_fingers = fingers.count(1)

            cv2.putText(
                frame,
                f'Open Fingers: {total_fingers}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f'States: {fingers}',
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow("ZeroTouch Workspace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()