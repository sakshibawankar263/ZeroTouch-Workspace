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

# Fingertip landmark IDs
finger_tips = [4, 8, 12, 16, 20]

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

            landmark_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmark_list.append((id, cx, cy))

                # Draw circle on fingertips
                if id in finger_tips:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Print index finger coordinates
            if landmark_list:
                index_finger = landmark_list[8]

                cv2.putText(
                    frame,
                    f'Index Finger: {index_finger[1]}, {index_finger[2]}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

    cv2.imshow("ZeroTouch Workspace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()