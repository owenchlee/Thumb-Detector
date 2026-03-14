import cv2
import mediapipe as mp
import pyautogui  # pip install pyautogui
import time
import numpy as np

# Safety feature - pyautogui won't throw an error if mouse goes to edge
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # No delay between commands for smooth movement

grid = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Smoothing - averages last N positions to reduce jitter
class Smoother:
    def __init__(self, size=5):
        self.size = size
        self.points = []

    def smooth(self, point):
        self.points.append(point)
        if len(self.points) > self.size:
            self.points.pop(0)
        return (
            int(sum(p[0] for p in self.points) / len(self.points)),
            int(sum(p[1] for p in self.points) / len(self.points))
        )

smoother = Smoother(size=10)

# State tracking
last_click_time = 0
click_cooldown = 0.5
is_clicking = False
last_action_time = 0
action_cooldown = 2
status_text = ""
status_time = 0

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def draw_grid_with_labels(frame, num_rows=10, num_cols=10):
    h, w = frame.shape[:2]
    for i in range(1, num_rows):
        y = h * i // num_rows
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(frame, str(y), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for i in range(1, num_cols):
        x = w * i // num_cols
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_ui(frame, mode, is_pinching):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Mode indicator
    mode_color = (0, 255, 0) if mode == "MOUSE" else (0, 165, 255)
    cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    # Pinch indicator
    pinch_color = (0, 255, 0) if is_pinching else (100, 100, 100)
    cv2.circle(frame, (w - 40, 25), 15, pinch_color, -1)
    cv2.putText(frame, "CLICK", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pinch_color, 1)

    # Controls legend at bottom
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "M: Mouse Mode | C: Copy/Paste Mode | G: Grid | Q: Quit", 
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Modes: "MOUSE" or "COPY_PASTE"
mode = "MOUSE"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('g'): grid = not grid
    if key == ord('m'): mode = "MOUSE"
    if key == ord('c'): mode = "COPY_PASTE"

    if grid:
        draw_grid_with_labels(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    is_pinching = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Key landmarks
            index_tip = hand_landmarks.landmark[8]   # Index fingertip
            thumb_tip = hand_landmarks.landmark[4]   # Thumb tip
            thumb_base = hand_landmarks.landmark[2]  # Thumb base

            # Convert to pixel coords
            index_px = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))
            thumb_px = (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h))
            thumb_base_px = (int(thumb_base.x * frame_w), int(thumb_base.y * frame_h))

            # Pinch detection (thumb tip close to index tip)
            pinch_distance = get_distance(index_px, thumb_px)
            # CHANGED
            is_pinching = pinch_distance < 10


            current_time = time.time()

            if mode == "MOUSE":
                # Map index finger position to screen coordinates
                # Use 20-80% of frame to avoid edge jitter
                margin_x = frame_w * 0.2
                margin_y = frame_h * 0.2
                mapped_x = np.interp(index_tip.x * frame_w, [margin_x, frame_w - margin_x], [0, screen_w])
                mapped_y = np.interp(index_tip.y * frame_h, [margin_y, frame_h - margin_y], [0, screen_h])
                mapped_x = np.clip(mapped_x, 0, screen_w)
                mapped_y = np.clip(mapped_y, 0, screen_h)

                # Smooth the movement
                smooth_x, smooth_y = smoother.smooth((mapped_x, mapped_y))
                pyautogui.moveTo(smooth_x, smooth_y)

                # Draw cursor indicator on frame
                cv2.circle(frame, index_px, 10, (255, 165, 0), -1)
                cv2.circle(frame, index_px, 12, (255, 255, 255), 2)

                # Pinch = left click
                if is_pinching and current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    status_text = "CLICKED!"
                    status_time = current_time
                    print("Click!")

                # Draw pinch line
                line_color = (0, 255, 0) if is_pinching else (100, 100, 100)
                cv2.line(frame, thumb_px, index_px, line_color, 2)

            elif mode == "COPY_PASTE":
                import keyboard
                is_thumb_up = thumb_px[1] < thumb_base_px[1]

                if current_time - last_action_time > action_cooldown:
                    if is_thumb_up:
                        keyboard.send('ctrl+c')
                        status_text = "COPIED!"
                        status_time = current_time
                        last_action_time = current_time
                        print("Copied!")
                    else:
                        keyboard.send('ctrl+v')
                        status_text = "PASTED!"
                        status_time = current_time
                        last_action_time = current_time
                        print("Pasted!")

                label = "Thumb Up (Copy)" if is_thumb_up else "Thumb Down (Paste)"
                cv2.putText(frame, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Flash status message
    if time.time() - status_time < 1.0:
        cv2.putText(frame, status_text,
                    (frame_w // 2 - 80, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    draw_ui(frame, mode, is_pinching)
    cv2.imshow('Gesture Control', frame)
    cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)

cap.release()
cv2.destroyAllWindows()