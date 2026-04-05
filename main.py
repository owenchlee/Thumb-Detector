import cv2
import mediapipe as mp
import pyautogui
import keyboard
import time
import numpy as np
import threading
import platform

# ─── PyAutoGUI config ────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# ─── Screen info ─────────────────────────────────────────────────────────────
screen_w, screen_h = pyautogui.size()

# ─── MediaPipe — load in background thread while camera warms up ──────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = None

def init_mediapipe():
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

mediapipe_thread = threading.Thread(target=init_mediapipe, daemon=True)
mediapipe_thread.start()

# ─── Camera — use fast backend ───────────────────────────────────────────────
if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif platform.system() == "Darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up camera while MediaPipe loads
for _ in range(5):
    cap.read()

# Wait for MediaPipe to finish loading
mediapipe_thread.join()

# ─── Smoother ────────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, size=10):
        self.size = size
        self.pts = []

    def smooth(self, point):
        self.pts.append(point)
        if len(self.pts) > self.size:
            self.pts.pop(0)
        return (
            int(sum(p[0] for p in self.pts) / len(self.pts)),
            int(sum(p[1] for p in self.pts) / len(self.pts)),
        )

smoother = Smoother()

# ─── State ───────────────────────────────────────────────────────────────────
mode = "MOUSE"          # "MOUSE" or "COPY_PASTE"
grid = False

last_click_time   = 0.0
last_action_time  = 0.0
click_cooldown    = 0.5
action_cooldown   = 2.0

status_text = ""
status_time = 0.0

# Pre-compute margins (only recalculate if resolution changes)
frame_w, frame_h = 640, 480
margin_x = frame_w * 0.2
margin_y = frame_h * 0.2

# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_distance(p1, p2):
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5          # avoids np overhead per-frame

def draw_grid(frame):
    h, w = frame.shape[:2]
    for i in range(1, 10):
        y = h * i // 10
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(frame, str(y), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x = w * i // 10
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_ui(frame, mode, is_pinching, status_text, status_time):
    h, w = frame.shape[:2]
    now = time.time()

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    mode_color = (0, 255, 0) if mode == "MOUSE" else (0, 165, 255)
    cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    pinch_color = (0, 255, 0) if is_pinching else (100, 100, 100)
    cv2.circle(frame, (w - 40, 25), 15, pinch_color, -1)
    cv2.putText(frame, "CLICK", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pinch_color, 1)

    # Bottom bar
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "M: Mouse | C: Copy/Paste | G: Grid | Q: Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Flash status
    if now - status_time < 1.0:
        cv2.putText(frame, status_text,
                    (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# ─── Main loop ───────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        grid = not grid
    elif key == ord('m'):
        mode = "MOUSE"
    elif key == ord('c'):
        mode = "COPY_PASTE"

    if grid:
        draw_grid(frame)

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # small speedup — MediaPipe reads only
    result = hands.process(rgb)
    rgb.flags.writeable = True

    is_pinching = False
    current_time = time.time()

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]   # only need first hand

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        lm = hand_landmarks.landmark
        index_tip   = lm[8]
        thumb_tip   = lm[4]
        thumb_base  = lm[2]

        index_px     = (int(index_tip.x  * frame_w), int(index_tip.y  * frame_h))
        thumb_px     = (int(thumb_tip.x  * frame_w), int(thumb_tip.y  * frame_h))
        thumb_base_px = (int(thumb_base.x * frame_w), int(thumb_base.y * frame_h))

        pinch_dist = get_distance(index_px, thumb_px)
        is_pinching = pinch_dist < 10

        # ── MOUSE mode ───────────────────────────────────────────────────────
        if mode == "MOUSE":
            mapped_x = np.interp(index_tip.x * frame_w, [margin_x, frame_w - margin_x], [0, screen_w])
            mapped_y = np.interp(index_tip.y * frame_h, [margin_y, frame_h - margin_y], [0, screen_h])
            mapped_x = np.clip(mapped_x, 0, screen_w)
            mapped_y = np.clip(mapped_y, 0, screen_h)

            sx, sy = smoother.smooth((mapped_x, mapped_y))
            pyautogui.moveTo(sx, sy)

            cv2.circle(frame, index_px, 10, (255, 165, 0), -1)
            cv2.circle(frame, index_px, 12, (255, 255, 255), 2)

            line_color = (0, 255, 0) if is_pinching else (100, 100, 100)
            cv2.line(frame, thumb_px, index_px, line_color, 2)

            if is_pinching and current_time - last_click_time > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                status_text = "CLICKED!"
                status_time = current_time
                print("Click!")

        # ── COPY/PASTE mode ──────────────────────────────────────────────────
        elif mode == "COPY_PASTE":
            is_thumb_up = thumb_px[1] < thumb_base_px[1]

            if current_time - last_action_time > action_cooldown:
                if is_thumb_up:
                    keyboard.send('ctrl+c')
                    status_text = "COPIED!"
                else:
                    keyboard.send('ctrl+v')
                    status_text = "PASTED!"
                status_time = current_time
                last_action_time = current_time
                print(status_text)

            label = "Thumb Up → Copy" if is_thumb_up else "Thumb Down → Paste"
            cv2.putText(frame, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    draw_ui(frame, mode, is_pinching, status_text, status_time)
    cv2.imshow('Gesture Control', frame)
    cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)

cap.release()
cv2.destroyAllWindows()