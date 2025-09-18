import sys
import os
import argparse
import subprocess

# GUI imports
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QMainWindow, QGridLayout, QGroupBox,
                             QSpacerItem, QSizePolicy)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt


# ------------------------------
# Shared: Instruction window
# ------------------------------
IMAGE_FILENAMES = {
    "session5": "S5.png",
    "session6": "S6.png",
    "session7": "S3_video.png",
    "session8": "S8.png",
    "session3": "S3.png",
    "session4": "S4.png",
    "session9": "S9.png",
    "session10": "S10.png",
    "ai_air_canvas": "S9.png"
}


class InstructionWindow(QWidget):
    def __init__(self, title: str, instructions: str, image_filename: str):
        super().__init__()
        self.setWindowTitle(title)
        self.setFixedSize(600, 600)
        self.layout = QVBoxLayout()
        self.setStyleSheet("background-color: #2e2e2e; color: #fff;")

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel()
        image_path = os.path.join("Images", image_filename)
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(500, 300,
                                                     Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText("Image not found: " + image_filename)
            self.image_label.setStyleSheet("color: #e74c3c;")
            self.image_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.instructions_label = QLabel(instructions)
        self.instructions_label.setFont(QFont('Arial', 14))
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions_label.setWordWrap(True)

        self.close_button = QPushButton("Close and Stop Mode")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("background-color: #c0392b; color: white; border-radius: 10px; padding: 10px; font-weight: bold;")

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.close_button)
        self.setLayout(self.layout)


# ------------------------------
# GUI Main Window
# ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #fff;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        title_label = QLabel("Gesture Control System GUI")
        title_label.setFont(QFont('Helvetica', 40, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)

        self.main_layout.addSpacerItem(QSpacerItem(20, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.buttons_layout = QGridLayout()

        sections = [
            {"title": "Movie & Presentation Mode", "buttons": [
                {"text": "Play/Pause", "mode": "session5", "instructions": "Make a closed fist to toggle Play/Pause for media."},
                {"text": "Volume Control", "mode": "session6", "instructions": "Control volume with a right-hand pinch gesture.\n\nUse your left hand to stop."},
                {"text": "Jump Mode", "mode": "session7", "instructions": "Use your right index finger to go to the next slide or skip 5 seconds forward.\n\nUse your left index finger to go to the previous slide or skip 5 seconds backward."}
            ]},
            {"title": "Mouse Control Mode", "buttons": [
                {"text": "Mouse Control", "mode": "session8", "instructions": "Move the cursor with your index finger.\n\nA new camera window will appear for gesture input.\n\n- Thumb-Index pinch for left-click.\n- Thumb-Middle pinch for right-click."}
            ]},
            {"title": "Testing", "buttons": [
                {"text": "Hand Detection", "mode": "session3", "instructions": "This mode shows how the system detects and tracks hands and landmarks."},
                {"text": "Gesture Detection", "mode": "session4", "instructions": "This mode shows how the system counts fingers and detects basic gestures."}
            ]},
            {"title": "Air Drawing & Games", "buttons": [
                {"text": "Air Canvas", "mode": "session9", "instructions": "Point with your index finger to draw in the air.\n\nMake a fist to clear the canvas."},
                {"text": "Flappy Bird Game", "mode": "session10", "instructions": "Wave your open palm to make the bird jump.\n\nSurvive as long as you can!"},
                {"text": "Air Canvas + AI Art", "mode": "ai_air_canvas", "instructions": "Use your index finger to draw.\n\nPinch (thumb-index) to generate an AI image from your sketch.\n\nPress 'C' to clear, 'Q' to quit."}
            ]}
        ]

        for col, section in enumerate(sections):
            heading_label = self.create_heading_label(section["title"])
            group_box = self.create_group_box()
            group_layout = QVBoxLayout()
            group_layout.addStretch()
            for button_data in section["buttons"]:
                button = self.create_button(button_data["text"], button_data["mode"], button_data["instructions"])
                group_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)
            group_layout.addStretch()
            group_box.setLayout(group_layout)
            self.buttons_layout.addWidget(heading_label, 0, col)
            self.buttons_layout.addWidget(group_box, 1, col)

        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addSpacerItem(QSpacerItem(20, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.quit_button = QPushButton("Quit Application")
        self.quit_button.setFont(QFont('Arial', 16))
        self.quit_button.setFixedSize(250, 60)
        self.quit_button.setStyleSheet("background-color: #e74c3c; color: white; border-radius: 10px; font-weight: bold;")
        self.quit_button.clicked.connect(self.close)
        self.main_layout.addWidget(self.quit_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.current_process = None

    def create_heading_label(self, title):
        label = QLabel(title)
        label.setFont(QFont('Arial', 20, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def create_group_box(self):
        group_box = QGroupBox()
        group_box.setStyleSheet("QGroupBox { border: 2px solid #3498db; border-radius: 5px;}")
        return group_box

    def create_button(self, text, mode, instructions):
        button = QPushButton(text)
        button.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        button.setFixedSize(280, 70)
        button.setStyleSheet("background-color: #27ae60; color: white; border-radius: 10px; padding: 5px;")
        button.clicked.connect(lambda: self.launch_mode(mode, text, instructions))
        return button

    def launch_mode(self, mode, title, instructions):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.current_process.wait()

        image_filename = IMAGE_FILENAMES.get(mode, "")
        self.instructions_window = InstructionWindow(title, instructions, image_filename)
        self.instructions_window.show()

        script_path = os.path.abspath(__file__)
        self.current_process = subprocess.Popen([sys.executable, script_path, "--mode", mode], cwd=os.path.dirname(script_path))
        self.instructions_window.closeEvent = lambda event: self.handle_close_event(event)

    def handle_close_event(self, event):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.current_process.wait()
        event.accept()

    def closeEvent(self, event):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.current_process.wait()
        event.accept()


# ------------------------------
# Modes implementations
# ------------------------------
def run_session3():
    import cv2 as cv
    import mediapipe as mp
    import numpy as np

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    colors = [(255, 0, 0), (0, 255, 0)]

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_color = colors[idx % len(colors)]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for tip_id in [4, 8, 12, 16, 20]:
                    lm = hand_landmarks.landmark[tip_id]
                    x, y = int(lm.x * w), int(lm.y * h)
                    radius = int(8 + 5 * abs(np.sin(frame_count * 0.1)))
                    cv.circle(frame, (x, y), radius, hand_color, -1)
                label = results.multi_handedness[idx].classification[0].label
                cv.putText(frame, f"{label} Hand", (50, 100 + idx * 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        else:
            cv.putText(frame, "Show your hands!", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_count += 1
        cv.imshow('Hand Detection Magic', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cam.release()
    cv.destroyAllWindows()


def run_session4():
    import cv2 as cv
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def count_fingers(landmarks):
        fingers = []
        if landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, 5):
            tip_id = finger_tips[i]
            pip_id = finger_pips[i]
            if landmarks[tip_id].y < landmarks[pip_id].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers, fingers.count(1)

    def detect_gesture(fingers, total):
        patterns = {
            (0,1,1,0,0): "Peace Sign",
            (1,0,0,0,0): "Thumbs Up",
            (1,1,1,1,1): "High Five",
            (0,0,0,0,0): "Fist"
        }
        return patterns.get(tuple(fingers), f"{total} Fingers Up")

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        h, w, _ = frame.shape

        cv.putText(frame, "FINGER GYMNASTICS", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers, total = count_fingers(hand_landmarks.landmark)
                gesture = detect_gesture(fingers, total)
                cv.putText(frame, f"Fingers: {total}", (50, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv.putText(frame, f"Gesture: {gesture}", (50, 170), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                for i, name in enumerate(finger_names):
                    color = (0, 255, 0) if fingers[i] else (0, 0, 255)
                    cv.putText(frame, f"{name}: {'UP' if fingers[i] else 'DOWN'}", (50, 220 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv.putText(frame, "Show your hand!", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv.imshow('Finger Gymnastics', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cam.release()
    cv.destroyAllWindows()


def run_session5():
    import cv2 as cv
    import mediapipe as mp
    import pyautogui

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    last_gesture = "none"

    def is_fist(landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y:
                return False
        return True

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        current_gesture = "none"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_fist(hand_landmarks):
                    current_gesture = "fist"
                    if last_gesture != current_gesture:
                        pyautogui.press('playpause')
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        last_gesture = current_gesture
        cv.putText(frame_bgr, "Make a closed fist to play/pause", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Gesture DJ Mixer', frame_bgr)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cam.release()
    cv.destroyAllWindows()


def run_session6():
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import math
    import time
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    volume = AudioUtilities.GetSpeakers().Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None).QueryInterface(IAudioEndpointVolume)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    volume_locked = False
    last_valid_volume = volume.GetMasterVolumeLevel()
    last_valid_frame_time = time.time()
    volume_hold_time = 1.0

    def get_pinch_distance(landmarks):
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        return math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        current_message = ""
        vol_range = volume.GetVolumeRange()
        min_vol, max_vol = vol_range[0], vol_range[1]

        if results.multi_hand_landmarks:
            left_hand_present = False
            right_hand_landmarks = None
            for idx, hand in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                if handedness == "Left":
                    left_hand_present = True
                elif handedness == "Right":
                    right_hand_landmarks = hand
            if left_hand_present:
                volume_locked = True
            else:
                volume_locked = False
            if not volume_locked and right_hand_landmarks:
                last_valid_frame_time = time.time()
                distance = get_pinch_distance(right_hand_landmarks)
                vol_level = np.interp(distance, [0.04, 0.15], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol_level, None)
                last_valid_volume = vol_level
                current_vol_pct = np.interp(volume.GetMasterVolumeLevel(), [min_vol, max_vol], [0, 100])
                current_message = f"Volume: {int(current_vol_pct)}%"

        if not results.multi_hand_landmarks:
            if time.time() - last_valid_frame_time < volume_hold_time:
                volume.SetMasterVolumeLevel(last_valid_volume, None)
                current_message = "Volume is stable."
            else:
                current_message = "Waiting for hand..."

        if volume_locked:
            current_vol_pct = np.interp(last_valid_volume, [min_vol, max_vol], [0, 100])
            current_message = f"Volume is Locked: STOP at {int(current_vol_pct)}%"
        cv.putText(frame, current_message, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Stable Volume Control with Stop', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cam.release()
    cv.destroyAllWindows()


def run_session7():
    import cv2 as cv
    import mediapipe as mp
    import pyautogui

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    last_gesture = "none"

    def is_index_finger_up(landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        is_index_up = landmarks.landmark[finger_tips[0]].y < landmarks.landmark[finger_pips[0]].y
        other_fingers_down = True
        for i in range(1, 4):
            if landmarks.landmark[finger_tips[i]].y < landmarks.landmark[finger_pips[i]].y:
                other_fingers_down = False
                break
        return is_index_up and other_fingers_down

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        current_gesture = "none"
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                if is_index_finger_up(hand_landmarks):
                    if hand_label == "Right":
                        current_gesture = "next_slide"
                        if last_gesture != "next_slide":
                            pyautogui.press('right')
                    elif hand_label == "Left":
                        current_gesture = "prev_slide"
                        if last_gesture != "prev_slide":
                            pyautogui.press('left')
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        last_gesture = current_gesture
        cv.putText(frame_bgr, f"Status: {last_gesture}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Presenter Mode', frame_bgr)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cam.release()
    cv.destroyAllWindows()


def run_session8():
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import pyautogui, math, time

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    screen_w, screen_h = pyautogui.size()

    prev_x, prev_y = 0, 0
    smoothing = 7
    clicking_left, clicking_right = False, False
    pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0

    def smooth(x, y, px, py, f):
        if px == 0 and py == 0:
            return x, y
        return px + (x - px)/f, py + (y - py)/f

    def pinch(landmarks, finger1_id, finger2_id, th=0.05):
        f1_tip = landmarks.landmark[finger1_id]
        f2_tip = landmarks.landmark[finger2_id]
        return math.hypot(f1_tip.x - f2_tip.x, f1_tip.y - f2_tip.y) < th

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = frame.shape
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            ix, iy = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
            prev_x, prev_y = smooth(ix, iy, prev_x, prev_y, smoothing)
            sx, sy = np.interp(prev_x, (0,w), (0,screen_w)), np.interp(prev_y, (0,h), (0,screen_h))
            pyautogui.moveTo(sx, sy)
            if pinch(lm, 4, 8) and not clicking_left:
                pyautogui.click()
                clicking_left = True
            if not pinch(lm, 4, 8):
                clicking_left = False
            if pinch(lm, 4, 12) and not clicking_right:
                pyautogui.rightClick()
                clicking_right = True
            if not pinch(lm, 4, 12):
                clicking_right = False
            r = int(8 + 4*abs(math.sin(time.time()*3)))
            cv.circle(frame, (ix, iy), r, (0,255,0), 2)
            cv.circle(frame, (ix, iy), 5, (0,255,0), -1)
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y, clicking_left, clicking_right = 0, 0, False, False
        cv.imshow("Gesture Clicker", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close(); cam.release(); cv.destroyAllWindows()


def run_session9():
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import time

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = np.zeros((720, 1280, 3), dtype='uint8')
    drawing_color = (0, 0, 255)
    last_drawing_point = None
    fist_start_time = None
    fist_hold_duration = 2

    def is_pointing_up(landmarks):
        if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
            other_finger_tips = [12, 16, 20]
            for tip_id in other_finger_tips:
                if landmarks.landmark[tip_id].y < landmarks.landmark[tip_id-2].y:
                    return False
            if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
                return False
            return True
        return False

    def is_fist(landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y:
                return False
        return True

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            current_x, current_y = int(index_tip.x * w), int(index_tip.y * h)
            if is_pointing_up(hand_landmarks):
                fist_start_time = None
                if last_drawing_point is not None:
                    cv.line(canvas, last_drawing_point, (current_x, current_y), drawing_color, 15)
                last_drawing_point = (current_x, current_y)
                cv.circle(frame_bgr, (current_x, current_y), 15, drawing_color, -1)
            else:
                last_drawing_point = None
            if is_fist(hand_landmarks):
                if fist_start_time is None:
                    fist_start_time = time.time()
                elapsed_time = time.time() - fist_start_time
                cv.putText(frame_bgr, f"Holding fist... {fist_hold_duration - elapsed_time:.1f}s left", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if elapsed_time > fist_hold_duration:
                    canvas = np.zeros((720, 1280, 3), dtype='uint8')
                    fist_start_time = None
            else:
                fist_start_time = None
        else:
            last_drawing_point = None
            fist_start_time = None
        combined_frame = cv.addWeighted(frame_bgr, 0.7, canvas, 1.0, 0)
        cv.imshow('Air Canvas', combined_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release(); cv.destroyAllWindows()


def run_session10():
    import cv2 as cv
    import mediapipe as mp
    import pygame
    import numpy as np
    import urllib.request
    import os
    import random
    import time

    def download_model(url, filename):
        if not os.path.exists(filename):
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception:
                pass

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    MODEL_FILENAME = "gesture_recognizer.task"
    download_model(MODEL_URL, MODEL_FILENAME)

    pygame.init()
    window_width, window_height = 800, 600
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.setCaption = pygame.display.set_caption  # alias to avoid linter warning
    pygame.display.set_caption("Gesture-Controlled Flappy Bird")
    font = pygame.font.Font(None, 36)

    bird_x, bird_y = 50, window_height // 2
    bird_radius = 20
    bird_velocity_y = 0
    gravity = 0.3
    jump_strength = -6
    score = 0
    game_over = False
    pipe_width = 70
    pipe_gap = 200
    pipe_speed = 5
    pipes = []
    scored_pipes = set()
    last_jump_time = 0
    jump_cooldown = 0.5

    def handle_gesture(result, output_image, timestamp_ms):
        nonlocal bird_velocity_y, last_jump_time
        if result.gestures and result.gestures[0][0].category_name == 'Open_Palm':
            current_time = time.time()
            if current_time - last_jump_time > jump_cooldown:
                bird_velocity_y = jump_strength
                last_jump_time = current_time

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_FILENAME)
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=handle_gesture
    )
    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    cam = cv.VideoCapture(0)
    timestamp = 0

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        ret, frame = cam.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        recognizer.recognize_async(mp_image, timestamp)
        timestamp += 1

        bird_velocity_y += gravity
        bird_y += bird_velocity_y

        if not pipes or pipes[-1]['x'] < window_width - 200:
            pipe_height = random.randint(100, window_height - pipe_gap - 100)
            pipes.append({'x': window_width, 'top_h': pipe_height, 'bottom_h': window_height - pipe_height - pipe_gap})
        for p in pipes:
            p['x'] -= pipe_speed
        pipes = [p for p in pipes if p['x'] > -pipe_width]

        bird_rect = pygame.Rect(bird_x - bird_radius, bird_y - bird_radius, bird_radius * 2, bird_radius * 2)
        if bird_y <= bird_radius or bird_y >= window_height - bird_radius:
            game_over = True
        for p in pipes:
            top_pipe_rect = pygame.Rect(p['x'], 0, pipe_width, p['top_h'])
            bottom_pipe_rect = pygame.Rect(p['x'], window_height - p['bottom_h'], pipe_width, p['bottom_h'])
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                game_over = True
            if p['x'] + pipe_width < bird_x and p['x'] not in scored_pipes:
                score += 1
                scored_pipes.add(p['x'])

        window.fill((135, 206, 235))
        pygame.draw.circle(window, (255, 255, 0), (bird_x, int(bird_y)), bird_radius)
        for p in pipes:
            pygame.draw.rect(window, (0, 128, 0), (p['x'], 0, pipe_width, p['top_h']))
            pygame.draw.rect(window, (0, 128, 0), (p['x'], window_height - p['bottom_h'], pipe_width, p['bottom_h']))
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        window.blit(score_text, (10, 10))
        if game_over:
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(window_width // 2, window_height // 2))
            window.blit(game_over_text, text_rect)
        pygame.display.flip()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release(); cv.destroyAllWindows(); pygame.quit()


def run_ai_air_canvas():
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import time, math, re
    from PIL import Image
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from huggingface_hub import InferenceClient
    except Exception as e:
        print("Required AI packages not installed:", e)
        return

    client = InferenceClient(provider="nscale", api_key="your_api_key_here")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_caption(canvas):
        pil = Image.fromarray(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
        inputs = processor(pil, return_tensors="pt")
        outputs = caption_model.generate(**inputs, max_new_tokens=30)
        return processor.decode(outputs[0], skip_special_tokens=True)

    def beautify_caption(caption: str) -> str:
        cleaned = caption.lower()
        cleaned = re.sub(r"\b(on|against)?\s*black background\b", "", cleaned)
        cleaned = cleaned.replace("a drawing of", "")
        cleaned = cleaned.replace("white", "").replace("black", "").strip()
        if not cleaned:
            cleaned = "abstract sketch"
        enhancements = ["digital painting", "bright colors", "fantasy style", "high quality"]
        return f"{cleaned}, {', '.join(enhancements)}"

    def generate_image_cloud(prompt: str) -> Image.Image:
        print("[INFO] Sending prompt to Hugging Face Inference API...")
        image = client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")
        return image

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    def pinch(landmarks, f1, f2, th=0.06):
        p1, p2 = landmarks.landmark[f1], landmarks.landmark[f2]
        return math.hypot(p1.x - p2.x, p1.y - p2.y) < th

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = np.zeros((720, 1280, 3), dtype="uint8")
    drawing_color = (255, 255, 255)
    last_point = None
    is_pinching = False
    last_trigger_time, cooldown = 0, 3

    print("======= AI Sketch-to-Art (Cloud Inference) =======")
    print("Draw: index finger up | Generate: pinch | Clear: 'C' | Quit: 'Q'")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        status = "Draw and pinch to generate art"
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame_bgr, lm, mp.solutions.hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            ix, iy = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
            ip, im = lm.landmark[6], lm.landmark[12]
            if lm.landmark[8].y < ip.y and lm.landmark[8].y < im.y:
                if last_point:
                    cv.line(canvas, last_point, (ix, iy), drawing_color, 8)
                last_point = (ix, iy)
                cv.circle(frame_bgr, (ix, iy), 10, (0, 255, 0), -1)
                status = "Drawing..."
            else:
                last_point = None
            current_pinch = pinch(lm, 4, 8, th=0.06)
            if current_pinch and not is_pinching and (time.time() - last_trigger_time > cooldown):
                if np.sum(canvas > 0) > 500:
                    is_pinching, last_trigger_time = True, time.time()
                    print("[INFO] Captioning sketch...")
                    caption = generate_caption(canvas)
                    prompt = beautify_caption(caption)
                    ai_img = generate_image_cloud(prompt)
                    ai_img.show()
                    ai_img.save("generated_cloud.png")
                    status = "Image generated! Check file"
                else:
                    status = "Draw more first"
            if not current_pinch:
                is_pinching = False
        combined = cv.addWeighted(frame_bgr, 0.5, canvas, 1.0, 0)
        cv.rectangle(combined, (20, 20), (900, 70), (0, 0, 0), -1)
        cv.putText(combined, status, (40, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.imshow("Sketch-to-Art (Cloud)", combined)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((720, 1280, 3), dtype="uint8")
            print("[CLEARED] Canvas")
    cam.release(); cv.destroyAllWindows()


# ------------------------------
# Entrypoint / dispatcher
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='', help='Which mode to run')
    args = parser.parse_args()
    mode = args.mode.strip().lower()

    if not mode:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())

    if mode == 'session3':
        run_session3()
    elif mode == 'session4':
        run_session4()
    elif mode == 'session5':
        run_session5()
    elif mode == 'session6':
        run_session6()
    elif mode == 'session7':
        run_session7()
    elif mode == 'session8':
        run_session8()
    elif mode == 'session9':
        run_session9()
    elif mode == 'session10':
        run_session10()
    elif mode == 'ai_air_canvas':
        run_ai_air_canvas()
    else:
        print(f"Unknown mode: {mode}")


if __name__ == '__main__':
    main()



