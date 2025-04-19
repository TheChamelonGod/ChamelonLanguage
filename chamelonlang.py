import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                             QWidget, QVBoxLayout, QPushButton, QLabel,
                             QTextEdit, QHBoxLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

# --- Configuration ---
MODEL_PATH = 'dist/model.p'
CAMERA_INDEX = 0
PREDICTION_CONFIDENCE_THRESHOLD = 0.4
STABLE_FRAMES_THRESHOLD = 10
GEMINI_API_KEY = "AIzaSyAFJOE3QLJcRXUqjqyeqmRRFdXrpwer4mg"  # Replace with your actual API key

# --- Sign Language Labels ---
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z'
}

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')


# ======================
# Main Application Class
# ======================
class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Learning")
        self.setGeometry(100, 100, 1200, 800)

        # Load the sign recognition model
        self.sign_model = self.load_sign_model()
        self.hands = self.initialize_hands()

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create screens
        self.home_screen = HomeScreen(self)
        self.learn_screen = LearnScreen(self)
        self.train_screen = TrainScreen(self)

        # Add screens to stacked widget
        self.stacked_widget.addWidget(self.home_screen)
        self.stacked_widget.addWidget(self.learn_screen)
        self.stacked_widget.addWidget(self.train_screen)

        # Start with home screen
        self.show_home()

    def load_sign_model(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_dict = pickle.load(f)
            return model_dict['model']
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def initialize_hands(self):
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.4
        )

    def show_home(self):
        self.stacked_widget.setCurrentWidget(self.home_screen)

    def show_learn(self):
        self.learn_screen.clear_chat()
        self.stacked_widget.setCurrentWidget(self.learn_screen)

    def show_train(self):
        self.train_screen.start_training()
        self.stacked_widget.setCurrentWidget(self.train_screen)


# =================
# Home Screen Class
# =================
class HomeScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Sign Language Learning")
        title.setStyleSheet("font-size: 48px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        # Description
        desc = QLabel("Learn and practice sign language with interactive tools")
        desc.setStyleSheet("font-size: 24px;")
        desc.setAlignment(Qt.AlignCenter)

        # Buttons
        learn_btn = QPushButton("Learn")
        learn_btn.setStyleSheet("font-size: 24px; padding: 20px;")
        learn_btn.clicked.connect(self.parent.show_learn)

        train_btn = QPushButton("Train")
        train_btn.setStyleSheet("font-size: 24px; padding: 20px;")
        train_btn.clicked.connect(self.parent.show_train)

        # Add widgets to layout
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch(1)
        layout.addWidget(learn_btn)
        layout.addWidget(train_btn)
        layout.addStretch(1)

        self.setLayout(layout)


# ==================
# Learn Screen Class
# ==================
class LearnScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.chat_history = []
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.hard_mode = False  # New flag for hard mode
        self.current_word = None
        self.time_limit = 10  # Time limit for hard mode (in seconds)
        self.start_time = None

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left side - Chat area
        chat_layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 16px;")

        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(100)
        self.user_input.setStyleSheet("font-size: 16px;")

        send_btn = QPushButton("Send")
        send_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        send_btn.clicked.connect(self.send_message)

        # Button row
        btn_layout = QHBoxLayout()
        quiz_btn = QPushButton("Generate Quiz")
        quiz_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        quiz_btn.clicked.connect(lambda: self.send_message("Generate a short quiz about sign language"))

        flashcards_btn = QPushButton("Generate Flashcards")
        flashcards_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        flashcards_btn.clicked.connect(
            lambda: self.send_message("Generate flashcards for learning sign language alphabet"))

        hard_mode_btn = QPushButton("Toggle Hard Mode")
        hard_mode_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        hard_mode_btn.clicked.connect(self.toggle_hard_mode)

        btn_layout.addWidget(quiz_btn)
        btn_layout.addWidget(flashcards_btn)
        btn_layout.addWidget(hard_mode_btn)

        # Add widgets to chat layout
        chat_layout.addWidget(QLabel("Sign Language Learning Assistant"))
        chat_layout.addWidget(self.chat_display)
        chat_layout.addLayout(btn_layout)
        chat_layout.addWidget(QLabel("Your Message:"))
        chat_layout.addWidget(self.user_input)
        chat_layout.addWidget(send_btn)

        # Right side - Camera preview
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)

        back_btn = QPushButton("Back to Home")
        back_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        back_btn.clicked.connect(self.parent.show_home)

        camera_layout.addWidget(QLabel("Camera Preview"))
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(back_btn)

        # Add layouts to main layout
        main_layout.addLayout(chat_layout, 2)
        main_layout.addLayout(camera_layout, 1)

        self.setLayout(main_layout)

        # Start camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            frame.flags.writeable = False
            results = self.parent.hands.process(frame)
            frame.flags.writeable = True

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Convert to QImage and display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img))

            # Handle Hard Mode Timer
            if self.hard_mode and self.start_time:
                elapsed_time = time.time() - self.start_time
                remaining_time = self.time_limit - elapsed_time
                if remaining_time <= 0:
                    self.instructions.setText(f"Time's up! You failed to sign the word.")
                    QTimer.singleShot(2000, self.next_word)
                else:
                    self.instructions.setText(f"Hard Mode: Time remaining: {int(remaining_time)}s")

    def send_message(self, message=None):
        if message is None:
            message = self.user_input.toPlainText().strip()
            if not message:
                return
            self.user_input.clear()

        # Add user message to chat
        self.add_to_chat("You", message)

        # Get response from Gemini
        try:
            response = self.gemini_model.generate_content(message)
            self.add_to_chat("Assistant", response.text)
        except Exception as e:
            self.add_to_chat("Assistant", f"Error: {str(e)}")

    def add_to_chat(self, sender, message):
        self.chat_history.append((sender, message))
        formatted_chat = ""
        for sender, msg in self.chat_history:
            formatted_chat += f"<b>{sender}:</b> {msg}<br><br>"
        self.chat_display.setHtml(formatted_chat)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum())

    def clear_chat(self):
        self.chat_history = []
        self.chat_display.clear()

    def toggle_hard_mode(self):
        # Toggle the hard mode and start the timer for hard mode
        self.hard_mode = not self.hard_mode
        if self.hard_mode:
            self.instructions.setText(f"Hard Mode Activated! Sign the word within {self.time_limit} seconds!")
            self.start_time = time.time()  # Start the timer
        else:
            self.instructions.setText("Hard Mode Deactivated. Take your time!")
            self.start_time = None  # Stop the timer

    def next_word(self):
        # Proceed to the next word after the time runs out or if the word is signed correctly
        self.current_word = self.get_random_word()
        self.instructions.setText(f"Sign the word: {self.current_word}")
        self.start_time = time.time()  # Restart the timer for the new word if in hard mode

    def get_random_word(self):
        # Here we just pick a random letter from the alphabet for the demonstration
        # You can expand this to use words instead of single letters
        words = list(labels_dict.values())
        return np.random.choice(words)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# ==================
# Train Screen Class
# ==================
class TrainScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_target = None
        self.stability_tracker = {
            'Left': {'last_prediction': None, 'stable_count': 0},
            'Right': {'last_prediction': None, 'stable_count': 0}
        }
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Top section - Instructions and target
        top_layout = QHBoxLayout()

        self.instructions = QLabel("Press Start to begin training")
        self.instructions.setStyleSheet("font-size: 24px;")
        self.instructions.setAlignment(Qt.AlignCenter)

        self.target_label = QLabel("")
        self.target_label.setStyleSheet("font-size: 72px; font-weight: bold;")
        self.target_label.setAlignment(Qt.AlignCenter)

        top_layout.addWidget(self.instructions)
        top_layout.addWidget(self.target_label)

        # Middle section - Camera feed
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)

        # Bottom section - Controls
        bottom_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("font-size: 24px; padding: 15px;")
        self.start_btn.clicked.connect(self.start_training)

        self.back_btn = QPushButton("Back to Home")
        self.back_btn.setStyleSheet("font-size: 24px; padding: 15px;")
        self.back_btn.clicked.connect(self.parent.show_home)

        bottom_layout.addWidget(self.start_btn)
        bottom_layout.addWidget(self.back_btn)

        # Add all to main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.camera_label)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        # Camera setup
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def start_training(self):
        self.current_target = self.get_random_letter()
        self.target_label.setText(self.current_target)
        self.instructions.setText(f"Sign the letter: {self.current_target}")
        self.start_btn.setEnabled(False)

    def get_random_letter(self):
        letters = list(labels_dict.values())
        return np.random.choice(letters)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.parent.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                data_aux = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                min_x, max_x = min(x_), max(x_)
                min_y, max_y = min(y_), max(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                if len(data_aux) == 42 and self.parent.sign_model is not None:
                    try:
                        prediction_num = int(self.parent.sign_model.predict([data_aux])[0])
                        predicted_letter = labels_dict.get(prediction_num)

                        if predicted_letter is not None:
                            # Handedness correction
                            detected_label = handedness_info.classification[0].label
                            physical_hand = "Right" if detected_label == "Left" else "Left"

                            # Update stability tracker
                            tracker = self.stability_tracker[physical_hand]

                            if predicted_letter == tracker['last_prediction']:
                                tracker['stable_count'] += 1
                            else:
                                tracker['last_prediction'] = predicted_letter
                                tracker['stable_count'] = 1

                            # Check if prediction is stable and correct
                            if (self.current_target and
                                    tracker['stable_count'] >= STABLE_FRAMES_THRESHOLD and
                                    predicted_letter == self.current_target):
                                # Correct sign detected
                                self.instructions.setText("Correct! Generating next letter...")
                                QTimer.singleShot(2000, self.next_letter)
                                break

                            # Draw landmarks
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                            # Draw prediction
                            x1 = max(0, int(min_x * W) - 20)
                            y1 = max(0, int(min_y * H) - 20)
                            color = (0, 255, 0) if physical_hand == "Right" else (255, 0, 0)
                            cv2.putText(frame, f"{physical_hand}: {predicted_letter}",
                                        (x1, y1 - 10 if y1 > 20 else y1 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Prediction error: {e}")

        # Convert to QImage and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))

    def next_letter(self):
        self.current_target = self.get_random_letter()
        self.target_label.setText(self.current_target)
        self.instructions.setText(f"Sign the letter: {self.current_target}")

        # Reset stability trackers
        self.stability_tracker['Left'] = {'last_prediction': None, 'stable_count': 0}
        self.stability_tracker['Right'] = {'last_prediction': None, 'stable_count': 0}

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# =============
# Main Function
# =============
def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
        QLabel {
            font-size: 16px;
        }
    """)

    # Create the application window
    window = SignLanguageApp()
    window.show()

    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
