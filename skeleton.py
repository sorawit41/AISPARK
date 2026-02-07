import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

class PoseTrackingApp:
    
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def __init__(self, window):
        self.window = window
        self.window.title("Pose Tracking + Action + Counter")

        self.status_label = ttk.Label(window, text="Status: Initializing...", font=("Helvetica", 14))
        self.status_label.pack(pady=5)

        self.action_label = ttk.Label(window, text="Action: ---", font=("Helvetica", 24, "bold"), foreground="blue")
        self.action_label.pack(pady=10)
        
        self.counter_label = ttk.Label(window, text="L-Curls: 0 | R-Curls: 0", font=("Helvetica", 18), foreground="green")
        self.counter_label.pack(pady=5)
        
        self.video_label = ttk.Label(window)
        self.video_label.pack()

        self.quit_button = ttk.Button(window, text="Quit", command=self.on_closing)
        self.quit_button.pack(pady=5)

        self.mp_pose_enum = mp.solutions.pose.PoseLandmark 
        self.mp_hand_enum = mp.solutions.holistic.HandLandmark

        self.left_curl_counter = 0
        self.left_curl_stage = 'DOWN'
        self.right_curl_counter = 0
        self.right_curl_stage = 'DOWN'

        self.update_frame()

    def detect_action(self, pose_landmarks):
        action = "---" 

        try:
            left_shoulder_y = pose_landmarks[self.mp_pose_enum.LEFT_SHOULDER.value].y
            right_shoulder_y = pose_landmarks[self.mp_pose_enum.RIGHT_SHOULDER.value].y
            left_shoulder_x = pose_landmarks[self.mp_pose_enum.LEFT_SHOULDER.value].x
            right_shoulder_x = pose_landmarks[self.mp_pose_enum.RIGHT_SHOULDER.value].x
            left_elbow_x = pose_landmarks[self.mp_pose_enum.LEFT_ELBOW.value].x
            right_elbow_x = pose_landmarks[self.mp_pose_enum.RIGHT_ELBOW.value].x
            left_wrist_y = pose_landmarks[self.mp_pose_enum.LEFT_WRIST.value].y
            right_wrist_y = pose_landmarks[self.mp_pose_enum.RIGHT_WRIST.value].y
            left_wrist_x = pose_landmarks[self.mp_pose_enum.LEFT_WRIST.value].x
            right_wrist_x = pose_landmarks[self.mp_pose_enum.RIGHT_WRIST.value].x
            left_hip_y = pose_landmarks[self.mp_pose_enum.LEFT_HIP.value].y
            right_hip_y = pose_landmarks[self.mp_pose_enum.RIGHT_HIP.value].y
            left_knee_y = pose_landmarks[self.mp_pose_enum.LEFT_KNEE.value].y
            right_knee_y = pose_landmarks[self.mp_pose_enum.RIGHT_KNEE.value].y

            if (left_wrist_y < left_shoulder_y) and (right_wrist_y < right_shoulder_y):
                action = "Hands Up!"
            
            elif (left_hip_y > left_knee_y) and (right_hip_y > right_knee_y):
                action = "Squatting!"
            
            hands_down = (left_wrist_y > left_shoulder_y) and (right_wrist_y > right_shoulder_y)
            arms_out_straight = (left_wrist_x < left_elbow_x < left_shoulder_x) and \
                                (right_wrist_x > right_elbow_x > right_shoulder_x)
            
            if hands_down and arms_out_straight:
                action = "T-Pose (Arms Out)!"
                
            return action

        except Exception as e:
            return "Partial View"

    def update_curl_counters(self, pose_landmarks):
        try:
            left_shoulder = pose_landmarks[self.mp_pose_enum.LEFT_SHOULDER.value]
            left_elbow = pose_landmarks[self.mp_pose_enum.LEFT_ELBOW.value]
            left_wrist = pose_landmarks[self.mp_pose_enum.LEFT_WRIST.value]
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if left_angle > 160:
                self.left_curl_stage = "DOWN"
            if left_angle < 30 and self.left_curl_stage == 'DOWN':
                self.left_curl_stage = "UP"
                self.left_curl_counter += 1

            right_shoulder = pose_landmarks[self.mp_pose_enum.RIGHT_SHOULDER.value]
            right_elbow = pose_landmarks[self.mp_pose_enum.RIGHT_ELBOW.value]
            right_wrist = pose_landmarks[self.mp_pose_enum.RIGHT_WRIST.value]
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if right_angle > 160:
                self.right_curl_stage = "DOWN"
            if right_angle < 30 and self.right_curl_stage == 'DOWN':
                self.right_curl_stage = "UP"
                self.right_curl_counter += 1

        except Exception as e:
            pass

    def update_frame(self):
        status = "Not Detected"
        action = "---"
        color = "blue"

        success, frame = cap.read()
        if not success:
            self.window.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(img_rgb)

        if results.pose_landmarks:
            status = "Tracking"
            
            mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )

            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            landmarks = results.pose_landmarks.landmark
            action = self.detect_action(landmarks)
            self.update_curl_counters(landmarks)
        
        if action == "Hands Up!":
            color = "blue"
        elif action == "Squatting!":
            color = "red"
        elif action == "T-Pose (Arms Out)!":
            color = "green"
        else:
            color = "black"

        self.status_label.config(text=f"Status: {status}")
        self.action_label.config(text=f"Action: {action}", foreground=color)
        
        self.counter_label.config(text=f"L-Curls: {self.left_curl_counter} | R-Curls: {self.right_curl_counter}")

        img_rgb_for_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_for_pil)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_frame)

    def on_closing(self):
        print("Closing application...")
        cap.release()
        holistic.close()
        self.window.destroy()

# --- ส่วนรันโปรแกรม ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()