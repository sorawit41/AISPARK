import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import time

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Game Constants
GESTURES = ["Rock", "Paper", "Scissors"]
GAME_TIME = 3  # Seconds for countdown

class RockPaperScissorsApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rock Paper Scissors AI")
        self.window.geometry("900x700")
        self.window.configure(bg="#f0f0f0")

        # Game State
        self.user_score = 0
        self.ai_score = 0
        self.game_state = "WAITING" # WAITING, COUNTDOWN, RESULT
        self.countdown_start_time = 0
        self.ai_choice = "None"
        self.user_choice = "None"
        self.result_text = "Standard Game"

        # UI Components
        self.create_widgets()
        
        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        
        # Start Loop
        self.update_frame()
        
        # Bind keys
        self.window.bind('<Return>', lambda event: self.start_game())

    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.window, bg="#333")
        header_frame.pack(fill="x", pady=10)
        
        title = tk.Label(header_frame, text="Rock Paper Scissors AI", font=("Helvetica", 28, "bold"), fg="white", bg="#333")
        title.pack(pady=10)

        # Score Board
        score_frame = tk.Frame(self.window, bg="#f0f0f0")
        score_frame.pack(pady=10)
        
        self.user_score_label = tk.Label(score_frame, text="Player: 0", font=("Helvetica", 20, "bold"), fg="#2196F3", bg="#f0f0f0")
        self.user_score_label.grid(row=0, column=0, padx=50)
        
        self.ai_score_label = tk.Label(score_frame, text="AI: 0", font=("Helvetica", 20, "bold"), fg="#E91E63", bg="#f0f0f0")
        self.ai_score_label.grid(row=0, column=1, padx=50)

        # Main Display
        self.status_label = tk.Label(self.window, text="Press 'Start Game' to play!", font=("Helvetica", 22), bg="#f0f0f0", fg="#555")
        self.status_label.pack(pady=5)
        
        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=10)

        # Control Buttons
        btn_frame = tk.Frame(self.window, bg="#f0f0f0")
        btn_frame.pack(pady=20)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Game", command=self.start_game)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        self.reset_btn = ttk.Button(btn_frame, text="Reset Score", command=self.reset_score)
        self.reset_btn.grid(row=0, column=1, padx=10)
        
        self.quit_btn = ttk.Button(btn_frame, text="Quit", command=self.on_closing)
        self.quit_btn.grid(row=0, column=2, padx=10)

    def start_game(self):
        if self.game_state == "WAITING" or self.game_state == "RESULT":
            self.game_state = "COUNTDOWN"
            self.countdown_start_time = time.time()
            self.ai_choice = "None"
            self.user_choice = "None"

    def reset_score(self):
        self.user_score = 0
        self.ai_score = 0
        self.update_score_labels()
        self.status_label.config(text="Score Reset!", fg="#555")

    def count_fingers(self, hand_landmarks):
        # Finger tips ids: 4, 8, 12, 16, 20
        # Finger pips ids: 2, 6, 10, 14, 18 (used for checking if finger is open)
        
        cnt = 0
        thresh = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2
        
        # Thumb (Check x position relative to other fingers depending on hand side, simplified here)
        # For simplicity, let's assume right hand or check relative to wrist
        # Better: check if tip is to the right/left of IP joint depending on hand
        # Simple heuristic: compare tip 4 to ip 3
        # Actually for thumb, we often check distance to index finger base or using x coordinates
        
        # Robust Finger counting logic
        fingers = []
        
        # Thumb: compare x of tip[4] and ip[3]
        # Assuming right hand for user (video flipped -> left hand on screen?)
        # Let's use a simpler heuristic for extension based on distance from wrist
        wrist = hand_landmarks.landmark[0]
        
        # 4 Fingers (Index, Middle, Ring, Pinky)
        # Check if tip y is lower than pip y (remember y grows downwards)
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # Thumb
        # Tip 4, IP 3. Check if 4 is away from palm center relative to 3
        # A simple check: offset from MCP (2)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Check if thumb tip is farther from pinky base (17) than thumb mcp
        pinky_base = hand_landmarks.landmark[17]
        
        dist_tip_pinky = ((thumb_tip.x - pinky_base.x)**2 + (thumb_tip.y - pinky_base.y)**2)**0.5
        dist_mcp_pinky = ((thumb_mcp.x - pinky_base.x)**2 + (thumb_mcp.y - pinky_base.y)**2)**0.5
        
        if dist_tip_pinky > dist_mcp_pinky:
             fingers.insert(0, 1) # Thumb open
        else:
             fingers.insert(0, 0) # Thumb closed

        return fingers

    def detect_gesture(self, hand_landmarks):
        fingers = self.count_fingers(hand_landmarks)
        count = fingers.count(1)
        
        # Rock: 0 or 1 finger (sometimes thumb is detected open)
        if count == 0:
            return "Rock"
        
        # Paper: 5 fingers
        if count == 5:
            return "Paper"
        
        # Scissors: 2 fingers (Index & Middle)
        if count == 2 and fingers[1] == 1 and fingers[2] == 1:
            return "Scissors"
            
        # Fallback / Tolerance
        if count == 1: 
             return "Rock" # Often people make rock with thumb sticking out
        
        if count == 4:
            return "Paper"
            
        return "Unknown"

    def get_winner(self, user, ai):
        if user == ai:
            return "Draw!"
        
        if (user == "Rock" and ai == "Scissors") or \
           (user == "Paper" and ai == "Rock") or \
           (user == "Scissors" and ai == "Paper"):
           
           self.user_score += 1
           return "You Win!"
           
        self.ai_score += 1
        return "AI Wins!"

    def update_score_labels(self):
        self.user_score_label.config(text=f"Player: {self.user_score}")
        self.ai_score_label.config(text=f"AI: {self.ai_score}")

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.window.after(10, self.update_frame)
            return

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks)
                
                # Display current detected gesture continuously for feedback
                cv2.putText(frame, f"Detected: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Game Logic Handling
        if self.game_state == "COUNTDOWN":
            elapsed = time.time() - self.countdown_start_time
            remaining = int(GAME_TIME - elapsed) + 1
            
            if remaining > 0:
                self.status_label.config(text=f"Get Ready... {remaining}", fg="orange")
                # Overlay on frame
                cv2.putText(frame, str(remaining), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
            else:
                self.game_state = "DETECTING"
                
        elif self.game_state == "DETECTING":
            # Just take the current frame's gesture
            if gesture in GESTURES:
                self.user_choice = gesture
                self.ai_choice = random.choice(GESTURES)
                
                result = self.get_winner(self.user_choice, self.ai_choice)
                self.update_score_labels()
                
                self.result_text = f"You: {self.user_choice} vs AI: {self.ai_choice}\n{result}"
                color = "green" if "You Win" in result else "red" if "AI Wins" in result else "blue"
                self.status_label.config(text=self.result_text, fg=color)
                
                self.game_state = "RESULT"
            else:
                self.status_label.config(text="Gesture not detected! Try again.", fg="red")
                self.game_state = "WAITING"
        
        elif self.game_state == "RESULT":
             # Keep showing the result text on status label
             # Overlay result on video
             cv2.putText(frame, f"AI chose: {self.ai_choice}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update Video Label
        img_rgb_final = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_final)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)
        
        self.window.after(10, self.update_frame)

    def on_closing(self):
        print("Closing application...")
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RockPaperScissorsApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
