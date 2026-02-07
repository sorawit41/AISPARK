import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import json
import os
import pyttsx3
import threading
import time
import itertools

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError:
    import tkinter.ttk as ttk
    print("ttkbootstrap not found, using standard ttk")

# --- Configuration ---
DATA_FILE = "gestures.json"
MODEL_PATH = "hand_landmarker.task"
MATCH_THRESHOLD = 0.6  # Slightly looser for 2-hand matching

class SignLanguageConverter:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language to Text Converter")
        self.window.geometry("1200x900")
        
        # Initialize State
        self.mode = "TRANSLATE" 
        self.history = []
        self.latest_result = None
        self.lock = threading.Lock()
        
        # Initialize MediaPipe Tasks
        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=2, # Enable 2 hands
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self.print_result
            )
            self.landmarker = HandLandmarker.create_from_options(options)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize MediaPipe: {e}\nEnsure {MODEL_PATH} is in the folder.")
            self.window.destroy()
            return
            
        self.mp_draw = None 
        
        # Initialize TTS
        try:
            self.engine = pyttsx3.init()
        except:
            self.engine = None
            
        self.last_spoken_time = 0
        self.last_spoken_text = ""
        
        # Data storage
        self.gestures = self.load_gestures()
        self.current_gesture_name = "---"
        
        # UI Setup
        self.setup_ui()
        
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        
        self.bind_shortcuts()
        self.update_loop()

    def bind_shortcuts(self):
        self.window.bind('<space>', lambda event: self.toggle_mode())
        self.window.bind('s', lambda event: self.save_gesture())
        self.window.bind('d', lambda event: self.delete_gesture())
        self.window.bind('p', lambda event: self.speak_current())
        self.window.bind('<Return>', lambda event: self.speak_current())

    def print_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.lock:
            self.latest_result = result

    def setup_ui(self):
        # Header
        header_frame = ttk.Frame(self.window, padding=10)
        header_frame.pack(fill=tk.X, pady=5)
        
        title_label = ttk.Label(header_frame, text="Sign Language Converter (2 Hands)", font=("Impact", 28), bootstyle="primary")
        title_label.pack(side=tk.LEFT, padx=10)
        
        self.mode_label = ttk.Label(header_frame, text=f"MODE: {self.mode}", font=("Verdana", 20, "bold"), bootstyle="info")
        self.mode_label.pack(side=tk.RIGHT, padx=10)

        # Main Layout
        content_frame = ttk.Frame(self.window)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left: Video
        video_frame = ttk.Labelframe(content_frame, text=" Live Feed ", bootstyle="secondary", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right: Controls
        control_frame = ttk.Frame(content_frame, width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Output Box
        output_card = ttk.Labelframe(control_frame, text=" Recognized Text ", bootstyle="success", padding=10)
        output_card.pack(fill=tk.X, pady=(0, 20))
        
        self.gesture_label = ttk.Label(output_card, text="---", font=("Segoe UI", 48, "bold"), foreground="#2ecc71", anchor="center")
        self.gesture_label.pack(pady=10, fill=tk.X)
        
        # Actions
        action_card = ttk.Labelframe(control_frame, text=" Actions ", bootstyle="info", padding=10)
        action_card.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(action_card, text="SWITCH MODE (Spacebar)", command=self.toggle_mode, bootstyle="warning-outline", width=25).pack(pady=5, fill=tk.X)
        self.save_btn = ttk.Button(action_card, text="SAVE NEW GESTURE (S)", command=self.save_gesture, state=tk.DISABLED, bootstyle="success", width=25)
        self.save_btn.pack(pady=5, fill=tk.X)
        ttk.Button(action_card, text="SPEAK AGAIN (P)", command=self.speak_current, bootstyle="primary-outline", width=25).pack(pady=5, fill=tk.X)
        
        # List
        ttk.Label(control_frame, text="Saved Gestures:", font=("Verdana", 10)).pack(anchor="w")
        self.listbox = tk.Listbox(control_frame, height=10, font=("Consolas", 12))
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.refresh_listbox()
        
        ttk.Button(control_frame, text="DELETE SELECTED (D)", command=self.delete_gesture, bootstyle="danger", width=25).pack(pady=5, fill=tk.X)

    def load_gestures(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_gestures_to_file(self):
        with open(DATA_FILE, 'w') as f:
            json.dump(self.gestures, f, indent=4)
        self.refresh_listbox()

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for name in sorted(self.gestures.keys()):
            self.listbox.insert(tk.END, name)

    def toggle_mode(self):
        if self.mode == "TRANSLATE":
            self.mode = "LEARN"
            self.save_btn.config(state=tk.NORMAL)
            self.mode_label.config(text="MODE: LEARN", bootstyle="danger")
        else:
            self.mode = "TRANSLATE"
            self.save_btn.config(state=tk.DISABLED)
            self.mode_label.config(text="MODE: TRANSLATE", bootstyle="info")

    def get_landmarks_list(self, hand_landmarks):
        points = []
        for lm in hand_landmarks:
            points.append([lm.x, lm.y, lm.z])
        
        wrist = points[0]
        relative_points = []
        for p in points:
            relative_points.extend([p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]])
            
        return relative_points

    def save_gesture(self):
        with self.lock:
             result = self.latest_result
        
        if not result or not result.hand_landmarks:
            messagebox.showwarning("Warning", "No hand detected to save!")
            return
            
        # Capture ALL hands
        current_hands = []
        for hand_lms in result.hand_landmarks:
            current_hands.append(self.get_landmarks_list(hand_lms))
            
        name = simpledialog.askstring("Input", "Enter name for this gesture:")
        if name:
            self.gestures[name] = current_hands # Save list of hands
            self.save_gestures_to_file()
            messagebox.showinfo("Success", f"Saved gesture: {name} ({len(current_hands)} hands)")

    def delete_gesture(self):
        selection = self.listbox.curselection()
        if selection:
            name = self.listbox.get(selection[0])
            del self.gestures[name]
            self.save_gestures_to_file()

    def calculate_distance(self, hand1, hand2):
        return np.linalg.norm(np.array(hand1) - np.array(hand2))

    def match_gesture(self, detected_hands):
        # detected_hands = list of (21*3) points
        if not detected_hands: return "---"
        
        best_match = "---"
        min_dist = float('inf')
        
        for name, saved_data in self.gestures.items():
            # Handle legacy format (single list) vs new format (list of lists)
            if isinstance(saved_data[0], float): # Legacy single hand
                saved_hands = [saved_data]
            else:
                saved_hands = saved_data
                
            # Logic: Match sets of hands
            # If counts differ significantly, penalty?
            # Start simple: Find best matching permutation
            
            # If detecting 1 hand but saved 2: impossible to be perfect match, but maybe partial?
            # Let's enforce strict count for now OR min distance across best pairs.
            
            # Simple greedy match: For each detected hand, find closest saved hand.
            # Total distance = Sum of best matches / num_hands
            
            # We must average distance per hand to keep threshold consistent
            
            total_dist = 0
            count = 0
            
            # Case 1: Same number of hands
            if len(detected_hands) == len(saved_hands):
                # Try all permutations for minimal distance (handles Left/Right swap effectively)
                import itertools
                perms = list(itertools.permutations(detected_hands))
                
                best_perm_dist = float('inf')
                for p in perms:
                    d = 0
                    for h1, h2 in zip(p, saved_hands):
                        d += self.calculate_distance(h1, h2)
                    if d < best_perm_dist:
                        best_perm_dist = d
                
                total_dist = best_perm_dist / len(saved_hands)
            
            else:
                # Count mismatch. Give huge penalty.
                total_dist = float('inf') 

            if total_dist < min_dist:
                min_dist = total_dist
                best_match = name
        
        if min_dist < MATCH_THRESHOLD: 
            return best_match
        else:
            return "?"

    def speak_text(self, text):
        if not self.engine: return
        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
        threading.Thread(target=_speak).start()

    def speak_current(self):
        text = self.gesture_label.cget("text")
        if text != "---" and text != "?":
            self.speak_text(text)
            
    def draw_landmarks_manual(self, image, landmarks):
        h, w, _ = image.shape
        CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4), 
            (0,5), (5,6), (6,7), (7,8), 
            (9,10), (10,11), (11,12),   
            (13,14), (14,15), (15,16),  
            (0,17), (17,18), (18,19), (19,20), 
            (5,9), (9,13), (13,17) 
        ]
        
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            
        for s, e in CONNECTIONS:
            if s < len(points) and e < len(points):
                cv2.line(image, points[s], points[e], (0, 255, 0), 2)

    def update_loop(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - self.start_time) * 1000)
            
            try:
                self.landmarker.detect_async(mp_image, timestamp_ms)
            except Exception as e:
                print(e)
            
            current_text = "---"
            with self.lock:
                result = self.latest_result
                
            detected_hands_list = []
            
            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    self.draw_landmarks_manual(frame, hand_landmarks)
                    detected_hands_list.append(self.get_landmarks_list(hand_landmarks))
                
                if self.mode == "TRANSLATE" and self.gestures:
                    match = self.match_gesture(detected_hands_list)
                    current_text = match
                    
                    self.history.append(match)
                    if len(self.history) > 10:
                        self.history.pop(0)
                        
                    if self.history.count(match) > 8 and match != "---" and match != "?":
                        if match != self.last_spoken_text or (time.time() - self.last_spoken_time > 3):
                            self.speak_text(match)
                            self.last_spoken_text = match
                            self.last_spoken_time = time.time()
                            
                elif self.mode == "LEARN":
                    current_text = f"Ready to Save ({len(detected_hands_list)} Hands)"

            self.gesture_label.config(text=current_text)
            
            # Resize for UI
            DISPLAY_WIDTH = 640
            DISPLAY_HEIGHT = 480
            frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_loop)

    def on_closing(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    try:
        root = ttk.Window(themename="darkly")
    except:
        root = tk.Tk()
        
    app = SignLanguageConverter(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
