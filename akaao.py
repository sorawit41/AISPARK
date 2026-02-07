import cv2
import mediapipe as mp
import numpy as np
import random
import math

class ColorMixingApp:
    def __init__(self):
        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # Application State
        self.current_mix_color = np.array([255.0, 255.0, 255.0]) # White initially (BGR)
        self.target_color = self.generate_new_target()
        
        # Hand State: { "Left": color_np, "Right": color_np } (None if no color)
        self.hand_held_colors = {"Left": None, "Right": None}
        
        # Auto-Reset State
        self.match_success_time = 0
        
        # UI Layout
        # Palette colors (BGR)
        self.palette = [
            {"color": (0, 0, 255), "name": "Red", "pos": (100, 100)},
            {"color": (0, 255, 0), "name": "Green", "pos": (200, 100)},
            {"color": (255, 0, 0), "name": "Blue", "pos": (300, 100)},
            {"color": (0, 255, 255), "name": "Yellow", "pos": (400, 100)},
            {"color": (255, 255, 255), "name": "White", "pos": (500, 100)},
            {"color": (0, 0, 0), "name": "Black", "pos": (600, 100)},
        ]
        self.palette_radius = 40
        self.mixing_area_center = (1000, 360)
        self.mixing_area_radius = 100
        
        self.target_area_rect = (50, 500, 250, 700) # x1, y1, x2, y2

    def generate_new_target(self):
        return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    def reset_game(self):
        self.target_color = self.generate_new_target()
        self.current_mix_color = np.array([255.0, 255.0, 255.0]) # Reset mix to white
        self.hand_held_colors = {"Left": None, "Right": None}
        self.match_success_time = 0

    def draw_ui(self, img):
        # Draw Palette
        for p in self.palette:
            cx, cy = p["pos"]
            color = p["color"]
            cv2.circle(img, (cx, cy), self.palette_radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), self.palette_radius, (200, 200, 200), 2) # Border

        # Draw Mixing Area
        mix_color_bgr = tuple(map(int, self.current_mix_color))
        cv2.circle(img, self.mixing_area_center, self.mixing_area_radius, mix_color_bgr, cv2.FILLED)
        cv2.circle(img, self.mixing_area_center, self.mixing_area_radius, (255, 255, 255), 3)
        cv2.putText(img, "Mix Here", (self.mixing_area_center[0]-60, self.mixing_area_center[1]-120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw Target Area
        x1, y1, x2, y2 = self.target_area_rect
        target_bgr = tuple(map(int, self.target_color))
        cv2.rectangle(img, (x1, y1), (x2, y2), target_bgr, cv2.FILLED)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(img, "Target", (x1+50, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img

    def check_pick_color(self, index_x, index_y, hand_label):
        # Check Palettes
        picked = False
        for i, p in enumerate(self.palette):
            cx, cy = p["pos"]
            dist = math.hypot(index_x - cx, index_y - cy)
            if dist < self.palette_radius:
                self.hand_held_colors[hand_label] = np.array(p["color"])
                picked = True
                return f"{hand_label} Picked {p['name']}"
        return ""

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1) # Mirror view
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            message = ""
            
            hand_tips = {} # Store tip positions: {'Left': (x, y), 'Right': (x, y)}

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    # Get handedness Label (Left/Right)
                    # Note: MediaPipe assumes mirror image by default = False? 
                    # But we flipped the image. 
                    # Usually: 'Left' hand in mirror view appears on the left side of screen?
                    # Let's rely on classification label.
                    handedness = results.multi_handedness[idx].classification[0].label
                    
                    # Draw Hands
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get Index Finger Tip (Landmark 8)
                    h, w, c = img.shape
                    index_tip = hand_lms.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    hand_tips[handedness] = (ix, iy)
                    
                    # 1. Visualization of Held Color
                    held_color = self.hand_held_colors.get(handedness)
                    if held_color is not None:
                        # Draw a halo/circle around the finger tip with the held color
                        color_bgr = tuple(map(int, held_color))
                        cv2.circle(img, (ix, iy), 25, color_bgr, cv2.FILLED)
                        cv2.circle(img, (ix, iy), 25, (255, 255, 255), 2)
                    else:
                        cv2.circle(img, (ix, iy), 15, (200, 200, 200), cv2.FILLED)

                    # 2. Check Picking
                    msg = self.check_pick_color(ix, iy, handedness)
                    if msg: 
                        message = msg

            # 3. Check Rubbing (Mixing)
            if 'Left' in hand_tips and 'Right' in hand_tips:
                lx, ly = hand_tips['Left']
                rx, ry = hand_tips['Right']
                dist_hands = math.hypot(lx - rx, ly - ry)
                
                # If hands are close (rubbing)
                if dist_hands < 60: 
                    # Mix Logic:
                    # If both hands have color, mix both.
                    # If one hand has color, mix that one.
                    
                    c1 = self.hand_held_colors['Left']
                    c2 = self.hand_held_colors['Right']
                    
                    mix_candidates = []
                    if c1 is not None: mix_candidates.append(c1)
                    if c2 is not None: mix_candidates.append(c2)
                    
                    if mix_candidates:
                        # Average of held colors
                        input_mix = np.mean(mix_candidates, axis=0)
                        
                        # Add to main mix
                        weight = 0.1
                        self.current_mix_color = self.current_mix_color * (1 - weight) + input_mix * weight
                        self.current_mix_color = np.clip(self.current_mix_color, 0, 255)
                        
                        message = "Rubbing & Mixing!"
                        
                        # Visual feedback for mixing - draw line between hands
                        cv2.line(img, (lx, ly), (rx, ry), (255, 255, 255), 5)

            # Draw UI
            img = self.draw_ui(img)
            
            # Draw Message
            if message:
                cv2.putText(img, message, (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calculate match score
            diff = np.linalg.norm(self.current_mix_color - self.target_color)
            score_text = f"Diff: {int(diff)}"
            cv2.putText(img, score_text, (50, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if diff < 30: # Threshold for success
                if self.match_success_time == 0:
                    self.match_success_time = cv2.getTickCount()
                
                cv2.putText(img, "MATCHED!", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                
                # Check 2 seconds elapsed
                current_time = cv2.getTickCount()
                elapsed = (current_time - self.match_success_time) / cv2.getTickFrequency()
                if elapsed > 2.0:
                    self.reset_game()
            else:
                self.match_success_time = 0

            cv2.imshow("Color Mixing App", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ColorMixingApp()
    app.run()
