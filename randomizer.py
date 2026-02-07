import cv2
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from PIL import Image, ImageTk, ImageDraw, ImageFont
import random
import os
from datetime import datetime

THEME_NAME = "cyborg" 
FONT_HEADER = ("Impact", 24)
FONT_WINNER = ("Verdana", 14, "bold")
try:
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        alt_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
        face_cascade = cv2.CascadeClassifier(alt_path)
        if face_cascade.empty():
             raise IOError("Cannot load cascade file.")
except Exception:
    print("Error: OpenCV Cascades not found.")
    exit()

cap = cv2.VideoCapture(0)

class ModernFaceRandomizer:
    def __init__(self, window):
        self.window = window
        self.window.title("⚡ FACE ROULETTE PRO (FREEZE MODE) ⚡")
        self.window.geometry("1400x900")
        
        self.visible_faces_data = [] 
        self.current_cv_frame = None
        self.history_log = []     
        self.history_widgets = [] 

        self.is_frozen = False  
        
        header_frame = ttk.Frame(window, bootstyle="dark")
        header_frame.pack(fill=X, pady=5, padx=20)
        
        lbl_title = ttk.Label(
            header_frame, 
            text="🎲 RANDOMIZER // FREEZE MODE", 
            font=FONT_HEADER, 
            bootstyle="info"
        )
        lbl_title.pack(side=LEFT, pady=10)

        self.lbl_status = ttk.Label(
            header_frame,
            text="DETECTING: 0",
            font=("Consolas", 14, "bold"),
            bootstyle="warning"
        )
        self.lbl_status.pack(side=RIGHT, pady=10)

        # Body
        body_frame = ttk.Frame(window)
        body_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)


        monitor_frame = ttk.Labelframe(body_frame, text=" [ LIVE FEED ] ", padding=10, bootstyle="info")
        monitor_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(monitor_frame)
        self.video_label.pack(fill=BOTH, expand=True)

        history_panel = ttk.Labelframe(body_frame, text=" [ HALL OF FAME ] ", padding=10, bootstyle="primary", width=400)
        history_panel.pack(side=RIGHT, fill=Y, padx=(10, 0))
        history_panel.pack_propagate(False)

        self.history_scroll = ScrolledFrame(history_panel, autohide=False)
        self.history_scroll.pack(fill=BOTH, expand=True)

        control_deck = ttk.Frame(window, padding=20, bootstyle="secondary")
        control_deck.pack(fill=X, side=BOTTOM)

        ttk.Button(control_deck, text="SHUTDOWN", command=self.on_closing, bootstyle="danger-outline", width=15).pack(side=LEFT)

        center_ctrl = ttk.Frame(control_deck, bootstyle="secondary")
        center_ctrl.pack(side=TOP)

        ttk.Label(center_ctrl, text="PICK QTY:", bootstyle="inverse-secondary").pack(side=LEFT, padx=5)
        self.spin_qty = ttk.Spinbox(center_ctrl, from_=1, to=5, width=3, font=("Consolas", 14))
        self.spin_qty.set(1)
        self.spin_qty.pack(side=LEFT, padx=5)

        self.btn_random = ttk.Button(
            center_ctrl, 
            text="⚡ INITIATE RANDOM (SPACEBAR) ⚡", 
            command=self.random_pick,
            bootstyle="success",
            width=40
        )
        self.btn_random.pack(side=LEFT, padx=20)

        self.window.bind_all('<space>', self.random_pick_event)
        
        self.update_frame()

    def random_pick_event(self, event):
        self.random_pick()

    def random_pick(self):
        if self.is_frozen:
            self.is_frozen = False
            self.btn_random.config(text="⚡ INITIATE RANDOM (SPACEBAR) ⚡", bootstyle="success")
            self.lbl_status.config(text="SYSTEM READY...", bootstyle="warning")
            return
        if not self.visible_faces_data:
            self.lbl_status.config(text="ERROR: NO TARGET", bootstyle="danger")
            return

        try:
            qty = int(self.spin_qty.get())
        except: 
            qty = 1
        candidates = self.visible_faces_data
        picks = random.sample(candidates, min(qty, len(candidates)))
        
        winner_names = []
        current_time = datetime.now().strftime("%H:%M:%S")
        frozen_frame = self.current_cv_frame.copy()

        for name, rect in picks:
            winner_names.append(name)

            x, y, w, h = rect
            cv2.rectangle(frozen_frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # กรอบหนาขึ้น
            cv2.putText(frozen_frame, "WINNER", (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

            cx, cy = x + w//2, y + h//2
            radius = max(w, h)//2 + 50
            h_img, w_img = self.current_cv_frame.shape[:2]
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w_img, cx + radius)
            y2 = min(h_img, cy + radius)

            face_img = self.current_cv_frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                img_rgb_crop = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img_pil_crop = Image.fromarray(img_rgb_crop)
                
                self.history_log.insert(0, {
                    "name": name,
                    "img": img_pil_crop,
                    "time": current_time
                })

 
        overlay = frozen_frame.copy()
        cv2.rectangle(overlay, (0, frozen_frame.shape[0]//2 - 80), (frozen_frame.shape[1], frozen_frame.shape[0]//2 + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frozen_frame, 0.4, 0, frozen_frame)

        text_str = f"WINNER: {', '.join(winner_names)}"
        text_size = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
        tx = (frozen_frame.shape[1] - text_size[0]) // 2
        ty = (frozen_frame.shape[0] + text_size[1]) // 2
        cv2.putText(frozen_frame, text_str, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        img_rgb_freeze = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB)
        img_pil_freeze = Image.fromarray(img_rgb_freeze)
        img_tk_freeze = ImageTk.PhotoImage(image=img_pil_freeze)
        
        self.video_label.imgtk = img_tk_freeze
        self.video_label.configure(image=img_tk_freeze)

        if len(self.history_log) > 50:
            self.history_log = self.history_log[:50]
        self.render_history()

        self.is_frozen = True
        self.btn_random.config(text="▶ PRESS SPACE TO CONTINUE", bootstyle="info")
        self.lbl_status.config(text="WINNER FOUND!", bootstyle="success")

    def render_history(self):
        for w in self.history_widgets: w.destroy()
        self.history_widgets.clear()

        for item in self.history_log:
            card = ttk.Frame(self.history_scroll, bootstyle="secondary", padding=5)
            card.pack(fill=X, pady=5, padx=5)
            self.history_widgets.append(card)

            try:
                img_rez = item['img'].resize((100, 100), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_rez)
                lbl_img = ttk.Label(card, image=img_tk, bootstyle="inverse-secondary")
                lbl_img.image = img_tk
                lbl_img.pack(side=LEFT)
            except: pass

            info = ttk.Frame(card, bootstyle="secondary")
            info.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
            ttk.Label(info, text=item['name'], font=FONT_WINNER, bootstyle="inverse-secondary").pack(anchor="w")
            ttk.Label(info, text=f"TIME: {item['time']}", font=("Consolas", 10), bootstyle="info").pack(anchor="w")

    def update_frame(self):
        if self.is_frozen:
            self.window.after(10, self.update_frame)
            return

        success, frame = cap.read()
        if not success:
            self.window.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        self.current_cv_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(30, 30))
        
        self.visible_faces_data = []
        
        COLOR_BOX = (255, 0, 255) 

        for i, (x, y, w, h) in enumerate(faces):
            name = f"ID-{i+1:03d}"
            self.visible_faces_data.append((name, (x, y, w, h)))

            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_BOX, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), COLOR_BOX, -1)
            cv2.putText(frame, name, (x+5, y-8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        self.lbl_status.config(text=f"DETECTING: {len(faces)}")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_frame)

    def on_closing(self):
        cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="cyborg") 
    app = ModernFaceRandomizer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
