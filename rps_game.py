"""
Rock–Paper–Scissors with MediaPipe hand gestures and Tkinter GUI.
"""

from __future__ import annotations

import random
import urllib.request
import math
import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from mediapipe.tasks.python.vision.core import image as mp_image
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import messagebox
class Move(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"

MOVE_EMOJI = {Move.ROCK: "✊", Move.PAPER: "✋", Move.SCISSORS: "✌️"}

COL_SURFACE = "#FFFFFF"
COL_TITLE = "#1A1B2E"
COL_SCORE_YOU = "#246BFD"
COL_SCORE_CPU = "#EB334D"
COL_CARD_YOU = "#1BC069"
COL_CARD_CPU = "#526DFF"
COL_RESULT_DRAW = "#F09117"
COL_RESULT_WIN = "#1BC069"
COL_RESULT_LOSE = "#EB334D"
COL_BTN_PLAY = "#1BC069"
COL_BTN_RESET = "#68768A"
COL_MUTED = "#9395A6"

VIDEO_DISP_W = 280
VIDEO_DISP_H = 220
CARD_ICON_BOX = 168
CARD_ICON_FILL = 0.78

_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_hand_landmarker_model() -> Path:
    path = Path(__file__).resolve().parent / "hand_landmarker.task"
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.download")
    req = urllib.request.Request(_HAND_MODEL_URL, headers={"User-Agent": "rps_mediapipe/1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(path)
    return path

def classify_gesture_tasks(landmarks: list) -> Optional[Move]:
    """Classify RPS using finger extension that works for upright and sideways hands."""
    lm = landmarks

    def dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # Wrist → middle MCP: stable palm size independent of hand rotation
    palm = dist(lm[0], lm[9])
    if palm < 1e-6:
        return None

    def finger_extended(tip: int, pip: int, mcp: int) -> bool:
        # Tip farther from wrist than PIP → finger is open (works when hand is sideways)
        tip_wrist = dist(lm[tip], lm[0])
        pip_wrist = dist(lm[pip], lm[0])
        tip_mcp = dist(lm[tip], lm[mcp])
        return tip_wrist > pip_wrist * 1.02 and tip_mcp > palm * 0.32

    index = finger_extended(8, 6, 5)
    middle = finger_extended(12, 10, 9)
    ring = finger_extended(16, 14, 13)
    pinky = finger_extended(20, 18, 17)
    fingers_up = sum([index, middle, ring, pinky])

    if index and middle and not ring and not pinky:
        return Move.SCISSORS
    if fingers_up >= 4:
        return Move.PAPER
    if fingers_up <= 1:
        return Move.ROCK
    if fingers_up == 3:
        return Move.PAPER
    return None

def winner(player: Move, computer: Move) -> int:
    if player == computer: return 0
    beats = {Move.ROCK: Move.SCISSORS, Move.PAPER: Move.ROCK, Move.SCISSORS: Move.PAPER}
    return 1 if beats[player] == computer else -1

def create_rounded_rect_image(width, height, radius, color, outline_color=None, outline_width=0) -> ImageTk.PhotoImage:
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=color, outline=outline_color, width=outline_width)
    return ImageTk.PhotoImage(img)

def _emoji_font(size: int) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for name in ("seguiemj.ttf", "SegoeUIEmoji.ttf", "seguisym.ttf"):
        path = windir / "Fonts" / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()

def create_centered_emoji_image(
    emoji: str,
    box: int = CARD_ICON_BOX,
    fill_ratio: float = CARD_ICON_FILL,
) -> ImageTk.PhotoImage:
    """Render emoji into a fixed square so every icon shares size and true center."""
    render = 256
    pad = render
    canvas = Image.new("RGBA", (render + pad * 2, render + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _emoji_font(render)
    cx = cy = (render + pad * 2) // 2
    try:
        draw.text((cx, cy), emoji, font=font, embedded_color=True, anchor="mm")
    except TypeError:
        draw.text((cx, cy), emoji, font=font, anchor="mm")
    bbox = canvas.getbbox()
    out = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    if bbox:
        cropped = canvas.crop(bbox)
        max_side = max(1, int(box * fill_ratio))
        w, h = cropped.size
        scale = min(max_side / w, max_side / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        cropped = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
        out.paste(cropped, ((box - nw) // 2, (box - nh) // 2), cropped)
    return ImageTk.PhotoImage(out)

class RPSApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors")
        self.root.minsize(1024, 760)
        self.root.geometry("1024x780")
        self.root.configure(bg=COL_SURFACE)
        self.cap = None
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_landmarker_model())),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
        )
        self._landmarker = vision.HandLandmarker.create_from_options(hand_opts)
        self._video_ts = 0
        self.state, self.countdown_val, self._job = "idle", 0, None
        self._gestures, self._capture = [], False
        self.score_you = self.score_cpu = 0
        self.particles = []
        self._assets = {}
        self._build_ui()
        self._build_confetti_layer()
        self._open_camera()
        self.root.after(33, self._loop)

    def _build_confetti_layer(self):
        self.conf_win = tk.Toplevel(self.root)
        self.conf_win.attributes("-transparentcolor", "#000001")
        self.conf_win.overrideredirect(True)
        self.conf_win.attributes("-topmost", True)
        
        # Attempt to make window click-through
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(self.conf_win.winfo_id())
                style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
                ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00000020 | 0x00080000)
            except Exception: pass
            
        self.conf_cvs = tk.Canvas(self.conf_win, bg="#000001", highlightthickness=0)
        self.conf_cvs.pack(fill=tk.BOTH, expand=True)

        def _sync(e=None):
            if self.state == "result" and self.particles:
                self.conf_win.geometry(f"{self.root.winfo_width()}x{self.root.winfo_height()}+{self.root.winfo_rootx()}+{self.root.winfo_rooty()}")
                
        self.root.bind("<Configure>", _sync)
        self.conf_win.withdraw()

    def _build_ui(self):
        center = tk.Frame(self.root, bg=COL_SURFACE)
        center.pack(fill=tk.BOTH, expand=True)

        tk.Label(center, text="Rock Paper Scissors", font=("Segoe UI", -47, "bold"), fg=COL_TITLE, bg=COL_SURFACE).pack(side=tk.TOP, pady=(15, 10))

        score_row = tk.Frame(center, bg=COL_SURFACE)
        score_row.pack(side=tk.TOP, pady=(0, 5))
        y_col = tk.Frame(score_row, bg=COL_SURFACE); y_col.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_sc_y = tk.Label(y_col, text="0", font=("Segoe UI", -74, "bold"), fg=COL_SCORE_YOU, bg=COL_SURFACE); self.lbl_sc_y.pack()
        tk.Label(y_col, text="You", font=("Segoe UI", -17), fg=COL_MUTED, bg=COL_SURFACE).pack()
        dash_col = tk.Frame(score_row, bg=COL_SURFACE); dash_col.pack(side=tk.LEFT, padx=10)
        tk.Label(dash_col, text="-", font=("Segoe UI", -74, "bold"), fg="#D1D5DB", bg=COL_SURFACE).pack(pady=(0,20))
        c_col = tk.Frame(score_row, bg=COL_SURFACE); c_col.pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_sc_c = tk.Label(c_col, text="0", font=("Segoe UI", -74, "bold"), fg=COL_SCORE_CPU, bg=COL_SURFACE); self.lbl_sc_c.pack()
        tk.Label(c_col, text="Computer", font=("Segoe UI", -17), fg=COL_MUTED, bg=COL_SURFACE).pack()

        cards = tk.Frame(center, bg=COL_SURFACE); cards.pack(side=tk.TOP, pady=(0, 10))
        w_c, h_c = 320, 270
        self._assets["bg_p"] = create_rounded_rect_image(w_c, h_c, 24, COL_CARD_YOU)
        self._assets["bg_c"] = create_rounded_rect_image(w_c, h_c, 24, COL_CARD_CPU)
        
        self.cvs_p = tk.Canvas(cards, width=w_c, height=h_c, bg=COL_SURFACE, highlightthickness=0)
        self.cvs_p.pack(side=tk.LEFT, padx=(0, 15))
        self.cvs_p.create_image(w_c//2, h_c//2, image=self._assets["bg_p"])
        self.id_vid = self.cvs_p.create_image(w_c//2, h_c//2 - 15, image=None, state=tk.HIDDEN)
        self.id_cd = self.cvs_p.create_text(w_c//2, h_c//2 - 15, text="", font=("Segoe UI", -85, "bold"), fill="#FFEB3B", state=tk.HIDDEN)
        self._assets["icon_you"] = create_centered_emoji_image("👤")
        self.id_you_emoji = self.cvs_p.create_image(w_c//2, h_c//2 - 15, image=self._assets["icon_you"])
        self.cvs_p.create_text(w_c//2, h_c - 25, text="You", font=("Segoe UI", -21), fill="#FFFFFF")

        self.cvs_c = tk.Canvas(cards, width=w_c, height=h_c, bg=COL_SURFACE, highlightthickness=0)
        self.cvs_c.pack(side=tk.LEFT, padx=(15, 0))
        self.cvs_c.create_image(w_c//2, h_c//2, image=self._assets["bg_c"])
        self._assets["icon_cpu"] = create_centered_emoji_image("🤖")
        self.id_cpu_emoji = self.cvs_c.create_image(w_c//2, h_c//2 - 15, image=self._assets["icon_cpu"])
        self.cvs_c.create_text(w_c//2, h_c - 25, text="Computer", font=("Segoe UI", -21), fill="#FFFFFF")

        self.footer = tk.Frame(center, bg=COL_SURFACE)
        self.footer.pack(fill=tk.BOTH, expand=True)

        # STATE: IDLE (How to play + Start)
        self.f_idle = tk.Frame(self.footer, bg=COL_SURFACE)
        hw_w, hw_h = 710, 125
        self._assets["hw_bg"] = create_rounded_rect_image(hw_w, hw_h, 12, "#F4F9FF", "#BBD8F5", 1)
        hw_cvs = tk.Canvas(self.f_idle, width=hw_w, height=hw_h, bg=COL_SURFACE, highlightthickness=0)
        hw_cvs.pack(side=tk.TOP, pady=(0, 5))
        hw_cvs.create_image(hw_w//2, hw_h//2, image=self._assets["hw_bg"])
        hw_cvs.create_text(25, 20, anchor=tk.W, text="How to Play:", font=("Segoe UI", -18, "bold"), fill="#194B8A")
        hw_cvs.create_text(25, 45, anchor=tk.W, text="• Click \"Start Game\" to begin", font=("Segoe UI", -14), fill="#2962FF")
        hw_cvs.create_text(25, 70, anchor=tk.W, text="• Wait for the 3-second countdown", font=("Segoe UI", -14), fill="#2962FF")
        hw_cvs.create_text(25, 95, anchor=tk.W, text="• Click on Rock (✊), Paper (✋), or Scissors (✌️) to make your choice", font=("Segoe UI", -14), fill="#2962FF")

        st_w, st_h = 240, 50
        self._assets["btn_start"] = create_rounded_rect_image(st_w, st_h, st_h//2, COL_BTN_PLAY)
        btn_start_cvs = tk.Canvas(self.f_idle, width=st_w, height=st_h, bg=COL_SURFACE, highlightthickness=0, cursor="hand2")
        btn_start_cvs.pack(side=tk.TOP, pady=(5, 0))
        btn_start_cvs.create_image(st_w//2, st_h//2, image=self._assets["btn_start"])
        btn_start_cvs.create_text(st_w//2, st_h//2, text="▷ Start Game", font=("Segoe UI", -18, "bold"), fill="#FFFFFF")
        btn_start_cvs.bind("<Button-1>", lambda e: self._start())
        self.f_idle.pack() 

        # STATE: RESULT
        self.f_result = tk.Frame(self.footer, bg=COL_SURFACE)
        self.lbl_res_t = tk.Label(self.f_result, text="", font=("Segoe UI", -47, "bold"), fg=COL_RESULT_DRAW, bg=COL_SURFACE)
        self.lbl_res_t.pack(side=tk.TOP, pady=(0, 5))
        self.lbl_res_d = tk.Label(self.f_result, text="", font=("Segoe UI", -21), fg=COL_MUTED, bg=COL_SURFACE)
        self.lbl_res_d.pack(side=tk.TOP, pady=(0, 20))
        
        btn_c = tk.Frame(self.f_result, bg=COL_SURFACE); btn_c.pack(anchor=tk.CENTER)
        w_b, h_b = 160, 50
        self._assets["btn_play"] = create_rounded_rect_image(w_b, h_b, h_b//2, COL_BTN_PLAY)
        self._assets["btn_res"] = create_rounded_rect_image(w_b, h_b, h_b//2, COL_BTN_RESET)
        b_p_cvs = tk.Canvas(btn_c, width=w_b, height=h_b, bg=COL_SURFACE, highlightthickness=0, cursor="hand2"); b_p_cvs.pack(side=tk.LEFT, padx=10)
        b_p_cvs.create_image(w_b//2, h_b//2, image=self._assets["btn_play"])
        b_p_cvs.create_text(w_b//2, h_b//2, text="▷ Play Again", font=("Segoe UI", -17, "bold"), fill="#FFFFFF")
        b_p_cvs.bind("<Button-1>", lambda e: self._start())
        b_r_cvs = tk.Canvas(btn_c, width=w_b, height=h_b, bg=COL_SURFACE, highlightthickness=0, cursor="hand2"); b_r_cvs.pack(side=tk.LEFT, padx=10)
        b_r_cvs.create_image(w_b//2, h_b//2, image=self._assets["btn_res"])
        b_r_cvs.create_text(w_b//2, h_b//2, text="↻ Reset", font=("Segoe UI", -17, "bold"), fill="#FFFFFF")
        b_r_cvs.bind("<Button-1>", lambda e: self._reset())

        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _set_card_icon(self, which: str, emoji: str):
        key = f"icon_{which}"
        self._assets[key] = create_centered_emoji_image(emoji)
        item = self.id_you_emoji if which == "you" else self.id_cpu_emoji
        cvs = self.cvs_p if which == "you" else self.cvs_c
        cvs.itemconfig(item, image=self._assets[key])

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def _start_confetti(self):
        self.conf_win.geometry(f"{self.root.winfo_width()}x{self.root.winfo_height()}+{self.root.winfo_rootx()}+{self.root.winfo_rooty()}")
        self.conf_win.deiconify()
        self.conf_cvs.delete("all")
        self.particles.clear()
        
        w = self.root.winfo_width()
        for _ in range(120):
            x = random.randint(0, w)
            y = random.randint(-500, -50)
            vx = random.uniform(-8, 8)
            vy = random.uniform(8, 16)
            s = random.uniform(6, 12)
            ang = random.uniform(0, math.pi*2)
            sa = random.uniform(0.05, 0.2)
            rot = random.uniform(0, math.pi*2)
            sr = random.uniform(-0.1, 0.1)
            c = random.choice(["#FF3366", "#33CCFF", "#FFCC00", "#00FF66", "#9933FF"])
            pid = self.conf_cvs.create_polygon(0,0,0,0,0,0,0,0, fill=c, outline="")
            self.particles.append([pid, x, y, vx, vy, s, ang, sa, rot, sr])

    def _step_confetti(self):
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        active = []
        for p in self.particles:
            pid, x, y, vx, vy, s, ang, sa, rot, sr = p
            y += vy
            x += vx + math.sin(y * 0.04) * 2.5
            vy += 0.8
            ang += sa
            rot += sr
            if y > h + 100:
                self.conf_cvs.delete(pid)
            else:
                p[1], p[2], p[3], p[4], p[6], p[8] = x, y, vx, vy, ang, rot
                hh = s * math.cos(ang)
                wh = s
                pts = [(-wh, -hh), (wh, -hh), (wh, hh), (-wh, hh)]
                cr, sr_ = math.cos(rot), math.sin(rot)
                t = []
                for px, py in pts:
                    t.extend([px*cr - py*sr_ + x, px*sr_ + py*cr + y])
                self.conf_cvs.coords(pid, *t)
                self.conf_win.lift()
                active.append(p)
        self.particles = active
        if not self.particles:
            self.conf_win.withdraw()

    def _start(self):
        if self.state == "countdown": return
        self.conf_win.withdraw()
        self.particles.clear()
        
        self.f_idle.pack_forget()
        self.f_result.pack_forget()
        
        self.cvs_c.itemconfig(self.id_cpu_emoji, state=tk.NORMAL)
        self._set_card_icon("cpu", "🤖")
        self.cvs_p.itemconfig(self.id_you_emoji, state=tk.HIDDEN)
        self.cvs_p.itemconfig(self.id_vid, state=tk.NORMAL)
        
        self.computer_move = self.player_move = None
        self._gestures.clear(); self._capture = False
        self.state, self.countdown_val = "countdown", 3
        self.cvs_p.itemconfig(self.id_cd, text=str(self.countdown_val), state=tk.NORMAL, font=("Segoe UI", -95, "bold"))
        self._tick()

    def _tick(self):
        if self.state != "countdown": return
        if self.countdown_val > 0:
            self.cvs_p.itemconfig(self.id_cd, text=str(self.countdown_val))
            self.countdown_val -= 1
            self._job = self.root.after(1000, self._tick)
        else:
            self.cvs_p.itemconfig(self.id_cd, text="Shoot!", font=("Segoe UI", -63, "bold"))
            self._gestures.clear(); self._capture = True
            self._job = self.root.after(500, self._finish)

    def _finish(self):
        if self.state != "countdown": return
        self._capture, self._job, self.state = False, None, "result"
        self.cvs_p.itemconfig(self.id_cd, state=tk.HIDDEN)
        c_m = random.choice(list(Move))
        p_m = Counter(self._gestures).most_common(1)[0][0] if self._gestures else None

        self._set_card_icon("cpu", MOVE_EMOJI[c_m])
        self.cvs_p.itemconfig(self.id_vid, state=tk.HIDDEN)
        self.cvs_p.itemconfig(self.id_you_emoji, state=tk.NORMAL)

        self.f_idle.pack_forget()
        self.f_result.pack()

        if not p_m:
            self.lbl_res_t.config(text="Could not read your gesture.", fg=COL_RESULT_DRAW)
            self.lbl_res_d.config(text="")
            self._set_card_icon("you", "?")
            return

        self._set_card_icon("you", MOVE_EMOJI[p_m])
        det = f"You: {MOVE_EMOJI[p_m]}  vs  Computer: {MOVE_EMOJI[c_m]}"
        self.lbl_res_d.config(text=det)
        
        w = winner(p_m, c_m)
        if w > 0:
            self.score_you += 1; self.lbl_sc_y.config(text=str(self.score_you))
            self.lbl_res_t.config(text="You won!", fg=COL_RESULT_WIN)
            self._start_confetti()
        elif w < 0:
            self.score_cpu += 1; self.lbl_sc_c.config(text=str(self.score_cpu))
            self.lbl_res_t.config(text="You lost!", fg=COL_RESULT_LOSE)
        else:
            self.lbl_res_t.config(text="It's a Draw!", fg=COL_RESULT_DRAW)

    def _reset(self):
        if self._job: self.root.after_cancel(self._job); self._job = None
        self.conf_win.withdraw()
        self.particles.clear()
        
        self.score_you = self.score_cpu = 0
        self.lbl_sc_y.config(text="0"); self.lbl_sc_c.config(text="0")
        self.f_result.pack_forget()
        self.f_idle.pack(fill=tk.BOTH)
        self.cvs_p.itemconfig(self.id_you_emoji, state=tk.NORMAL)
        self._set_card_icon("you", "👤")
        self.cvs_p.itemconfig(self.id_vid, state=tk.HIDDEN)
        self._set_card_icon("cpu", "🤖")
        self.state = "idle"

    def _close(self):
        if self._job: self.root.after_cancel(self._job)
        if self.cap: self.cap.release()
        self._landmarker.close()
        self.root.destroy()

    def _loop(self):
        if self.cap and self.cap.isOpened():
            ok, f = self.cap.read()
            if ok:
                f = cv2.flip(f, 1)
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                mpf = mp_image.Image(mp_image.ImageFormat.SRGB, np.ascontiguousarray(rgb))
                self._video_ts += 33
                res = self._landmarker.detect_for_video(mpf, self._video_ts)
                
                g = None
                if res.hand_landmarks:
                    for lm in res.hand_landmarks:
                        vision.drawing_utils.draw_landmarks(f, lm, vision.HandLandmarksConnections.HAND_CONNECTIONS)
                        g = classify_gesture_tasks(lm)
                        if g and self.state == "countdown" and self._capture:
                            self._gestures.append(g)

                pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                pil = pil.resize((VIDEO_DISP_W, VIDEO_DISP_H), Image.Resampling.LANCZOS)
                m = Image.new("L", (VIDEO_DISP_W, VIDEO_DISP_H), 0)
                ImageDraw.Draw(m).rounded_rectangle((0, 0, VIDEO_DISP_W-1, VIDEO_DISP_H-1), radius=16, fill=255)
                pil.putalpha(m)
                
                self._assets["live"] = ImageTk.PhotoImage(image=pil)
                self.cvs_p.itemconfig(self.id_vid, image=self._assets["live"])

        if self.particles:
            self._step_confetti()
            
        self.root.after(33, self._loop)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    RPSApp().run()
