import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import time
from nn import EmotionCNN

EMOTIONS = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


class EmotionDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection System")
        self.root.geometry("920x720")
        self.root.minsize(640, 480)

        self.cnn = EmotionCNN('model.h5')
        self.face_net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )

        self.cam_active = False
        self.cap = None
        self.current_frame = None
        self.video_thread_running = False

        self._after_id = None
        self.setup_ui()
        self.update_controls()

    def setup_ui(self):
        # --- Mode selection ---
        mode_frame = ttk.LabelFrame(self.root, text=" Input Mode ", padding=8)
        mode_frame.pack(fill="x", padx=10, pady=(10, 0))

        self.mode_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(mode_frame, text="Webcam", variable=self.mode_var,
                        value="webcam", command=self.on_mode_change).pack(side="left", padx=12)
        ttk.Radiobutton(mode_frame, text="Image", variable=self.mode_var,
                        value="image", command=self.on_mode_change).pack(side="left", padx=12)
        ttk.Radiobutton(mode_frame, text="Video", variable=self.mode_var,
                        value="video", command=self.on_mode_change).pack(side="left", padx=12)

        # --- Display ---
        display_frame = ttk.LabelFrame(self.root, text=" Output ", padding=5)
        display_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.display_label = ttk.Label(display_frame)
        self.display_label.pack(fill="both", expand=True)
        self.show_placeholder()

        # --- Controls ---
        control_frame = ttk.Frame(self.root, padding=8)
        control_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.btn_action = ttk.Button(control_frame, text="Start Camera", command=self.action_button)
        self.btn_action.pack(side="left", padx=4)

        self.btn_browse = ttk.Button(control_frame, text="Browse...", command=self.browse_file)
        self.btn_browse.pack(side="left", padx=4)

        self.btn_save = ttk.Button(control_frame, text="Save Output", command=self.save_output)
        self.btn_save.pack(side="left", padx=4)

        # --- Progress bar (hidden) ---
        self.progress = ttk.Progressbar(self.root, mode='determinate')

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief="sunken", anchor="w", padding=4)
        status_bar.pack(fill="x", padx=10, pady=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------------------------------------------

    def show_placeholder(self):
        ph = np.ones((480, 640, 3), dtype=np.uint8) * 230
        cv2.putText(ph, "Select a mode and start", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2, cv2.LINE_AA)
        self.display_image(ph)

    def display_image(self, cv_img):
        h, w = cv_img.shape[:2]
        mw, mh = 860, 540
        scale = min(mw / w, mh / h, 1.2)
        if scale != 1:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=interp)

        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.display_label.imgtk = imgtk
        self.display_label.configure(image=imgtk)

    # ----------------------------------------------------------------

    def on_mode_change(self):
        self.stop_webcam()
        self.update_controls()
        self.show_placeholder()
        self.status_var.set("Ready")

    def update_controls(self):
        mode = self.mode_var.get()
        if mode == "webcam":
            self.btn_action.config(state="normal", text="Start Camera")
            self.btn_browse.config(state="disabled")
            self.btn_save.config(state="normal")
        elif mode == "image":
            self.btn_action.config(state="disabled", text="Start Camera")
            self.btn_browse.config(state="normal", text="Browse Image...")
            self.btn_save.config(state="normal")
        elif mode == "video":
            self.btn_action.config(state="disabled", text="Start Camera")
            self.btn_browse.config(state="normal", text="Browse Video...")
            self.btn_save.config(state="disabled")

    def action_button(self):
        self.toggle_webcam()

    # ----------------------------------------------------------------
    #  Webcam
    # ----------------------------------------------------------------

    def toggle_webcam(self):
        if self.cam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Error: Could not open webcam")
            return
        self.cam_active = True
        self.btn_action.config(text="Stop Camera")
        self.status_var.set("Webcam active")
        self._webcam_loop()

    def stop_webcam(self):
        self.cam_active = False
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_action.config(text="Start Camera")
        if self.mode_var.get() == "webcam":
            self.show_placeholder()
            self.status_var.set("Ready")

    def _webcam_loop(self):
        if not self.cam_active or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            self.status_var.set("Error: Camera disconnected")
            return
        self.current_frame = self._detect_and_annotate(frame)
        self.display_image(self.current_frame)
        self._after_id = self.root.after(30, self._webcam_loop)

    # ----------------------------------------------------------------
    #  Detection
    # ----------------------------------------------------------------

    def _detect_and_annotate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        fh, fw = frame.shape[:2]
        ratio = min(fh, fw) / 480.0
        fnt = round(1.0 * ratio, 2)
        thick = max(1, int(3 * ratio))
        ox, oy = int(20 * ratio), int(60 * ratio)
        emotions = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - oy + 10), (x + w, y + h + 10), (255, 0, 0), thick)
            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, (48, 48))
            pred = self.cnn.predict(roi_resized)
            idx = int(np.argmax(pred))
            label = EMOTIONS[idx]
            emotions.append(label)
            cv2.putText(frame, label, (x + ox, max(y - oy, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, fnt, (255, 255, 255), thick, cv2.LINE_AA)

        if emotions:
            self.status_var.set(f"{len(faces)} face(s) | {emotions[0]}")
        else:
            self.status_var.set("No face detected")
        return frame

    # ----------------------------------------------------------------
    #  Image
    # ----------------------------------------------------------------

    def browse_file(self):
        mode = self.mode_var.get()
        if mode == "image":
            self._browse_image()
        elif mode == "video":
            self._browse_video()

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            self.status_var.set(f"Error: Could not load {os.path.basename(path)}")
            return
        self.current_frame = self._detect_and_annotate(frame)
        self.display_image(self.current_frame)
        self.status_var.set(f"Image: {os.path.basename(path)}")

    # ----------------------------------------------------------------
    #  Video
    # ----------------------------------------------------------------

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if not path:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save processed video as...",
            defaultextension=".avi",
            filetypes=[("AVI", "*.avi"), ("MP4", "*.mp4")]
        )
        if not out_path:
            return
        self._process_video(path, out_path)

    def _process_video(self, in_path, out_path):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            self.status_var.set(f"Error: Could not open {os.path.basename(in_path)}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            self.status_var.set("Error: Could not read video properties")
            cap.release()
            return

        self.progress["value"] = 0
        self.progress["maximum"] = total
        self.progress.pack(fill="x", padx=10, pady=(0, 5))

        self.video_thread_running = True
        self.btn_browse.config(state="disabled")

        def process():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            count = 0
            while self.video_thread_running:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = self._detect_and_annotate(frame)
                writer.write(annotated)
                count += 1
                self.root.after(0, self._update_progress, count)
            cap.release()
            writer.release()
            self.root.after(0, self._video_done, out_path, count)

        threading.Thread(target=process, daemon=True).start()

    def _update_progress(self, count):
        self.progress["value"] = count

    def _video_done(self, out_path, count):
        self.video_thread_running = False
        self.progress.pack_forget()
        self.btn_browse.config(state="normal")
        self.status_var.set(f"Done! {count} frames saved to {os.path.basename(out_path)}")

    # ----------------------------------------------------------------
    #  Save / Close
    # ----------------------------------------------------------------

    def save_output(self):
        if self.current_frame is None:
            self.status_var.set("Nothing to save")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self.status_var.set(f"Saved: {os.path.basename(path)}")

    def on_close(self):
        self.video_thread_running = False
        self.stop_webcam()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorGUI(root)
    app.run()
