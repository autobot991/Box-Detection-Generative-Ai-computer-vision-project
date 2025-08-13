import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
import pygame
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import tkinter as tk
from tkinter import Label, Button, filedialog, ttk
from PIL import Image, ImageTk
from fpdf import FPDF
import csv

# Initialize Gemini API
os.environ["GOOGLE_API_KEY"] = ""

class PackageDetectionProcessor:
    def __init__(self, gui=None):
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.gui = gui
        self.cap = None

        self.yolo_model = YOLO("yolo12s.pt")
        self.names = self.yolo_model.names
        self.cx1 = 416
        self.offset = 6

        self.output_filename = f"package_data_{time.strftime('%Y-%m-%d')}.txt"
        self.csv_filename = f"package_data_{time.strftime('%Y-%m-%d')}.csv"
        self.processed_track_ids = set()
        self.cropped_images_folder = "cropped_packages"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as f:
                f.write("Timestamp | Track ID | Package Type | Front Open | Damage Condition\n")
                f.write("-" * 80 + "\n")

        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Track ID", "Package Type", "Front Open", "Damage Condition"])

        pygame.mixer.init()
        self.alert_sound = "alert.mp3"
        self.sound_playing = False
        self.running = False

    def set_video_source(self, source):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)

    def play_alert(self):
        if not self.sound_playing:
            pygame.mixer.music.load(self.alert_sound)
            pygame.mixer.music.play(-1)
            self.sound_playing = True
            if self.gui:
                self.gui.update_alert_label("ALERT: Front open box detected!", "red")

    def stop_alert(self):
        if self.sound_playing:
            pygame.mixer.music.stop()
            self.sound_playing = False
            if self.gui:
                self.gui.update_alert_label("No alert", "green")

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": """
                    Analyze the given image of a package and extract the following details:
                    - **Box Front Open (Yes/No)**
                    - **Damage Condition (Yes/No)**
                    Return results in table format only:
                    | Package Type (box) | Box Front Flap Open (Yes/No) | Damage Condition (Yes/No) |
                    |--------------------|----------------------------|--------------------------|
                    """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            return "Error"

    def process_crop_image(self, image, track_id):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(self.cropped_images_folder, f"{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_path, image)

        result = self.analyze_image_with_gemini(image_path)
        extracted = result.split("\n")[2:]

        alert_triggered = False
        with open(self.output_filename, "a", encoding="utf-8") as file:
            with open(self.csv_filename, "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for row in extracted:
                    if "-" in row or not row.strip():
                        continue
                    values = [v.strip() for v in row.split("|")[1:-1]]
                    if len(values) == 3:
                        box_type, flap, damage = values
                        file.write(f"{timestamp} | Track ID: {track_id} | {box_type} | {flap} | {damage}\n")
                        writer.writerow([timestamp, track_id, box_type, flap, damage])
                        if flap.lower() == "yes":
                            alert_triggered = True

        if alert_triggered:
            self.play_alert()
        else:
            self.stop_alert()

    def crop_and_process(self, frame, box, track_id):
        if track_id in self.processed_track_ids:
            return
        self.processed_track_ids.add(track_id)

        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]
        threading.Thread(target=self.process_crop_image, args=(cropped, track_id), daemon=True).start()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1]*len(boxes)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                if self.cx1 - self.offset < cx < self.cx1 + self.offset:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {tid}", (x2, y2), 1, 1)
                    self.crop_and_process(frame, box, tid)
        return frame

    def start(self):
        self.running = True
        while self.cap and self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_alert()
                break
            frame = self.process_frame(frame)
            cv2.line(frame, (self.cx1, 0), (self.cx1, 599), (0, 255, 0), 2)
            if self.gui:
                self.gui.update_video(frame)
            if cv2.waitKey(1) == ord("q"):
                break
        if self.cap:
            self.cap.release()
        self.stop_alert()

class PackageDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Package Detection AI System")
        self.label = Label(self.root)
        self.label.pack()

        self.alert_label = Label(self.root, text="No alert", bg="green", fg="white", font=("Arial", 14))
        self.alert_label.pack(fill="x")

        self.select_btn = Button(self.root, text="Select Video File", command=self.select_video)
        self.select_btn.pack(pady=5)

        self.live_btn = Button(self.root, text="Use Live Camera", command=self.use_camera)
        self.live_btn.pack(pady=5)

        self.start_button = Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = Button(self.root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=5)

        self.processor = PackageDetectionProcessor(gui=self)
        self.thread = None

    def select_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            self.processor.set_video_source(filepath)

    def use_camera(self):
        self.processor.set_video_source(0)

    def update_alert_label(self, text, color):
        self.alert_label.config(text=text, bg=color)

    def update_video(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.root.update_idletasks()

    def start_detection(self):
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.processor.start, daemon=True)
            self.thread.start()

    def stop_detection(self):
        self.processor.running = False
        self.update_alert_label("Detection stopped", "grey")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = PackageDetectionGUI()
    gui.run()

