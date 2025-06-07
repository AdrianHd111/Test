import sys
import time
import datetime
import json
from queue import Queue
import threading

import cv2
import numpy as np
from ultralytics import YOLO
from climbcheck import process_frame

from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# --- wczytanie konfiguracji ---
with open('config.json', 'r') as f:
    cfg = json.load(f)

RTSP_URL = cfg['rtsp_url']
MODEL_PATH = cfg['model_path']
TH = cfg['thresholds']
CONFIRM_DELAY = cfg['confirm_delay']
ALARM_DURATION = cfg['alarm_duration']
BBOX_COLORS = [tuple(c) for c in cfg['bbox_colors']]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTSP Stream + YOLOv8")

        self.daily_date = datetime.date.today()
        self.daily_count = 0
        self.rope_missing_since = None
        self.alarm_start_time = None
        self.spider_alert = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # strumień RTSP
        self.cap = cv2.VideoCapture(RTSP_URL)
        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.latest_frame = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)

        self.model = YOLO(MODEL_PATH)
        self.labels = self.model.names
        self.name2id = {n: i for i, n in self.labels.items()}
        self.idx_with = self.name2id["human_with_rope"]
        self.idx_no = self.name2id["human_without_rope"]

        self.min_thresh = [TH['min_thresh']]
        self.skip_rate = TH['skip_rate']
        self.min_box_frac = TH['min_box_frac']
        self.bbox_colors = BBOX_COLORS

        self.frame_queue = Queue()
        self.stop_event = threading.Event()
        threading.Thread(target=self.camera_thread, daemon=True).start()

        self.global_count = 0
        self.fps = 0.0
        self.fps_timer = cv2.getTickCount()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)

    def camera_thread(self):
        frame_id = 0
        last_det = None
        while not self.stop_event.is_set():
            # reset licznika o północy
            today = datetime.date.today()
            if today != self.daily_date:
                self.daily_date = today
                self.daily_count = 0

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (self.cam_w, self.cam_h))
            now = time.time()
            frame_id += 1

            # zakończenie trwającego alarmu
            if self.spider_alert and (now - self.alarm_start_time) >= ALARM_DURATION:
                self.spider_alert = False
                self.rope_missing_since = None

            if frame_id % self.skip_rate == 0:
                frame, last_det = process_frame(
                    frame, self.min_thresh[0], self.model,
                    self.labels, self.bbox_colors, do_update=True
                )

                saw_no = saw_with = False
                if last_det:
                    for *box, conf, cls in last_det:
                        x1, y1, x2, y2 = box
                        cls = int(cls)
                        if cls == self.idx_no and y2 <= (1-self.min_box_frac)*self.cam_h:
                            saw_no = True
                        if cls == self.idx_with:
                            saw_with = True

                if saw_with:
                    self.rope_missing_since = None
                    self.spider_alert = False

                elif saw_no:
                    if self.rope_missing_since is None:
                        self.rope_missing_since = now
                    elif not self.spider_alert and (now - self.rope_missing_since) >= CONFIRM_DELAY:
                        self.spider_alert = True
                        self.alarm_start_time = now
                        self.daily_count += 1
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{ts}] ALARM START (RTSP)")

            self.frame_queue.put(frame)

    def update_frame(self):
        while not self.frame_queue.empty():
            self.latest_frame = self.frame_queue.get()

        self.global_count += 1
        if self.global_count >= 20:
            now = cv2.getTickCount()
            self.fps = 20.0/((now-self.fps_timer)/cv2.getTickFrequency())
            self.fps_timer = now
            self.global_count = 0

        disp = self.latest_frame.copy()
        if self.spider_alert:
            cv2.putText(disp, "UWAGA SPIDERMAN!",
                        (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        legend = (f"thr={self.min_thresh[0]:.2f}  FPS={self.fps:.1f}"
                  f"  Alarmy dzisiaj={self.daily_count}")
        (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ov = disp.copy()
        cv2.rectangle(ov, (5, self.cam_h-th-10),
                      (5+tw+10, self.cam_h-5), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, disp, 0.4, 0, disp)
        cv2.putText(disp, legend, (10, self.cam_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()
        elif e.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.min_thresh[0] = min(1.0, self.min_thresh[0]+0.05)
        elif e.key() == Qt.Key_Minus:
            self.min_thresh[0] = max(0.0, self.min_thresh[0]-0.05)
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.stop_event.set()
        self.cap.release()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
