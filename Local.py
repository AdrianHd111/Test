import sys
import time
import datetime
import json
from queue import Queue
import threading
import numpy as np

# Tymczasowy hack, żeby np.float znów istniało
if not hasattr(np, 'float'):
    np.float = float

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from climbcheck import process_frame
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# --- wczytanie konfiguracji ---
with open('config.json', 'r') as f:
    cfg = json.load(f)

MONITOR        = cfg['monitor']
MODEL_PATH     = cfg['model_path']
TH             = cfg['thresholds']
CONFIRM_DELAY  = cfg['confirm_delay']
ALARM_DURATION = cfg['alarm_duration']
BBOX_COLORS    = [tuple(c) for c in cfg['bbox_colors']]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Screen + YOLOv8")

        # 1) inicjalizacja DeepSort
        # ustaw max_age i inne parametry według potrzeb
        self.tracker = DeepSort(max_age=30)

        # 2) słownik: track_id -> lista bboxów [(frame_id, (x1,y1,x2,y2)), ...]
        self.track_histories = defaultdict(list)
    
        # ile klatek z rzędu może nie być liny, zanim zaczniemy timer
        self.max_consecutive_missing = 5  

        # per id:
        self.no_rope_counts             = defaultdict(int)   # track_id -> kolejne klatki bez liny
        self.rope_missing_since_by_id   = {}                 # track_id -> timestamp rozpoczęcia odliczania
        self.spider_alert_by_id         = {}                 # track_id -> czy alarm uruchomiony
        self.alarm_start_time_by_id     = {}                 # track_id -> timestamp startu alarmu (zamiennie)

        # licznik alarmów w bieżącym dniu
        self.daily_date  = datetime.date.today()
        self.daily_count = 0

        # stany per-strefa: left/right
        self.rope_missing_since = {"left": None, "right": None}
        self.alarm_start_time   = {"left": None, "right": None}
        self.spider_alert       = {"left": False, "right": False}

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # model YOLO
        self.model      = YOLO(MODEL_PATH)
        self.labels     = self.model.names
        self.name2id    = {n: i for i, n in self.labels.items()}
        self.idx_with   = self.name2id["human_with_rope"]
        self.idx_no     = self.name2id["human_without_rope"]

        self.min_thresh   = [TH['min_thresh']]
        self.skip_rate    = TH['skip_rate']
        self.min_box_frac = TH['min_box_frac']
        self.bbox_colors  = BBOX_COLORS

        self.cam_w = MONITOR['width']
        self.cam_h = MONITOR['height']
        self.latest_frame = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)
        
        # wątek przechwytywania
        self.frame_queue = Queue()
        self.stop_event  = threading.Event()
        threading.Thread(target=self.camera_thread, daemon=True).start()

        # timer UI & FPS
        self.global_count = 0
        self.fps          = 0.0
        self.fps_timer    = cv2.getTickCount()
        self.timer        = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)

    def camera_thread(self):
        sct = mss()
        last_det = None
        mid_x = self.cam_w // 2
        frame_id = 0

        while not self.stop_event.is_set():
            now = time.time()
            # reset dziennego licznika o północy
            today = datetime.date.today()
            if today != self.daily_date:
                self.daily_date  = today
                self.daily_count = 0

            # zakończ alarm po czasie ALARM_DURATION
            for zone in ("left", "right"):
                if (self.spider_alert[zone] and
                    self.alarm_start_time[zone] is not None and
                    now - self.alarm_start_time[zone] >= ALARM_DURATION):
                    self.spider_alert[zone]       = False
                    self.rope_missing_since[zone] = None
                    print(f"[{zone.upper()}] ALARM STOP po {ALARM_DURATION}s")

            img   = sct.grab(MONITOR)
            frame = cv2.resize(np.array(img)[:, :, :3], (self.cam_w, self.cam_h))
            frame_id += 1

            if frame_id % self.skip_rate == 0:
                # 1) pełna detekcja
                frame, last_det = process_frame(
                    frame, self.min_thresh[0], self.model,
                    self.labels, self.bbox_colors, do_update=True
                )

                # 2) zbuduj surowe detekcje dla DeepSort
                detections = []
                for *box, conf, cls in (last_det or []):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    # ([left, top, width, height], confidence, class_id)
                    detections.append(([x1, y1, w, h], float(conf), int(cls)))

                # 3) zaktualizuj tracker tylko jeśli są detekcje
                if detections:
                    tracks = self.tracker.update_tracks(detections, frame=frame)
                else:
                    tracks = []

                # 4) przejdź po potwierdzonych trackach
                now = time.time()

                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    tid = t.track_id

                    # 1) Pobierz klasę, którą przekazaliśmy jako "others" do DeepSort:
                    detected_cls = t.others  # powinno być albo idx_with, albo idx_no

                    # 2) Ustal, czy lina jest widoczna:
                    has_rope = (detected_cls == self.idx_with)
                    no_rope  = (detected_cls == self.idx_no)

                    # 3) Mechanizm tłumienia: licznik kolejnych klatek bez liny
                    if has_rope:
                        # jeżeli lina pojawiła się z powrotem – resetujemy wszystko
                        self.no_rope_counts[tid] = 0
                        self.rope_missing_since_by_id[tid] = None
                        self.spider_alert_by_id[tid] = False

                    elif no_rope:
                        # każda klatka bez liny zwiększa licznik
                        self.no_rope_counts[tid] += 1

                    else:
                        # ani has_rope ani no_rope (np. inna klasa) – reset licznika
                        self.no_rope_counts[tid] = 0

                    # 4) Gdy licznik przekroczy próg, zaczynamy odliczać CONFIRM_DELAY
                    if self.no_rope_counts[tid] >= self.max_consecutive_missing:
                        # jeśli jeszcze nie rozpoczęliśmy odliczania
                        if self.rope_missing_since_by_id.get(tid) is None:
                            self.rope_missing_since_by_id[tid] = now

                        # jeśli już odliczamy i minął CONFIRM_DELAY, uruchamiamy alarm
                        elif (not self.spider_alert_by_id.get(tid, False)
                            and now - self.rope_missing_since_by_id[tid] >= CONFIRM_DELAY):
                            self.spider_alert_by_id[tid]     = True
                            self.alarm_start_time_by_id[tid] = now
                            self.daily_count += 1
                            print(f"[{datetime.datetime.now()}] ALARM for ID={tid}")

                    # 5) (opcjonalnie) narysuj bbox i ID, zaznacz kolorem, jeśli jest alert
                    x, y, w, h = t.to_ltwh()
                    x2, y2 = x + w, y + h
                    color = (0,0,255) if self.spider_alert_by_id.get(tid, False) else (255,0,0)
                    cv2.rectangle(frame, (int(x),int(y)), (int(x2),int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (int(x), int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



                # przygotuj detekcje per-strefa
                saw_no   = {"left": False, "right": False}
                saw_with = {"left": False, "right": False}

                for *box, conf, cls in (last_det or []):
                    x1, y1, x2, y2 = box
                    cls = int(cls)
                    zone = "left" if (x1 + x2) / 2 < mid_x else "right"
                    # tylko jeśli bbox ma odpowiednią wysokość
                    if cls == self.idx_no and y2 <= (1 - self.min_box_frac) * self.cam_h:
                        saw_no[zone] = True
                    if cls == self.idx_with:
                        saw_with[zone] = True

                # logika per-strefa z debugiem
                for zone in ("left", "right"):
                    if saw_with[zone]:
                        # lina wykryta — reset całego cyklu alarmowego
                        if (self.rope_missing_since[zone] is not None or
                            self.spider_alert[zone]):
                            print(f"[{zone.upper()}] RESET – wykryto linę, anuluję proces alarmowy.")
                        self.rope_missing_since[zone] = None
                        self.spider_alert[zone]       = False
                        self.alarm_start_time[zone]   = None

                    elif saw_no[zone]:
                        # brak liny: start pomiaru lub alarm
                        if self.rope_missing_since[zone] is None:
                            self.rope_missing_since[zone] = now
                            print(f"[{zone.upper()}] START pomiaru – brak liny wykryty.")
                        elif (not self.spider_alert[zone] and
                              now - self.rope_missing_since[zone] >= CONFIRM_DELAY):
                            self.spider_alert[zone]     = True
                            self.alarm_start_time[zone] = now
                            self.daily_count           += 1
                            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{ts}] ALARM START w strefie: {zone.upper()}")

            self.frame_queue.put(frame)

    def update_frame(self):
        # pobierz najnowszą klatkę
        while not self.frame_queue.empty():
            self.latest_frame = self.frame_queue.get()

        # liczenie FPS
        self.global_count += 1
        if self.global_count >= 20:
            now = cv2.getTickCount()
            self.fps = 20.0 / ((now - self.fps_timer) / cv2.getTickFrequency())
            self.fps_timer    = now
            self.global_count = 0

        disp = self.latest_frame.copy()

        # wyświetlanie alarmów per strefa
        if self.spider_alert["left"]:
            cv2.putText(disp, "UWAGA SPIDERMAN (LEWA)!",
                        (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        if self.spider_alert["right"]:
            cv2.putText(disp, "UWAGA SPIDERMAN (PRAWA)!",
                        (self.cam_w//2 + 10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        # legenda + licznik
        legend = (
            f"Q=exit  +/=up thr  - =down thr  thr={self.min_thresh[0]:.2f}"
            f"  FPS={self.fps:.1f}  Alarmy dzisiaj={self.daily_count}"
        )
        (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        overlay = disp.copy()
        cv2.rectangle(overlay,
                      (5, self.cam_h - th - 10),
                      (5 + tw + 10, self.cam_h - 5),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)
        cv2.putText(disp, legend, (10, self.cam_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # konwersja do QPixmap
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()
        elif e.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.min_thresh[0] = min(1.0, self.min_thresh[0] + 0.05)
        elif e.key() == Qt.Key_Minus:
            self.min_thresh[0] = max(0.0, self.min_thresh[0] - 0.05)
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.stop_event.set()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
