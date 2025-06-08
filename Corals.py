#!/usr/bin/env python3
"""
spiderman_alarm_edgetpu.py
──────────────────────────
Wykrywa wspinacza bez liny modelem TFLite skompilowanym dla Google Coral USB
Accelerator („Edge TPU”) i steruje lampką 12 V przez przekaźnik na GPIO 17.

• Model: best2504_int8_edgetpu.tflite  (2 klasy: 0=rope, 1=norop)
• Lampka zapala się po potwierdzonym braku liny przez ≥ CONFIRM_DELAY s.
• Lampka gaśnie po ALARM_DUR s albo gdy lina znowu się pojawi.
• Przekaźnik jest aktywowany stanem wysokim (active_high=True).

Wymagane pakiety:
    pip install tflite-runtime pycoral opencv-python gpiozero
(Ubuntu/Raspberry Pi OS 64-bit: `sudo apt install python3-pycoral`)
"""

# ------------------------------------------------------------
# Importy
# ------------------------------------------------------------
import sys
import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
from gpiozero import OutputDevice          # sterowanie lampką

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# ------------------------------------------------------------
# Konfiguracja modelu i źródła wideo
# ------------------------------------------------------------
MODEL_PATH   = "/home/pi/Yolo/best2504_int8_edgetpu.tflite"
VIDEO_PATH   = "test.mp4"      # 0 dla kamery /dev/video0

# Progi detekcji i czasy
MIN_THRESH    = 0.5      # minimalna pewność detekcji
MIN_BOX_FRAC  = 0.10     # bbox „norop” musi kończyć się ≥10 % nad dołem klatki
CONFIRM_DELAY = 5.0      # s przed ogłoszeniem alarmu
ALARM_DUR     = 10.0     # s trwania alarmu
SKIP_RATE     = 5        # detekcja co N klatek

# ------------------------------------------------------------
# GPIO – lampka 12 V sterowana przekaźnikiem na BCM 17
# ------------------------------------------------------------
lamp = OutputDevice(17, active_high=True, initial_value=False)

def lamp_on():
    lamp.on()
    print("💡  LAMPKA ON")

def lamp_off():
    lamp.off()
    print("💡  LAMPKA OFF")

# ------------------------------------------------------------
# Inicjalizacja Edge TPU
# ------------------------------------------------------------
if not Path(MODEL_PATH).exists():
    sys.exit(f"❌  Nie znaleziono modelu: {MODEL_PATH}")

print("⏳  Ładowanie modelu TFLite + Edge TPU…")
interpreter = make_interpreter(f"delegate:auto:{MODEL_PATH}")
interpreter.allocate_tensors()

input_height, input_width = common.input_size(interpreter)

# ------------------------------------------------------------
# Inicjalizacja wideo
# ------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"🚨  Nie można otworzyć źródła wideo: {VIDEO_PATH}")

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ------------------------------------------------------------
# Zmienne stanu alarmu / FPS
# ------------------------------------------------------------
rope_since   = None
alarm_start  = None
in_alert     = False

daily_date   = datetime.date.today()
daily_alarms = 0

fps, fps_count, fps_timer = 0.0, 0, time.time()
frame_id = 0

LABELS = {0: "rope", 1: "norop"}

print("🔎  Start – naciśnij 'q' lub Ctrl-C, aby zakończyć")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)        # pętla dla pliku
            continue

        frame_id += 1
        now = time.time()

        # FPS obliczany co 1 s
        fps_count += 1
        if now - fps_timer >= 1.0:
            fps = fps_count / (now - fps_timer)
            fps_count, fps_timer = 0, now

        # reset dzienny licznika alarmów
        if datetime.date.today() != daily_date:
            daily_date, daily_alarms = datetime.date.today(), 0

        # zakończenie alarmu po ALARM_DUR
        if in_alert and alarm_start and (now - alarm_start >= ALARM_DUR):
            print(f"[{datetime.datetime.now()}] ⏹  Alarm OFF")
            in_alert, rope_since = False, None
            lamp_off()

        # detekcja co SKIP_RATE klatek
        if frame_id % SKIP_RATE == 0:
            # przygotuj wejście dla modelu
            img_resized = cv2.resize(frame, (input_width, input_height))
            common.set_input(interpreter, img_resized)
            interpreter.invoke()
            objs = detect.get_objects(interpreter, MIN_THRESH,
                                      image_scale=(frame.shape[0] / input_height,
                                                   frame.shape[1] / input_width))

            # konwersja wyników
            detections = []
            for obj in objs:
                x0, y0, x1, y1 = obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax
                detections.append((x0, y0, x1, y1, obj.score, obj.id))

            # status liny
            has_rope = any(d[5] == 0 for d in detections)
            no_rope  = any(
                d[5] == 1 and d[3] <= (1 - MIN_BOX_FRAC) * cam_h for d in detections
            )

            # logika alarmu
            if has_rope:
                rope_since = None
                if in_alert:
                    in_alert = False
                    lamp_off()

            elif no_rope:
                if rope_since is None:
                    rope_since = now
                    print(f"[{datetime.datetime.now()}] ⚠️  Potencjalny brak liny – odliczam…")
                elif (not in_alert) and (now - rope_since >= CONFIRM_DELAY):
                    in_alert, alarm_start = True, now
                    daily_alarms += 1
                    print(f"[{datetime.datetime.now()}] 🚨  ALARM! (#{daily_alarms})")
                    lamp_on()

            # rysowanie bboxów
            for x0, y0, x1, y1, score, cid in detections:
                color = (0, 255, 0) if cid == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{LABELS[cid]}:{score:.2f}",
                            (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # overlay alarmowy i statystyki
        if in_alert:
            cv2.putText(frame, "!! UWAGA SPIDERMAN !!", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        legend = f"FPS: {fps:.1f}   Alarmy dziś: {daily_alarms}"
        cv2.putText(frame, legend, (10, cam_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Spiderman Alarm – Edge TPU", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n⏹  Przerwano przez użytkownika")

finally:
    cap.release()
    cv2.destroyAllWindows()
    lamp_off()
    print("✅  Zakończono – zasoby zwolnione")
