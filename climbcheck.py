import cv2
import numpy as np


def draw_warning_box(frame, box, label, color=(0, 0, 255)):
    """Rysuje prostokąt i etykietę ostrzeżenia."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def is_on_wall(box, frame_height):
    """
    Sprawdza, czy obiekt znajduje się na ścianie wspinaczkowej,
    czyli NIE w dolnych 10% obrazu (na ziemi).

    Parametry:
        box: [x1, y1, x2, y2] – współrzędne ramki wykrycia
        frame_height: wysokość całego obrazu

    Zwraca:
        True – jeśli obiekt jest na ścianie,
        False – jeśli obiekt jest blisko ziemi (dolne 10%).
    """
    _, y1, _, y2 = box
    bottom_of_box = max(y1, y2)
    # Granica dolnych 10% kadru
    ground_level = frame_height * 0.9
    return bottom_of_box < ground_level


def process_frame(frame, thresh, model, labels, colors, last_detections=None, do_update=True):
    """
    Przetwarza pojedynczą klatkę:
    - wykonuje predykcję co N-tą klatkę (jeśli do_update = True),
    - lub wykorzystuje poprzednie detekcje.
    """
    results = []
    frame_height, frame_width = frame.shape[:2]

    if do_update:
        results = model.predict(frame, verbose=False)[
            0].boxes.data.cpu().numpy()
        detections = []
        for box in results:
            x1, y1, x2, y2, conf, cls = box
            if conf < thresh:
                continue
            detections.append([x1, y1, x2, y2, conf, int(cls)])
        last_detections = detections

    # Rysujemy detekcje
    for det in last_detections or []:
        x1, y1, x2, y2, conf, cls = det
        label = labels[cls]
        color = colors[cls % len(colors)]
        box = (x1, y1, x2, y2)

        if label == "human_without_rope" and is_on_wall(box, frame_height):
            draw_warning_box(frame, box, f"Uwaga Spider-Man!",
                             color=(0, 0, 255))
        else:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, 1)
            cv2.putText(frame, label, (int(x1), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, last_detections
