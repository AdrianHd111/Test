# 🧗‍♂️ System Wykrywania Wspinaczy bez Liny

## 📌 Opis
Ten projekt służy do wykrywania wspinaczy niekorzystających z zabezpieczenia linowego przy pomocy modelu YOLOv8 oraz DeepSort. System może działać w dwóch trybach: lokalnego skanowania ekranu oraz przez podłączenie kamer.

## ✅ Wersje

### `local.py`
- W pełni działająca wersja.
- Skanuje ekran (np. z filmikiem) i analizuje obraz w czasie rzeczywistym.
- Używa modelu YOLOv8 oraz algorytmu DeepSort do śledzenia postaci.
- Wyświetla alerty oraz śledzi liczbę wykrytych przypadków wspinania się bez liny.

### `cameras.py`
- Wersja przystosowana do pracy z kamerami (np. RTSP).
- Wymaga naprawy: obecnie zawiera błędy wykrywane już przy próbie uruchomienia.

## 🍓 Raspberry Pi
Na Raspberry Pi znajduje się folder `rasbery`, który zawiera:

- Wirtualne środowisko `env`
- Różne pliki do testów sprzętowych

### Działające testy:
- `tust.py` – test kontrolujący system alarmowy wraz z `Testdiody.py`
- `Testdiody.py` – włącza zieloną diodę na przekaźniku w trakcie alarmu

## 🔧 Pliki konfiguracyjne

### `config.json`
Zawiera konfigurację systemu:
```json
{
  "monitor": {
    "top": 100,
    "left": 200,
    "width": 1280,
    "height": 720
  },
  "rtsp_url": "rtsp://AdrianHd1110:Kamerka123@172.16.234.13:554/stream1",
  "model_path": "best2504.pt",
  "thresholds": {
    "min_thresh": 0.6,
    "skip_rate": 6,
    "min_box_frac": 0.10
  },
  "confirm_delay": 5.0,
  "alarm_duration": 15.0,
  "bbox_colors": [
    [164, 120, 87],
    [68, 148, 228],
    [93, 97, 209],
    [178, 182, 133],
    [88, 159, 106]
  ]
}
```

## 🧠 Logika detekcji
- Wykrywane są osoby wspinające się z liną (`human_with_rope`) oraz bez liny (`human_without_rope`).
- Brak liny przez określony czas (`confirm_delay`) uruchamia alarm.
- Wspierane są dwa sposoby analizy: per-strefa (lewa/prawa część kadru) oraz per-ID (śledzenie konkretnej osoby).

## 🖼️ Przetwarzanie obrazu (`climbcheck.py`)
- Wykorzystuje YOLOv8 do wykrywania osób.
- Funkcja `process_frame` analizuje pojedynczą klatkę i zwraca przetworzone detekcje.
- Osoby bez liny w górnej części kadru wywołują ostrzeżenie (rysowana czerwona ramka + komunikat).

## 🚀 Uruchomienie
### Lokalnie (np. z filmiku):
```bash
python local.py
```

### Z kamery (wymaga naprawy):
```bash
python cameras.py
```

### Testy na Raspberry Pi:
```bash
source rasbery/env/bin/activate
python rasbery/tust.py
```

## 📋 Zależności
- `ultralytics` (YOLOv8)
- `deep_sort_realtime`
- `mss`
- `PyQt5`
- `OpenCV`

## 🛠️ Do zrobienia
- Naprawa błędu w `cameras.py`
- Testowanie i integracja z kamerami RTSP
- Rozszerzenie obsługi sygnałów alarmowych na Raspberry Pi

---
Autor: Adrian Ciupka

