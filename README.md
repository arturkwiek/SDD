# Simple Drone Detector (SDD)

Prosty projekt w Pythonie do detekcji obiektów (np. dronów) w:

- pojedynczym obrazie
- pliku wideo
- strumieniu z kamery USB
- strumieniu sieciowym (RTSP/HTTP)

Domyślna implementacja używa modelu YOLO (biblioteka `ultralytics`) jako
ogólnego detektora obiektów. Aby uzyskać lepsze wyniki dla dronów, można
podmienić model na checkpoint nauczony specyficznie do tego zadania.

## Instalacja (Windows, PowerShell)

W katalogu projektu:

```powershell
py -3.10 -m venv .venv-win
. .\.venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uruchomienie (CLI)

Z aktywnym środowiskiem `.venv-win` ustaw zmienną `PYTHONPATH` i uruchom moduł CLI:

```powershell
$env:PYTHONPATH = "src"
python -m sdd.cli <source>
```

Gdzie `<source>` to:

- ścieżka do obrazu lub pliku wideo (np. `C:/VideoSource/drones.jpg`, `C:/VideoSource/drones.mp4`)
- indeks kamery, np. `0` dla pierwszej kamery USB
- URL strumienia (np. RTSP/HTTP kamery IP)

Przykłady (Windows, PowerShell):

```powershell
# Plik wideo, zapis wideo wynikowego + logi JSON/CSV
$env:PYTHONPATH = "src"
python -m sdd.cli "C:/VideoSource/drones.mp4" `
	--save-video --video-out drones_out.mp4 `
	--json-out drones.json --csv-out drones.csv

# Kamera USB 0 z podglądem
$env:PYTHONPATH = "src"
python -m sdd.cli 0

# Strumień RTSP bez podglądu
$env:PYTHONPATH = "src"
python -m sdd.cli "rtsp://user:pass@host/stream" --no-preview
```

Opcjonalnie można użyć przełącznika `--from-samples`, który interpretuje
`<source>` względnie do katalogu `SAMPLE_PATH` z configu (domyślnie
`C:/VideoSource` na Windows):

```powershell
$env:PYTHONPATH = "src"
python -m sdd.cli drones.mp4 --from-samples
```

Wyniki pojedynczego uruchomienia są zapisywane jako:

- `detections.json` – lista wszystkich detekcji (ramka, czas, bbox, klasa, pewność)
- `detections.csv` – ten sam log w formacie CSV
- `detections_events.csv` – podsumowanie per klasa (liczba, czasy pierwszej/ostatniej detekcji, statystyki score)
- `detections_label_counts.txt` – lista klas z liczbą detekcji (wygodne do szybkiego podejrzenia)

Plik `*_events.csv` jest tworzony automatycznie na podstawie ścieżki podanej
przez `--csv-out`. Przykładowo `--csv-out wyniki.csv` spowoduje utworzenie
pliku `wyniki_events.csv`.

### Główne opcje CLI

- `--model` – ścieżka/nazwa checkpointu YOLO (domyślnie `yolov8n.pt`)
- `--conf` – próg pewności detekcji (domyślnie `0.35`)
- `--iou` – próg IoU dla NMS (domyślnie `0.45`)
- `--target-classes` – lista nazw klas, które mają być zachowane (np. `drone`)
- `--no-preview` – wyłącza okno podglądu OpenCV
- `--save-video` – włącza zapis wideo z narysowanymi detekcjami
- `--video-out` – ścieżka pliku wideo wyjściowego (domyślnie `output_detection.mp4`)
- `--json-out` – ścieżka pliku JSON z detekcjami (domyślnie `detections.json`)
- `--csv-out` – ścieżka pliku CSV z detekcjami (domyślnie `detections.csv`)
- `--from-samples` – interpretuje `source` względem katalogu `SAMPLE_PATH`

## Dalszy rozwój

- Podmiana modelu na specjalistyczny model dronów (zmiana `--model` lub konfiguracji).
- Dodanie filtrów logiki biznesowej (np. alarm, liczenie przelotów, strefy).
- Integracja z systemami zewnętrznymi (MQTT, REST itd.).
