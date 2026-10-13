# SIGNALVEX — Drone-Based Human Detection and Rescue Mapping

> **Department of Electronics and Communication Engineering**  
> **Chaitanya Bharathi Institute of Technology (CBIT)**  
> **Project Team: Sky Seekers — Searching the Skies. Saving Lives.**

---

## 1. Project Overview

**SIGNALVEX** is a software-only search-and-rescue (SAR) prototype designed to assist ground rescue operators in identifying human survivors from recorded aerial RGB drone footage.

The prototype executes the following core workflow:
```
Recorded Drone Video ➔ Preprocess ➔ YOLOv8 Detection (Class 0: Person) ➔ Confidence Filtering ➔ ByteTrack (Persistent Track IDs) ➔ Detection Logger ➔ Streamlit Dashboard ➔ Human Operator Review (Keep / Reject / Needs Review) ➔ Exported CSV Log
```

> **Important Scope Note:** In strict compliance with the project proposal, this submission implements the **RGB-video human detection and tracking pipeline**. Physical drone hardware, live streaming, thermal sensors, RF sensing, and real GPS telemetry are explicitly reserved for future research. No GPS coordinates are simulated or claimed in this prototype.

---

## 2. Directory Structure

```
signalvex/
├── app.py                 # Streamlit web-based ground rescue dashboard
├── detector.py            # YOLO human detection module (COCO person class 0 filter)
├── tracker.py             # ByteTrack integration & track persistence with fallback
├── video_processor.py     # Frame extraction, bounding box annotation, and video writer
├── logger.py              # Record logging, operator review status, and CSV export
├── test_pipeline.py       # Automated verification test script
├── requirements.txt       # Project Python dependencies
├── README.md              # Documentation, setup guide, and presentation instructions
├── models/
│   └── yolov8n.pt         # Lightweight YOLOv8 nano model weights (~6.2 MB)
└── sample_data/
    └── sample_drone_flight.mp4  # Sample aerial drone video for demonstration
```

---

## 3. Quickstart & Installation (Windows)

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit) installed on Windows.
- Windows PowerShell or Command Prompt.

### Step-by-Step Setup

1. **Open PowerShell and navigate to the project directory:**
   ```powershell
   cd C:\Users\HP\.gemini\antigravity\scratch\signalvex
   ```

2. **Create a Python Virtual Environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled on your system, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

4. **Install Required Packages:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the Verification Test:**
   ```powershell
   python test_pipeline.py
   ```
   *Expected output:* Confirms model loading, tracking, detection logging, and video generation in ~5 seconds.

6. **Launch the SIGNALVEX Dashboard:**
   ```powershell
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 4. How to Demonstrate to Faculty (Demo Checklist)

1. **Launch Dashboard:** Run `streamlit run app.py`.
2. **Select Video Source:**
   - Under **"1. Video Source Selection"**, select **"Use Built-in Sample Aerial Drone Video"** (or upload any test `.mp4` file).
   - Point out that the system automatically inspects the video resolution (`960x540`), frame rate (`24 FPS`), and duration (`8.0s`).
3. **Configure Parameters:**
   - Show the left sidebar: Confidence threshold is set to `0.35`, and low-confidence review threshold is set to `0.50`.
   - Explain to faculty that this prevents weak false alarms while flagging borderline candidates in amber for human review.
4. **Execute Analysis:**
   - Click the blue **"🚀 Start SAR Analysis"** button.
   - Observe the live frame feed rendering bounding boxes, track IDs (`#1`, `#2`, `#3`), confidence percentages, and running counts in real-time.
5. **Explain Detection Summary:**
   - Review the metrics cards: **Total Detections**, **Approx. Distinct Tracks**, **Flagged for Review**, and **Measured FPS** (~13–15 FPS on CPU).
   - Explain that distinct tracks approximate individuals across time rather than raw repeated frame detections.
6. **Demonstrate Human Operator Review:**
   - Navigate to the **Detection Table**. Filter by **Track ID** or **Review Status**.
   - Under **"Update Operator Review Status"**, select a track (e.g., Track #1), change decision to **Keep** or **Reject**, and click **"Apply Decision to Track"**. Show that the table updates dynamically.
7. **Export Results:**
   - Click **"📥 Download Detection Log (CSV)"** to save the structured incident report.
   - Click **"📥 Download Annotated Video (MP4)"** to view the saved video file with overlaid bounding boxes.

---

## 5. Exported CSV Structure

The exported CSV file (`signalvex_detection_log.csv`) follows this standardized schema:

| Column Header | Data Type | Description |
| :--- | :--- | :--- |
| `frame_number` | Integer | Frame index in the video sequence (1, 2, 3...) |
| `timestamp_seconds` | Float | Timestamp within the video timeline in seconds |
| `track_id` | Integer | Persistent ByteTrack ID assigned to the person |
| `confidence` | Float | Detection confidence score (0.00 to 1.00) |
| `x1` | Integer | Top-left X pixel coordinate of bounding box |
| `y1` | Integer | Top-left Y pixel coordinate of bounding box |
| `x2` | Integer | Bottom-right X pixel coordinate of bounding box |
| `y2` | Integer | Bottom-right Y pixel coordinate of bounding box |
| `review_status` | String | Operator review label: `Keep`, `Reject`, or `Needs Review` |

---

## 6. Key Limitations & Future Work

In accordance with the project presentation:

| Aspect | Current Prototype Status | Future Scope (Hardware & Research) |
| :--- | :--- | :--- |
| **Sensing Modality** | Standard RGB video (visible light only) | Thermal (LWIR) camera fusion for night/smoke conditions |
| **Object Scope** | Human-like visual shapes only; does not diagnose injury or vitality | Audio/acoustic arrays and RF channel-state sensing |
| **Obstructions** | Cannot detect individuals obscured by rubble or heavy canopy | Ground-penetrating radar / specialized multi-spectral sensors |
| **Tracking** | ByteTrack reduces duplicate counts; ID switches can still occur | Re-identification (Re-ID) deep appearance embeddings |
| **Geolocation** | Image coordinates only `[x1, y1, x2, y2]`; no false GPS invented | Synchronized drone IMU & GPS telemetry with flat-ground projection |
| **Life Safety** | Operator decision support prototype; not certified for life safety | Formal ruggedization and SAR protocol certification |

---

## 7. Troubleshooting

- **Error: `ModuleNotFoundError: No module named 'ultralytics'`:**  
  Make sure you activated your virtual environment before running (`.\venv\Scripts\Activate.ps1`).
- **First-time Run Delay:**  
  On the very first launch, Ultralytics checks and caches the model weights. The included `models/yolov8n.pt` avoids downloading if present.
- **Port 8501 in use:**  
  Run Streamlit on an alternate port: `streamlit run app.py --server.port 8502`.
- **OpenCV Video Codec Warning on Windows:**  
  The system uses the built-in `mp4v` codec compatible with Windows standard Media Foundation.
