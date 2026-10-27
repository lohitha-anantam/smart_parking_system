# Smart Parking System with Object Tracking and Dashboard

A Python project that tracks vehicles across a parking-lot video using Ultralytics YOLO tracking (ByteTrack), estimates configured parking-slot occupancy, and displays available/occupied slot counts in a Streamlit dashboard.

## Features

- YOLO vehicle detection with persistent track IDs using ByteTrack.
- Configurable parking-slot polygons.
- Available vs. occupied slot status and vehicle IDs assigned to slots.
- Dashboard metrics: total slots, occupied slots in the final frame, available slots in the final frame, unique track IDs, and processed frames.
- Occupancy-over-time line chart.
- Annotated tracking video and JSON report downloads.

## Project structure

```text
smart_parking_tracking_dashboard/
├── app.py
├── parking_core.py
├── config.py
├── slots.example.json
├── requirements.txt
├── .gitignore
├── input/
├── output/
└── README.md
```

## Run in VS Code on Windows

### 1. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure slot coordinates

The example coordinates are illustrative. Copy `slots.example.json` to `slots.json`, then change each polygon's points to match the slots in your video.

The JSON can include `image_width` and `image_height` to describe the coordinate system in which points were drawn. The app scales coordinates if the video dimensions differ. If you upload a custom JSON through the dashboard, include these dimensions for correct scaling.

### 4. Start the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a parking-lot video, optionally upload your slot JSON, adjust the model/confidence settings, and click **Analyze video**.

The first run may download `yolov8n.pt` automatically. A short video is recommended for the first test.

## How tracking and slot availability work

1. YOLO detects vehicles in each video frame.
2. ByteTrack associates detections between frames and assigns track IDs when possible.
3. Each vehicle box is checked against manually configured slot polygons.
4. A slot is marked occupied if a vehicle's bottom-center point is inside it or its bounding box overlaps enough of the slot polygon.
5. Streamlit displays the final-frame slot status and an occupancy-over-time chart.

## Limitations

- Slot polygons must be calibrated for the camera view; the included sample points are only an example.
- This is a video-upload workflow, not a live CCTV or IP-camera integration.
- Track IDs can change or be missed when vehicles are occluded or leave the frame.
- Slot occupancy is a geometric heuristic using bounding boxes, so perspective and overlapping vehicles can cause mistakes.
- The dashboard's "available" count is based on the last processed frame, not a booking system or guaranteed real-time availability.
- Detection quality depends on image resolution, lighting, camera angle, and model performance.

## GitHub safety

Do not commit private CCTV footage, identifiable license plates without permission, model weights, virtual environments, or generated videos. `.gitignore` excludes local input/output files and model weights.
