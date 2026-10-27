"""Streamlit dashboard for tracking vehicles and monitoring parking slots."""

import json
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    OUTPUT_DIR,
    SAMPLE_SLOTS_FILE,
    VEHICLE_CLASS_IDS,
)
from parking_core import load_slots, analyze_slots, annotate_frame


st.set_page_config(
    page_title="Smart Parking Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Smart Parking System")
st.caption("YOLO object tracking · Parking-slot availability · Occupancy dashboard")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("YOLO model", value=DEFAULT_MODEL, help="Example: yolov8n.pt")
    confidence = st.slider("Detection confidence", 0.10, 0.90, DEFAULT_CONFIDENCE, 0.05)
    uploaded_slots = st.file_uploader("Optional slot configuration JSON", type=["json"])
    st.markdown(
        "**Slot coordinates:** Each slot must be a polygon in the video frame's "
        "pixel coordinates. Use `slots.example.json` as a template and adjust it "
        "for your camera view."
    )

video_file = st.file_uploader(
    "Upload a parking-lot video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Vehicle tracking runs across video frames. A shorter clip is recommended for the first test.",
)

if uploaded_slots is not None:
    try:
        slots_data = json.loads(uploaded_slots.getvalue().decode("utf-8"))
        slots = slots_data.get("slots", [])
        if not slots:
            st.error("The uploaded JSON does not contain any parking slots.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read slot JSON: {exc}")
        st.stop()
else:
    try:
        slots, _, _ = load_slots(SAMPLE_SLOTS_FILE)
    except Exception:
        slots = []

if not slots:
    st.warning("No parking slots configured. Upload a slot JSON file to continue.")
    st.stop()

if video_file is None:
    st.info("Upload a parking-lot video to start tracking. The sample slot coordinates are illustrative and must be adjusted for your camera.")
    st.subheader("Configured parking slots")
    st.dataframe(pd.DataFrame([
        {"Slot": slot["id"], "Polygon points": str(slot["points"])}
        for slot in slots
    ]), use_container_width=True, hide_index=True)
    st.stop()

if st.button("▶ Analyze video", type="primary"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0, text="Preparing video...")
    status_text = st.empty()
    preview = st.empty()

    temp_input = None
    temp_output = None

    try:
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(video_file.getvalue())
            temp_input = temp.name

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video. Try MP4 encoded with H.264.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid frame dimensions.")

        # Rescale configured slot points if the sample JSON's reference size differs.
        source_width = None
        source_height = None
        if uploaded_slots is not None:
            source_width = slots_data.get("image_width")
            source_height = slots_data.get("image_height")
        else:
            sample_data = json.loads(SAMPLE_SLOTS_FILE.read_text(encoding="utf-8"))
            source_width = sample_data.get("image_width")
            source_height = sample_data.get("image_height")

        active_slots = json.loads(json.dumps(slots))
        if source_width and source_height and (source_width != width or source_height != height):
            sx, sy = width / source_width, height / source_height
            for slot in active_slots:
                slot["points"] = [
                    [round(p[0] * sx), round(p[1] * sy)] for p in slot["points"]
                ]

        out_path = OUTPUT_DIR / "tracked_parking_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Could not create output video on this system.")

        model = YOLO(model_name)
        slot_rows = []
        seen_track_ids = set()
        processed_frames = 0
        last_vehicles = []
        last_statuses = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                classes=VEHICLE_CLASS_IDS,
                verbose=False,
            )
            detections = []
            if results and results[0].boxes is not None:
                result = results[0]
                names = result.names
                for box in result.boxes:
                    coords = tuple(map(int, box.xyxy[0].tolist()))
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                        seen_track_ids.add(track_id)
                    detections.append({
                        "box": coords,
                        "track_id": track_id,
                        "label": names.get(cls_id, str(cls_id)),
                        "confidence": conf,
                    })

            statuses = analyze_slots(active_slots, detections)
            annotated = annotate_frame(frame, active_slots, detections, statuses)
            writer.write(annotated)

            for item in statuses:
                slot_rows.append({
                    "frame": processed_frames + 1,
                    "slot_id": item["slot_id"],
                    "status": item["status"],
                    "vehicle_ids": ", ".join(map(str, item["vehicle_ids"])),
                })

            last_vehicles = detections
            last_statuses = statuses
            processed_frames += 1

            if frame_count > 0:
                progress.progress(
                    min(processed_frames / frame_count, 1.0),
                    text=f"Processed {processed_frames}/{frame_count} frames",
                )
            elif processed_frames % 10 == 0:
                status_text.info(f"Processed {processed_frames} frames...")

            if processed_frames == 1 or processed_frames % max(1, int(fps)) == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live processing preview")

        cap.release()
        writer.release()
        progress.progress(1.0, text=f"Finished: {processed_frames} frames processed")

        if processed_frames == 0:
            raise ValueError("No frames could be read from the uploaded video.")

        summary = {
            "Total configured slots": len(active_slots),
            "Occupied in final frame": sum(x["status"] == "Occupied" for x in last_statuses),
            "Available in final frame": sum(x["status"] == "Available" for x in last_statuses),
            "Unique tracked IDs seen": len(seen_track_ids),
            "Frames processed": processed_frames,
        }

        st.session_state["parking_summary"] = summary
        st.session_state["parking_statuses"] = last_statuses
        st.session_state["parking_history"] = pd.DataFrame(slot_rows)
        st.session_state["parking_output"] = str(out_path)
        st.session_state["parking_slots"] = active_slots
        st.success("Video analysis completed.")

    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
    finally:
        if temp_input and Path(temp_input).exists():
            Path(temp_input).unlink(missing_ok=True)

if "parking_summary" in st.session_state:
    st.divider()
    st.header("Parking dashboard")
    summary = st.session_state["parking_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total slots", summary["Total configured slots"])
    c2.metric("Occupied (last frame)", summary["Occupied in final frame"])
    c3.metric("Available (last frame)", summary["Available in final frame"])
    c4.metric("Unique tracked IDs", summary["Unique tracked IDs seen"])

    st.subheader("Current slot status (last processed frame)")
    status_df = pd.DataFrame([
        {
            "Slot": item["slot_id"],
            "Status": item["status"],
            "Tracked vehicle IDs": ", ".join(map(str, item["vehicle_ids"])) or "—",
        }
        for item in st.session_state["parking_statuses"]
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    history = st.session_state["parking_history"]
    if not history.empty:
        st.subheader("Occupancy over time")
        chart_data = (
            history.assign(occupied=history["status"].eq("Occupied").astype(int))
            .groupby("frame", as_index=False)["occupied"].sum()
            .rename(columns={"occupied": "Occupied slots"})
        )
        st.line_chart(chart_data.set_index("frame"))

    output_path = Path(st.session_state["parking_output"])
    if output_path.exists():
        st.subheader("Download annotated tracking video")
        st.video(str(output_path))
        st.download_button(
            "Download processed video",
            data=output_path.read_bytes(),
            file_name="tracked_parking_output.mp4",
            mime="video/mp4",
        )

    report = {
        "summary": summary,
        "final_slot_status": st.session_state["parking_statuses"],
        "frames": st.session_state["parking_history"].to_dict(orient="records"),
    }
    st.download_button(
        "Download occupancy report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="parking_report.json",
        mime="application/json",
    )
