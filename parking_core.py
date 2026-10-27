"""Core functions for vehicle tracking and parking-slot occupancy."""

import json
from pathlib import Path

import cv2
import numpy as np

from config import SLOT_OVERLAP_THRESHOLD


def load_slots(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Slot configuration '{path.name}' was not found. "
            "Copy slots.example.json to slots.json and adjust the coordinates."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = data.get("slots", [])
    if not slots:
        raise ValueError("The slot configuration has no slots.")
    for slot in slots:
        if not slot.get("id") or len(slot.get("points", [])) < 3:
            raise ValueError("Each slot needs an id and at least 3 polygon points.")
    return slots, data.get("image_width"), data.get("image_height")


def point_inside(point, polygon):
    contour = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (int(point[0]), int(point[1])), False) >= 0


def slot_area(points):
    return max(0.0, float(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def intersection_over_slot(box, points):
    x1, y1, x2, y2 = box
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    poly = np.asarray(points, dtype=np.float32)
    area = slot_area(points)
    if area <= 0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(rect, poly)
        return max(0.0, float(intersection)) / area
    except cv2.error:
        return 0.0


def analyze_slots(slots, tracked_vehicles):
    """Return slot statuses and IDs of tracked vehicles overlapping each slot."""
    output = []
    for slot in slots:
        occupant_ids = []
        for vehicle in tracked_vehicles:
            box = vehicle["box"]
            x1, y1, x2, y2 = box
            bottom_center = ((x1 + x2) // 2, y2)
            inside = point_inside(bottom_center, slot["points"])
            overlap = intersection_over_slot(box, slot["points"])
            if inside or overlap >= SLOT_OVERLAP_THRESHOLD:
                occupant_ids.append(vehicle["track_id"] if vehicle["track_id"] is not None else "Untracked")
        occupant_ids = list(dict.fromkeys(occupant_ids))
        output.append({
            "slot_id": slot["id"],
            "status": "Occupied" if occupant_ids else "Available",
            "vehicle_ids": occupant_ids,
        })
    return output


def annotate_frame(frame, slots, tracked_vehicles, slot_statuses):
    annotated = frame.copy()
    status_by_id = {item["slot_id"]: item for item in slot_statuses}

    for slot in slots:
        points = np.asarray(slot["points"], dtype=np.int32)
        status = status_by_id[slot["id"]]["status"]
        # OpenCV BGR: red for occupied, green for available
        color = (0, 0, 230) if status == "Occupied" else (0, 190, 0)
        cv2.polylines(annotated, [points], True, color, 2)
        x, y = points[0]
        ids = status_by_id[slot["id"]]["vehicle_ids"]
        label = f"{slot['id']} - {status}"
        if ids:
            label += f" ({','.join(map(str, ids))})"
        cv2.putText(
            annotated, label, (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = vehicle["box"]
        track_id = vehicle["track_id"]
        label = f"{vehicle['label']} ID:{track_id if track_id is not None else '?'} {vehicle['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 130, 0), 2)
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 130, 0), 2, cv2.LINE_AA
        )

    total = len(slot_statuses)
    occupied = sum(s["status"] == "Occupied" for s in slot_statuses)
    available = total - occupied
    banner = f"Slots: {total} | Occupied: {occupied} | Available: {available} | Tracked vehicles: {len(tracked_vehicles)}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(
        annotated, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255, 255, 255), 2, cv2.LINE_AA
    )
    return annotated
