import os
import glob
import csv
import cv2
import numpy as np
from memory_tracker import YoloMemoryTracker


def sanitize_filename_part(value):
    """
    Make a string safe for use in filenames.
    Keeps letters, numbers, hyphen, and underscore.
    """
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "frame"


def main():
    # 1. Configuration & Directories
    input_dir = '../images'
    output_dir = 'output_frames'
    video_output_path = 'output_tracking1.mp4'
    gallery_dir = 'memory_gallery'
    crop_log_path = os.path.join(gallery_dir, 'crop_log.csv')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    # Optional:
    # Set max_crops_per_car = 3 if you want to save only 3 crops per vehicle ID.
    # Set max_crops_per_car = None if you want to save every crop.
    max_crops_per_car = None

    # 2. Get sorted frame paths
    frame_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png')),
        key=lambda x: os.path.basename(x)
    )

    if not frame_paths:
        print(f"[Error] No image frames found in directory '{input_dir}'!")
        return

    print(f"[Main] Found {len(frame_paths)} frames in '{input_dir}'.")

    # 3. Initialize YOLO + Memory Bank Tracker
    # yolo_model options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8x.pt'
    tracker = YoloMemoryTracker(
        yolo_model='../yolov8n.pt',
        strict_reid_threshold=0.65,
        relaxed_reid_threshold=0.45,
        conf_threshold=0.35
    )

    # Track centroid history for drawing motion trails
    # trajectory_history: { track_id: [(cx, cy), ...] }
    trajectory_history = {}

    # Setup Video Writer
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"[Error] Could not read first frame: {frame_paths[0]}")
        return

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 10.0, (w, h))

    if not video_writer.isOpened():
        print(f"[Warning] Could not open video writer for: {video_output_path}")

    # Initialize crop log CSV
    if not os.path.exists(crop_log_path) or os.path.getsize(crop_log_path) == 0:
        with open(crop_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['vehicle_id', 'frame_idx', 'frame_name', 'crop_filename'])

    print("\n[Main] Starting processing...")
    print("=" * 60)

    reid_events = []
    saved_crops_count = {}

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[Warning] Could not read frame: {frame_path}")
            continue

        filename = os.path.basename(frame_path)
        frame_idx = idx + 1

        # Frame name without extension
        # Example: "frame_0001.jpg" -> "frame_0001"
        frame_stem = os.path.splitext(filename)[0]
        safe_frame_stem = sanitize_filename_part(frame_stem)

        # Process frame through Tracker + Memory Bank
        results = tracker.process_frame(frame, frame_idx)

        # Annotate Frame
        annotated_frame = frame.copy()

        for res in results:
            trk_id = res['id']
            x1, y1, x2, y2 = res['bbox']
            conf = res['conf']
            reid_score = res['reid_score']
            is_reid = res['is_reid']
            color = res['color']  # (B, G, R)

            # Initialize crop counter for this vehicle ID
            if trk_id not in saved_crops_count:
                saved_crops_count[trk_id] = 0

            # Decide whether this crop should be saved
            should_save_crop = (
                max_crops_per_car is None or
                saved_crops_count[trk_id] < max_crops_per_car
            )

            # === SAVE CAR CROPS WITH VEHICLE ID + FRAME NAME ===
            if should_save_crop:
                car_crop = frame[y1:y2, x1:x2]

                if car_crop.size > 0:
                    view_idx = saved_crops_count[trk_id]

                    crop_filename = (
                        f"car_id_{trk_id:03d}_"
                        f"frame_{frame_idx:04d}_"
                        f"source_{safe_frame_stem}_"
                        f"view_{view_idx:03d}.jpg"
                    )

                    crop_path = os.path.join(gallery_dir, crop_filename)
                    cv2.imwrite(crop_path, car_crop)

                    # Save the relationship in CSV:
                    # vehicle_id, frame_idx, frame_name, crop_filename
                    with open(crop_log_path, 'a', newline='', encoding='utf-8') as log_file:
                        writer = csv.writer(log_file)
                        writer.writerow([
                            trk_id,
                            frame_idx,
                            filename,
                            crop_filename
                        ])

                    saved_crops_count[trk_id] += 1

            # Calculate centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Optional trajectory history
            # if trk_id not in trajectory_history:
            #     trajectory_history[trk_id] = []
            # trajectory_history[trk_id].append((cx, cy))
            # if len(trajectory_history[trk_id]) > 30:
            #     trajectory_history[trk_id].pop(0)

            # Draw Bounding Box
            thickness = 3 if is_reid else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare Label Text
            if is_reid:
                label = f"ID:{trk_id} [RE-ID {reid_score:.2f}]"
                reid_events.append(
                    f"Frame {frame_idx:03d} ({filename}): "
                    f"Car ID {trk_id} RE-IDENTIFIED from Memory Bank! "
                    f"(Score: {reid_score:.2f})"
                )
            else:
                label = f"Car ID:{trk_id} ({conf:.2f})"

            # Draw Label Background Box
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            # Draw Label Text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Draw Frame Info overlay
        info_text = (
            f"Frame: {frame_idx}/{len(frame_paths)} | "
            f"Active Cars: {len(results)} | "
            f"Memory Gallery IDs: {len(tracker.memory_bank.gallery)}"
        )

        cv2.putText(
            annotated_frame,
            info_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Save output annotated frame
        out_img_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_img_path, annotated_frame)

        # Write frame to video
        video_writer.write(annotated_frame)

        print(
            f"[{frame_idx:03d}/{len(frame_paths):03d}] "
            f"Processed {filename} -> "
            f"Tracked {len(results)} vehicles "
            f"(Memory Bank Size: {len(tracker.memory_bank.gallery)})"
        )

    video_writer.release()

    # Export / summary for gallery
    print(f"\n[Gallery] Saved car crops and crop log to '{gallery_dir}/'")

    for trk_id in tracker.memory_bank.gallery.keys():
        print(f"  -> Car ID {trk_id}: Memory Bank contains multi-view keyframe embeddings.")

    # Create simple HTML viewer
    html_content = """
<html>
<head>
    <meta charset="utf-8">
    <title>Vehicle Tracking Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 30px;
            background: #111;
            color: #eee;
        }
        h1 {
            color: #00ff88;
        }
        code {
            background: #222;
            padding: 4px 8px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <h1>Vehicle Tracking Results</h1>
    <p>Annotated frames are saved in: <code>output_frames/</code></p>
    <p>Vehicle crops are saved in: <code>memory_gallery/</code></p>
    <p>Crop log with vehicle ID and frame name is saved in: <code>memory_gallery/crop_log.csv</code></p>
    <p>Output video is saved as: <code>output_tracking1.mp4</code></p>
</body>
</html>
"""

    with open('viewer.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("[Viewer] Created interactive HTML viewer at 'viewer.html'")

    print("=" * 60)
    print("[Main] Processing Completed Successfully!")
    print(f"[Main] Output annotated frames saved to: '{output_dir}/'")
    print(f"[Main] Output video saved to: '{video_output_path}'")
    print(f"[Main] Crop gallery saved to: '{gallery_dir}/'")
    print(f"[Main] Crop log saved to: '{crop_log_path}'")
    print(f"[Main] Interactive HTML Viewer: 'viewer.html'")
    print(f"[Main] Total Unique Vehicles Tracked: {tracker.next_track_id - 1}")

    if reid_events:
        print(f"\n[Re-ID Events Summary - {len(reid_events)} Events]:")
        for evt in reid_events:
            print(f"  -> {evt}")
    else:
        print("\n[Re-ID Events Summary]: All tracks maintained continuous or were distinct objects.")


if __name__ == '__main__':
    main()