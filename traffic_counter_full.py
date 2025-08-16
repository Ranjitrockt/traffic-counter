import yt_dlp
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv

# 1️⃣ Video Download
url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
VIDEO_PATH = "traffic-video.mp4"

ydl_opts = {'outtmpl': VIDEO_PATH, 'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
print(f"✅ Video downloaded: {VIDEO_PATH}")

# 2️⃣ Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3️⃣ Output video writer
out = cv2.VideoWriter('traffic_output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# 4️⃣ Lane coordinates
lane1_x = (0, width // 3)
lane2_x = (width // 3 + 1, 2 * width // 3)
lane3_x = (2 * width // 3 + 1, width)

# 5️⃣ YOLOv8 model
model = YOLO('yolov8n')  # Model will auto-download

# 6️⃣ DeepSORT tracker
tracker = DeepSort(max_age=30)

# 7️⃣ CSV file
csv_file = open("vehicle_count.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["VehicleID", "Lane", "Frame", "Timestamp"])

frame_count = 0
lane_counts = {"lane1": 0, "lane2": 0, "lane3": 0}
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    timestamp = frame_count / fps

    # Vehicle detection
    results = model(frame)
    detections = []
for r in results:
    boxes = r.boxes.xyxy
    confs = r.boxes.conf
    classes = r.boxes.cls
    for box, conf, cls in zip(boxes, confs, classes):
        if int(cls) in [2, 3, 5, 7]:  # COCO vehicle classes
            x1, y1, x2, y2 = box.tolist()
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
   
    # Tracking
    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)
    else:
        tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2

        # Determine lane
        if lane1_x[0] <= cx <= lane1_x[1]:
            lane = "lane1"
        elif lane2_x[0] <= cx <= lane2_x[1]:
            lane = "lane2"
        else:
            lane = "lane3"

        # ✅ Unique counting
        if tid not in counted_ids:
            counted_ids.add(tid)
            lane_counts[lane] += 1

        # Draw box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Write CSV
        csv_writer.writerow([tid, lane, frame_count, timestamp])

    # Draw lanes & counts
    cv2.line(frame, (lane1_x[1], 0), (lane1_x[1], height), (0, 255, 0), 2)
    cv2.line(frame, (lane2_x[1], 0), (lane2_x[1], height), (0, 255, 0), 2)
    cv2.putText(frame, f"Lane1:{lane_counts['lane1']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Lane2:{lane_counts['lane2']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Lane3:{lane_counts['lane3']}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to stop
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("✅ Traffic counting complete! Output video and CSV saved.")