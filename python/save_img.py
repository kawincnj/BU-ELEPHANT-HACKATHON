import cv2
from ultralytics import YOLO

model = YOLO(r"model\elephant_obj_detect.pt")

save_path = r"D:\CODING\BU-ELEPHANT-HACKATHON\Elephant_Image\NAMFON"

cap = cv2.VideoCapture(r"Videos\น้ำฝน.MP4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 0.7
save_count = 0
frame_interval = 5

frame_index = 0
while True:
    ret, frame = cap.read()
    fram_cop = frame.copy()
    if not ret:
        print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    frame_index += 1
    results = model(frame)[0]

    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_index % frame_interval == 0:
            crop = fram_cop[y1:y2, x1:x2]
            if crop.size > 0:
                save_count += 1
                filename = fr"{save_path}\NAMFON_{save_count:04d}.jpg"
                cv2.imwrite(filename, crop)
                print(f"💾 Saved: {filename}")

    cv2.imshow("Custom YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {save_count} cropped images to: {save_path}")
