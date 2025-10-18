import cv2
import os
from ultralytics import YOLO

# โหลดโมเดลตรวจจับช้าง
model = YOLO(r"D:\Code\BU-ELEPHENT-HACKATHON\model\elephant_obj_detect.pt")

# เปิดวิดีโอ
cap = cv2.VideoCapture(r"\Videos\rus.MP4")


# โฟลเดอร์เก็บภาพ
save_dir = os.path.join("elephant_img", "rus")
os.makedirs(save_dir, exist_ok=True)

# ค่าความมั่นใจขั้นต่ำ
CONFIDENCE_THRESHOLD = 0.85

# ตัวนับจำนวนภาพที่บันทึก
img_count = 0
max_images = 3000

# เว้นเฟรม (บันทึกช้า)
frame_skip = 1
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or img_count >= max_images:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # รัน YOLO ตรวจจับ
    results = model(frame)[0]

    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # ตรวจว่าคลาสคือ "elephant"
        if label.lower() == "elephant":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # วาดกรอบรอบช้าง
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # บันทึกเฉพาะกรอบช้าง
            filename = os.path.join(save_dir, f"Rus_{img_count:04d}.jpg")
            cv2.imwrite(filename, crop)
            img_count += 1

            print(f"✅ บันทึกภาพที่ {img_count} : {filename}")

            if img_count >= max_images:
                break

    # แสดงวิดีโอแบบเรียลไทม์
    cv2.imshow("Elephant Detection (Saving...)", frame)

    # กด Q เพื่อหยุดก่อนครบ 3000 ภาพ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📸 บันทึกครบ {img_count} ภาพแล้วในโฟลเดอร์: {save_dir}")
