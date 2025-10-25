import cv2
from ultralytics import YOLO
import os
import glob
from tqdm import tqdm
 
# --- กำหนดค่าคงที่ส่วนกลาง ---
MODEL_PATH = r"models\yolo11m.pt"
CONFIDENCE_THRESHOLD = 0.85
TARGET_CLASS_NAME = 'elephant'
 
# กำหนดพาธหลัก
BASE_INPUT_PATH = r"D:\BU-ELEPHANT-HACKATHON-main\Day1_workshop_data"
BASE_OUTPUT_PATH = r"D:\BU-ELEPHANT-HACKATHON-main\dataset"
VIEW_NAMES = ['Back_view', 'Front_view', 'Left_side_view', 'Right_side_view']
image_extensions = ['jpg', 'jpeg', 'png']
 
# *** รายการชื่อช้างทั้งหมดที่ต้องการประมวลผล ***
ELEPHANT_NAMES = ['Ai_ko', 'Nam_Fon', 'Rat_sa_mee']
 
# --- การเตรียมการ: โหลดโมเดล (โหลดครั้งเดียว) ---
try:
    model = YOLO(MODEL_PATH)
    if TARGET_CLASS_NAME not in model.names.values():
        print(f"⚠️ คำเตือน: ชื่อคลาส '{TARGET_CLASS_NAME}' อาจไม่ถูกต้อง. คลาสที่มีในโมเดลคือ: {list(model.names.values())}")
except Exception as e:
    print(f"❌ ERROR: ไม่สามารถโหลดโมเดลจาก {MODEL_PATH} ได้. ตรวจสอบพาธของไฟล์: {e}")
    exit()
 
total_saved_images = 0
 
# ====================================================================
# *** ลูปใหญ่เพื่อวนซ้ำประมวลผลช้างแต่ละตัว ***
# ====================================================================
for ELEPHANT_NAME in ELEPHANT_NAMES:
    print("\n" + "#"*70)
    print(f"🐘 เริ่มประมวลผลข้อมูลสำหรับช้าง: {ELEPHANT_NAME}")
    print("#"*70)
   
    # 1. กำหนดพาธสำหรับช้างตัวปัจจุบัน
    OUTPUT_SAVE_PATH = os.path.join(BASE_OUTPUT_PATH, ELEPHANT_NAME)
    SOURCE_ROOT = os.path.join(BASE_INPUT_PATH, ELEPHANT_NAME)
 
    # สร้างรายการโฟลเดอร์ต้นฉบับโดยอัตโนมัติ
    SOURCE_FOLDERS = [os.path.join(SOURCE_ROOT, view) for view in VIEW_NAMES]
   
    # ตรวจสอบและสร้างโฟลเดอร์สำหรับบันทึก
    os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
    print(f"✅ โฟลเดอร์สำหรับบันทึก: {OUTPUT_SAVE_PATH}")
 
    save_count = 0 # นับจำนวนรูปที่เซฟสำหรับช้างตัวปัจจุบัน
 
    # --- ลูปประมวลผลรูปภาพของช้างแต่ละตัว ---
    for folder_path in SOURCE_FOLDERS:
        all_files = []
        for ext in image_extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
       
        if not all_files:
            print(f"⚠️ ไม่พบรูปภาพในโฟลเดอร์: {folder_path}")
            continue
       
        print(f"\n▶️ กำลังประมวลผลรูปภาพ {len(all_files)} ไฟล์จาก: {os.path.basename(folder_path)}")
       
        for image_path in tqdm(all_files, desc=f"{ELEPHANT_NAME}/{os.path.basename(folder_path)}", unit="img"):
            try:
                # 1. อ่านรูปภาพ
                frame = cv2.imread(image_path)
                if frame is None:
                    continue
 
                # 2. ทำการตรวจจับวัตถุ
                results = model(frame, verbose=False)[0]
 
                # 3. ค้นหา Bounding Box ของช้างที่ใหญ่ที่สุด
                largest_box = None
                max_area = 0
 
                for box in results.boxes:
                    conf = float(box.conf)
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
 
                    # กรอง: ต้องเป็นช้างและมีความเชื่อมั่นสูงพอ
                    if label != TARGET_CLASS_NAME or conf < CONFIDENCE_THRESHOLD:
                        continue
 
                    # ดึงพิกัด
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                   
                    # คำนวณพื้นที่ (Area) ของ Bounding Box
                    width = x2 - x1
                    height = y2 - y1
                    current_area = width * height
 
                    # ตรวจสอบว่า Bounding Box ปัจจุบันใหญ่กว่าที่เคยพบหรือไม่
                    if current_area > max_area and width > 0 and height > 0:
                        max_area = current_area
                        largest_box = (x1, y1, x2, y2)
 
 
                # 4. ครอปและบันทึกรูปภาพช้างที่ใหญ่ที่สุด (ถ้าพบ)
                if largest_box is not None:
                    x1, y1, x2, y2 = largest_box
                   
                    crop = frame[y1:y2, x1:x2]
 
                    if crop.size > 0:
                        save_count += 1
                        total_saved_images += 1
                        # ใช้ ELEPHANT_NAME ในชื่อไฟล์เพื่อระบุแหล่งที่มาได้ง่ายขึ้น
                        # และใช้ save_count สำหรับการนับในช้างแต่ละตัว
                        filename = os.path.join(OUTPUT_SAVE_PATH, f"{ELEPHANT_NAME}_{save_count:05d}.jpg")
                        cv2.imwrite(filename, crop)
                   
            except Exception as e:
                #print(f"❌ เกิดข้อผิดพลาดขณะประมวลผล {os.path.basename(image_path)}: {e}")
                pass
 
    # สรุปผลสำหรับช้างตัวปัจจุบัน
    print(f"\n✅ ประมวลผล {ELEPHANT_NAME} เสร็จสมบูรณ์! บันทึก {save_count} รูป")
 
# --- สรุปผลรวม ---
print("\n" + "="*70)
print("🎉 **การประมวลผลทั้งหมดเสร็จสมบูรณ์!**")
print(f"✅ บันทึกรูปภาพที่ครอปแล้วรวม **{total_saved_images}** รูป ไปยังโฟลเดอร์ย่อยใน: {BASE_OUTPUT_PATH}")
print("="*70)