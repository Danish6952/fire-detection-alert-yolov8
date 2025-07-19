import cv2
import time
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
from datetime import datetime

# === 📦 Model Configuration ===
MODEL_PATH = "best.pt"  # Replace with your YOLOv8 .pt file
CONF_THRESHOLD = 0.6  # Strict detection confidence
DETECTION_CLASSES = ['fire', 'smoke', 'spark', 'flame']
EMAIL_COOLDOWN = 300  # 5 minutes cooldown

# === 📧 Your Email Settings ===
smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'rajadanish52286@gmail.com'
smtp_password = 'pkgx tkem rkjd gvuf'
recipient_email = 'da2662475@gmail.com'

last_email_time = 0  # For cooldown tracking

# === 🔥 Load YOLOv8 Model ===
model = YOLO(MODEL_PATH)

# === ✉️ Email Alert Function ===
def send_email_alert(image_path, label):
    global last_email_time
    if time.time() - last_email_time < EMAIL_COOLDOWN:
        return  # Still cooling down

    msg = EmailMessage()
    msg['Subject'] = f'🔥 {label.upper()} Detected!'
    msg['From'] = smtp_username
    msg['To'] = recipient_email
    msg.set_content(f'Alert: {label.upper()} detected. See attached image.')

    with open(image_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename='alert.jpg')

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        print(f"✅ Email sent for {label}")
        last_email_time = time.time()
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# === 🎥 Start Webcam Feed ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Inference
    results = model(frame, verbose=False)[0]

    alert_triggered = False
    alert_label = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in DETECTION_CLASSES and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            alert_triggered = True
            alert_label = label

    # Send alert email if any fire-like detection found
    if alert_triggered:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_path = f"alert_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)
        send_email_alert(img_path, alert_label)

    cv2.imshow("🔥 YOLOv8 Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
