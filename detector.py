from ultralytics import YOLO

# Load trained marine debris detection model
model = YOLO("model/final_detector.pt")


def detect(image):
    # Run object detection with a lower confidence threshold
    results = model(image, conf=0.10)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return detections