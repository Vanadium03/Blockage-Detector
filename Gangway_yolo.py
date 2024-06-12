import cv2
from ultralytics import YOLO


class YOLOProcessor:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.detected_persons = []

    def process_frame(self, frame):
        results = self.model.predict(source=frame, imgsz=640, conf=0.7)
        person_boxes = []

        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # 'person' class id is 0 in COCO dataset
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    confidence = box.conf.item()
                    label = f"Person: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    person_boxes.append((x1, y1, x2, y2))

        self.detected_persons = person_boxes
        return frame


if __name__ == "__main__":
    processor = YOLOProcessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = processor.process_frame(frame)
        cv2.imshow("YOLOv8 Person Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
