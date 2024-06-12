import cv2
import time
from Gangway_opencv import VideoProcessor
from Gangway_yolo import YOLOProcessor


def is_within(person_box, blockage_box):
    x1_p, y1_p, x2_p, y2_p = person_box
    x1_b, y1_b, x2_b, y2_b = blockage_box
    return x1_b >= x1_p and y1_b >= y1_p or x2_b <= x2_p and y2_b <= y2_p


def main():
    vp = VideoProcessor()
    yp = YOLOProcessor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        if vp.blockage_detected:
            while True:
                if vp.blockage_image is not None:
                    frame = vp.blockage_image.copy()
                    for blockage in vp.detected_blockages:
                        x1, y1, x2, y2 = blockage
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Blockage", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.putText(frame, 'Blockage detected', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Combined Detection", frame)

                key = cv2.waitKey(0) & 0xFF
                if key == ord('\r'):
                    print("Resuming video...")
                    time.sleep(2)
                    vp.blockage_detected = False
                    vp.reference_frame = None
                    vp.blockage_list.clear()
                    vp.object_start_times.clear()
                    vp.start_new_video_capture()
                    break
                elif key == ord('q'):
                    break
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame_vp = vp.process_frame(frame.copy())
        processed_frame_yp = yp.process_frame(frame.copy())

        blockages = vp.detected_blockages
        persons = yp.detected_persons

        valid_blockages = []
        debug_blockages = []

        for blockage in blockages:
            if not any(is_within(person, blockage) for person in persons):
                valid_blockages.append(blockage)
            else:
                debug_blockages.append(blockage)

        for person in persons:
            x1, y1, x2, y2 = person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Person: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        for blockage in valid_blockages:
            x1, y1, x2, y2 = blockage
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Blockage", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Blockage: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        for blockage in debug_blockages:
            x1, y1, x2, y2 = blockage
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Person_Blockage", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Person_Blockage: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        cv2.imshow("Combined Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
