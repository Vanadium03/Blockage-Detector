import cv2
import numpy as np
import time


class VideoProcessor:
    def __init__(self, camera_index=0, fps=24):
        self.camera_index = camera_index
        self.vid = cv2.VideoCapture(self.camera_index)
        self.reference_frame = None
        self.object_start_times = {}
        self.no_change_start_time = None
        self.blockage_image = None
        self.blockage_detected = False
        self.fps = fps
        self.consecutive_frames = 5 * self.fps
        self.blockage_list = []
        self.new_video_capture = False
        self.new_vid = None
        # Adjusted kernel size for opening
        self.kernel_open = np.ones((5, 5), np.uint8)
        # Adjusted kernel size for closing
        self.kernel_close = np.ones((10, 10), np.uint8)
        self.detected_blockages = []

    def process_frame(self, frame):
        if self.reference_frame is None:
            self.reference_frame = frame.copy()
            return frame

        object_diff = cv2.absdiff(frame, self.reference_frame)
        object_gray = cv2.cvtColor(object_diff, cv2.COLOR_BGR2GRAY)
        _, object_thresh = cv2.threshold(
            object_gray, 50, 255, cv2.THRESH_BINARY)
        object_thresh = cv2.morphologyEx(
            object_thresh, cv2.MORPH_OPEN, self.kernel_open)
        object_thresh = cv2.morphologyEx(
            object_thresh, cv2.MORPH_CLOSE, self.kernel_close)

        if np.count_nonzero(object_thresh) == 0:
            if self.no_change_start_time is None:
                self.no_change_start_time = time.time()
            elif time.time() - self.no_change_start_time >= 4:
                self.reference_frame = frame.copy()
                self.no_change_start_time = None
                self.object_start_times.clear()
                self.blockage_list.clear()
        else:
            self.no_change_start_time = None

        contours, _ = cv2.findContours(
            object_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        current_time = time.time()
        detected_objects = []

        for cnt in contours:
            if 5000 <= cv2.contourArea(cnt) < 200000:
                hull = cv2.convexHull(cnt)
                x, y, w, h = cv2.boundingRect(hull)
                detected_objects.append((x, y, x + w, y + h))

                # Draw the bounding box only if not in frozen frame state
                if not self.blockage_detected:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 2)

        self.update_object_times(detected_objects, current_time, frame)
        self.detected_blockages = detected_objects

        if np.count_nonzero(object_thresh) > 0:
            # Update the block detection flag only if a blockage is detected
            if len(detected_objects) > 0:
                if self.blockage_list.count(True) >= self.consecutive_frames:
                    self.blockage_detected = True
                    cv2.putText(frame, 'Blockage detected', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    self.blockage_image = frame.copy()  # Save the current frame as blockage image

        return frame  # Return the original frame with bounding boxes

    def update_object_times(self, detected_objects, current_time, frame):
        new_object_start_times = {}

        for obj in detected_objects:
            if obj in self.object_start_times:
                start_time = self.object_start_times[obj]
                if current_time - start_time > 5:
                    self.blockage_image = frame.copy()
                    self.blockage_detected = True
                    cv2.putText(self.blockage_image, 'Blockage detected', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imwrite('blockage.jpg', self.blockage_image)
                    print("Blockage detected, image saved as blockage.jpg")
            else:
                start_time = current_time

            new_object_start_times[obj] = start_time

        self.object_start_times = new_object_start_times

        self.blockage_list.append(len(detected_objects) > 0)

        if len(self.blockage_list) > self.consecutive_frames:
            self.blockage_list.pop(0)

        if self.blockage_list.count(True) >= self.consecutive_frames:
            self.blockage_detected = True
            cv2.putText(frame, 'Blockage detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite('blockage.jpg', frame)
            print("Blockage detected! Frame frozen.")

    def overlay_blockage_image(self, frame):
        if self.blockage_image is not None:
            blockage_resized = cv2.resize(self.blockage_image, (150, 150))
            bh, bw, _ = blockage_resized.shape
            frame[0:bh, 0:bw] = blockage_resized
        return frame

    def start_new_video_capture(self):
        self.new_video_capture = True
        self.new_vid = cv2.VideoCapture(self.camera_index)
        ret, frame = self.new_vid.read()
        if ret:
            self.reference_frame = frame.copy()
            print("New reference frame set")
        else:
            print("Failed to capture new frame")

    def run(self):
        while True:
            if self.blockage_detected:
                while True:
                    ret, frame = self.vid.read()
                    if not ret:
                        print("Failed to capture frame")
                        break

                    cv2.putText(frame, 'Blockage detected', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Webcam Live Feed', frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('\r'):
                        print("Resuming video...")
                        time.sleep(2)
                        self.blockage_detected = False
                        self.reference_frame = None
                        self.blockage_list.clear()
                        self.object_start_times.clear()
                        self.start_new_video_capture()
                        break
                    elif key == ord('q'):
                        break
                continue

            if self.new_video_capture:
                ret, frame = self.new_vid.read()
                if not ret:
                    print("Failed to capture new frame from new video")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Webcam Live Feed', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            else:
                ret, frame = self.vid.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Webcam Live Feed', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        self.vid.release()
        if self.new_video_capture:
            self.new_vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()
