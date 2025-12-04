import cv2
import numpy as np
from pathlib import Path

class ImageRecognizer:
    """Detect and outline a specific object in webcam frames using SIFT."""

    def __init__(self, training_image_path):

        self.training_image = cv2.imread(training_image_path, cv2.IMREAD_GRAYSCALE)
        if self.training_image is None:
            raise ValueError(f"Could not load training image: {training_image_path}")
        self.sift = cv2.SIFT_create()
        self.kp_train, self.des_train = self.sift.detectAndCompute(self.training_image, None)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def outline_object(self, frame):
        """Draws green outline if object detected."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        if des_frame is None:
            return frame

        matches = self.matcher.knnMatch(self.des_train, des_frame, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 8:
            return frame 

        train_pts = np.float32([self.kp_train[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        H, mask = cv2.findHomography(train_pts, frame_pts, cv2.RANSAC, 5.0)
        if H is None:
            return frame

        h, w = self.training_image.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        projected = cv2.perspectiveTransform(corners, H)

        cv2.polylines(frame, [np.int32(projected)], True, (0,255,0), 4)
        return frame


def webcam_demo():
    image_path = "/Users/mitch/Desktop/Robotics/Diddy CV/monkeydo.png"

    recognizer = ImageRecognizer(image_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(3, 1280) 
    cap.set(4, 720) 

    print("🎥 Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognizer.outline_object(frame)
        cv2.imshow("Object Outline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_demo()
