import cv2
import numpy as np
import random
import requests

class MultiCrabTracker:

    def __init__(self,
                 min_present=8,
                 distance_thresh=120,
                 missing_frames=25,
                 recount_cooldown=60,
                 detection_interval=3):

        # Faster SIFT
        self.sift = cv2.SIFT_create(nfeatures=800)

        # Faster FLANN
        index_params = dict(algorithm=1, trees=4)
        search_params = dict(checks=32)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.training_images = []

        self.tracked = []
        self.counter = 0
        self.next_id = 0

        self.min_present = min_present
        self.distance_thresh = distance_thresh
        self.max_missing = missing_frames
        self.recount_cooldown = recount_cooldown

        self.counting = True

        # detection skipping
        self.detection_interval = detection_interval
        self.frame_index = 0

        # scale for faster processing
        self.scale = 0.5


    # -----------------------------------------
    # Load Training Image
    # -----------------------------------------

    def load_training_image(self, url):

        try:
            resp = requests.get(url)
            img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except:
            print("Failed to load training image")
            return False

        frame = cv2.resize(frame,(0,0),fx=self.scale,fy=self.scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)

        if des is None or len(kp) < 15:
            print("Not enough features")
            return False

        color = (
            random.randint(80,255),
            random.randint(80,255),
            random.randint(80,255)
        )

        self.training_images.append({
            "gray": gray,
            "kp": kp,
            "des": des,
            "color": color
        })

        print("Training image added")
        return True


    # -----------------------------------------
    # Detection
    # -----------------------------------------

    def detect(self, frame):

        self.frame_index += 1

        small = cv2.resize(frame,(0,0),fx=self.scale,fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        detections = []

        # skip detection some frames
        if self.frame_index % self.detection_interval == 0:

            kp_frame, des_frame = self.sift.detectAndCompute(gray, None)

            if des_frame is not None:

                for train in self.training_images:

                    matches = self.matcher.knnMatch(train["des"], des_frame, k=2)

                    good = []
                    for m,n in matches:
                        if m.distance < 0.55*n.distance:
                            good.append(m)

                    if len(good) < 18:
                        continue

                    src = np.float32(
                        [train["kp"][m.queryIdx].pt for m in good]
                    ).reshape(-1,1,2)

                    dst = np.float32(
                        [kp_frame[m.trainIdx].pt for m in good]
                    ).reshape(-1,1,2)

                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                    if H is None:
                        continue

                    h,w = train["gray"].shape

                    corners = np.float32(
                        [[0,0],[w,0],[w,h],[0,h]]
                    ).reshape(-1,1,2)

                    projected = cv2.perspectiveTransform(corners,H)

                    projected /= self.scale

                    cx = int(np.median(projected[:,0,0]))
                    cy = int(np.median(projected[:,0,1]))

                    detections.append({
                        "centroid": (cx,cy),
                        "bbox": projected,
                        "color": train["color"]
                    })

        self.update_tracked(detections)

        self.draw(frame)

        return frame


    # -----------------------------------------
    # Tracking
    # -----------------------------------------

    def update_tracked(self, detections):

        assigned = set()

        for crab in self.tracked:

            crab["missing"] += 1

            best_det = None
            best_dist = self.distance_thresh

            for i,det in enumerate(detections):

                if i in assigned:
                    continue

                dist = np.hypot(
                    crab["centroid"][0]-det["centroid"][0],
                    crab["centroid"][1]-det["centroid"][1]
                )

                if dist < best_dist:
                    best_dist = dist
                    best_det = i

            if best_det is not None:

                det = detections[best_det]
                assigned.add(best_det)

                # smoothing
                new_x = int(0.8*crab["centroid"][0] + 0.2*det["centroid"][0])
                new_y = int(0.8*crab["centroid"][1] + 0.2*det["centroid"][1])

                crab["centroid"] = (new_x,new_y)
                crab["bbox"] = det["bbox"]
                crab["color"] = det["color"]

                crab["frames"] += 1
                crab["missing"] = 0


        # new tracks
        for i,det in enumerate(detections):

            if i in assigned:
                continue

            self.tracked.append({
                "id": self.next_id,
                "centroid": det["centroid"],
                "bbox": det["bbox"],
                "color": det["color"],
                "frames": 1,
                "missing": 0,
                "counted": False,
                "cooldown": 0
            })

            self.next_id += 1


        # counting
        for crab in self.tracked:

            if crab["cooldown"] > 0:
                crab["cooldown"] -= 1

            if (crab["frames"] >= self.min_present
                and not crab["counted"]
                and self.counting
                and crab["cooldown"] == 0):

                self.counter += 1
                crab["counted"] = True
                crab["cooldown"] = self.recount_cooldown

                print("Total crabs:", self.counter)


        # remove lost
        self.tracked = [
            c for c in self.tracked
            if c["missing"] < self.max_missing
        ]


    # -----------------------------------------
    # Drawing
    # -----------------------------------------

    def draw(self,frame):

        for crab in self.tracked:

            if crab["frames"] < self.min_present:
                continue

            box = crab["bbox"]

            x_min = int(np.min(box[:,0,0]))
            y_min = int(np.min(box[:,0,1]))
            x_max = int(np.max(box[:,0,0]))
            y_max = int(np.max(box[:,0,1]))

            cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),crab["color"],3)

            cv2.putText(frame,
                        f"Crab {crab['id']}",
                        (x_min,y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        crab["color"],
                        2)

        cv2.putText(frame,f"Crabs counted: {self.counter}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        cv2.putText(frame,"SPACE = Start/Stop Counting",
                    (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

        cv2.putText(frame,"Q = Quit",
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------

if __name__ == "__main__":

    tracker = MultiCrabTracker()

    # Load your training image
    tracker.load_training_image(
        "https://raw.githubusercontent.com/one-degree-north/mate-2026-blue-lobster/main/monkeydo.png"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    cv2.namedWindow("Crab Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crab Tracker",1280,720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame")
            continue

        # Process frame
        frame = tracker.detect(frame)

        # Display
        cv2.imshow("Crab Tracker", frame)

        # Wait key required for macOS
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            tracker.counting = not tracker.counting

    cap.release()
    cv2.destroyAllWindows()