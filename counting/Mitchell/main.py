import cv2
import numpy as np
import random
import requests

class MultiCrabTracker:

    def __init__(self,
                 min_present=8,
                 distance_thresh=120,
                 missing_frames=20,
                 recount_cooldown=60):

        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        # MULTIPLE training images
        self.training_images = []

        self.tracked = []
        self.counter = 0

        self.min_present = min_present
        self.distance_thresh = distance_thresh
        self.max_missing = missing_frames
        self.recount_cooldown = recount_cooldown

        self.counting = True
        self.next_id = 0


    # ------------------------------------------------
    # Load training image from GitHub RAW
    # ------------------------------------------------

    def load_training_image(self, url):

        try:
            resp = requests.get(url)
            img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except:
            print("Failed to load training image")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)

        if des is None or len(kp) < 10:
            print("Not enough features")
            return False

        color = (
            random.randint(50,255),
            random.randint(50,255),
            random.randint(50,255)
        )

        self.training_images.append({
            "gray": gray,
            "kp": kp,
            "des": des,
            "color": color
        })

        print("Training image added")
        return True


    # ------------------------------------------------
    # Detection
    # ------------------------------------------------

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)

        detections = []

        if des_frame is not None:

            for train in self.training_images:

                matches = self.matcher.knnMatch(train["des"], des_frame, k=2)

                good = [m for m,n in matches if m.distance < 0.5*n.distance]

                if len(good) < 15:
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

                cx = int(np.median(projected[:,0,0]))
                cy = int(np.median(projected[:,0,1]))

                detections.append({
                    "centroid": (cx,cy),
                    "bbox": projected,
                    "color": train["color"]
                })

        self.update_tracked(detections)

        # DRAW
        for crab in self.tracked:

            if crab["frames"] < self.min_present:
                continue

            box = crab["bbox"]

            x_min = int(np.min(box[:,0,0]))
            y_min = int(np.min(box[:,0,1]))
            x_max = int(np.max(box[:,0,0]))
            y_max = int(np.max(box[:,0,1]))

            cv2.rectangle(
                frame,
                (x_min,y_min),
                (x_max,y_max),
                crab["color"],
                3
            )

            cv2.putText(
                frame,
                f"Crab {crab['id']}",
                (x_min,y_min-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                crab["color"],
                2
            )

        # UI text

        cv2.putText(frame,f"Crabs counted: {self.counter}",
                    (20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv2.putText(frame,"SPACE = Start/Stop Counting",
                    (20,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.putText(frame,"Q = Quit",
                    (20,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        return frame


    # ------------------------------------------------
    # Tracking
    # ------------------------------------------------

    def update_tracked(self, detections):

        # update existing tracks
        for crab in self.tracked:

            crab["missing"] += 1

            for det in detections:

                dist = np.hypot(
                    crab["centroid"][0] - det["centroid"][0],
                    crab["centroid"][1] - det["centroid"][1]
                )

                if dist < self.distance_thresh:

                    # centroid smoothing
                    new_x = int(0.7*crab["centroid"][0] + 0.3*det["centroid"][0])
                    new_y = int(0.7*crab["centroid"][1] + 0.3*det["centroid"][1])

                    crab["centroid"] = (new_x,new_y)
                    crab["bbox"] = det["bbox"]
                    crab["color"] = det["color"]

                    crab["frames"] += 1
                    crab["missing"] = 0

                    det["matched"] = True

        # add new tracks
        for det in detections:

            if "matched" in det:
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


        # count stable crabs
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


        # remove lost tracks
        self.tracked = [
            c for c in self.tracked
            if c["missing"] < self.max_missing
        ]


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    tracker = MultiCrabTracker()

    # MULTIPLE TRAINING IMAGES
    tracker.load_training_image(
        "https://raw.githubusercontent.com/one-degree-north/mate-2026-blue-lobster/main/monkeydo.png"
    )

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Crab Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crab Tracker",1280,720)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = tracker.detect(frame)

        cv2.imshow("Crab Tracker",frame)

        if cv2.getWindowProperty("Crab Tracker",cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            tracker.counting = not tracker.counting


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()