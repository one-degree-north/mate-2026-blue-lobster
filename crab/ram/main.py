import sys

import cv2
import numpy as np

# ---------- parameters ----------
MIN_INLIERS = 12
RATIO_TEST = 0.75
RANSAC_THR = 5.0
# --------------------------------

ref_path = sys.argv[1]

# Load reference image
ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
if ref is None:
    print("Could not read reference image.")
    sys.exit(1)

h_ref, w_ref = ref.shape[:2]

# SIFT feature extractor (best simple option)
sift = cv2.SIFT_create()

# Compute reference features once
kp_ref, des_ref = sift.detectAndCompute(ref, None)

# Matcher
bf = cv2.BFMatcher(cv2.NORM_L2)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features in frame
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    if des_frame is None:
        cv2.imshow("detect", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # KNN match
    matches = bf.knnMatch(des_ref, des_frame, k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    # Need enough matches
    if len(good) >= MIN_INLIERS:
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate homography
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
        if H is not None:
            mask = mask.ravel().astype(bool)
            inliers = mask.sum()

            if inliers >= MIN_INLIERS:
                # Map reference corners → frame
                corners = np.float32(
                    [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]
                ).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, H).astype(int)

                # Draw bounding box
                cv2.polylines(frame, [projected.reshape(-1, 2)], True, (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    f"inliers: {inliers}",
                    tuple(projected[0, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("detect", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
