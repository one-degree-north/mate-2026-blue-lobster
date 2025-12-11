import cv2
import sys
import numpy as np

# ----------------------------
# Load reference image
# ----------------------------
ref = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
if ref is None:
    raise ValueError("Could not load reference image")

# Create SIFT detector
sift = cv2.SIFT_create()

# Compute keypoints and descriptors of the reference image
kp1, des1 = sift.detectAndCompute(ref, None)

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_MATCH_COUNT = 10

# ----------------------------
# Open camera
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in camera frame
    kp2, des2 = sift.detectAndCompute(gray, None)
    
    if des2 is not None:
        # Match descriptors
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # If enough matches, draw bounding box
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = ref.shape
            pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(frame, [np.int32(dst)], True, (0,255,0), 3)

    # Show result
    cv2.imshow("SIFT Camera Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()