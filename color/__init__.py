import sys

import cv2
import numpy as np


def _debug_img(img):
    cv2.imshow("Debug", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Debug")

def remove_inner_boxes(boxes):
    keep = []
    for i, A in enumerate(boxes):
        Ax1, Ay1, Ax2, Ay2 = A
        inside = False

        for j, B in enumerate(boxes):
            if i == j: 
                continue
            Bx1, By1, Bx2, By2 = B

            # Check if A is completely inside B
            if (Ax1 >= Bx1 and Ay1 >= By1 and
                Ax2 <= Bx2 and Ay2 <= By2):
                inside = True
                break

        if not inside:
            keep.append(A)

    return keep

def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

img = cv2.imread(sys.argv[1])


kernel_size = 5
min_area = 100

# convert image from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# create a binary mask selecting pixels in the "green" HSV range
mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
_debug_img(mask)

n_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(mask)

boxes = []
for i in range(1, n_labels): # labels[0] is the background
    x, y, w, h, area = stats[i]
    cx, cy = centroid[i]
    if area > min_area:
        boxes.append((x, y, x + w, y + h))

boxes = remove_inner_boxes(boxes)
draw_boxes(img, boxes)
_debug_img(img)

"""

# define a small 5×5 kernel for morphological ops
k = np.ones((kernel_size, kernel_size), np.uint8)

# remove noise and close small gaps in the mask
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

# find contours (outlines) of connected green regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# store bounding boxes (x1, y1, x2, y2)
boxes = []
for c in contours:
    # ignore tiny noise regions
    if cv2.contourArea(c) > min_area:
        # compute bounding box of each contour
        x, y, w, h = cv2.boundingRect(c)

        # store as top-left and bottom-right corners
        boxes.append((x, y, x + w, y + h))


for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Green", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""