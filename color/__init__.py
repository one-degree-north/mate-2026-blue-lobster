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
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])

h, s, v = cv2.split(hsv)

# Put in range [lower, upper]. https://www.desmos.com/calculator/o28zw2tvwz
# Normalize distances only inside the green range
h_mask = np.clip((h - lower[0]) / (upper[0] - lower[0]), 0, 1)
s_mask = np.clip((s - lower[1]) / (upper[1] - lower[1]), 0, 1)
v_mask = np.clip((v - lower[2]) / (upper[2] - lower[2]), 0, 1)

# Combine the normalized distances
gradient = (h_mask + s_mask + v_mask) / 3 * 255

# Clamp everything outside the range to 0
inside_mask = cv2.inRange(hsv, lower, upper)
mask = cv2.bitwise_and(gradient.astype(np.uint8), gradient.astype(np.uint8), mask=inside_mask)

# mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
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
