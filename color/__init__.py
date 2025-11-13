import sys

import cv2
import numpy as np


def detect_green_boxes(img: cv2.typing.MatLike, min_area=100):
    # convert image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create a binary mask selecting pixels in the "green" HSV range
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # define a small 5×5 kernel for morphological ops
    k = np.ones((5, 5), np.uint8)

    # remove noise and close small gaps in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

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

    return boxes

def draw_boxes(img: cv2.typing.MatLike, boxes: list[tuple[int, int, int, int]]):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Green", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    boxes = detect_green_boxes(img)
    print(boxes)

    draw_boxes(img, boxes)
