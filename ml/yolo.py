import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Green color range
lower_green = np.array([35, 40, 40])   # adjust if necessary
upper_green = np.array([85, 255, 255])

# Create mask of green pixels
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: remove noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Draw contours around green objects
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

print(f"Detected {len(contours)} green objects.")

cv2.imshow("Green Objects", img)
# cv2.imshow("Mask", mask)
cv2.waitKey(0)
