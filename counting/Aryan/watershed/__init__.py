import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Load image
# --------------------------
img = cv2.imread("Image_20251113_160910_132.jpeg")
img_original = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --------------------------
# 2. Segment green leaves
# --------------------------
lower_green = np.array([20, 0, 0])
upper_green = np.array([100, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Smooth mask
mask_blur = cv2.GaussianBlur(mask, (5,5), 0)

# --------------------------
# 3. Distance transform
# --------------------------
dist = cv2.distanceTransform(mask_blur, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# --------------------------
# 4. Sure background
# --------------------------
kernel = np.ones((3,3), np.uint8)
sure_bg = cv2.dilate(mask_blur, kernel, iterations=2)

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# --------------------------
# 5. Marker labeling
# --------------------------
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0

# --------------------------
# 6. Watershed
# --------------------------
markers = cv2.watershed(img, markers)
img_result = img.copy()

# Draw boundaries in red
img_result[markers == -1] = [0, 0, 255]

# Optional: assign random color to each leaf for instance segmentation
instance_img = np.zeros_like(img)
for marker_id in range(2, markers.max()+1):
    instance_img[markers == marker_id] = np.random.randint(0,255,3)

# --------------------------
# 7. Display results
# --------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

plt.subplot(1,4,2)
plt.title("Boundaries (Watershed)")
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

plt.subplot(1,4,3)
plt.title("Instance Segmentation")
plt.imshow(cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB))

plt.subplot(1,4,4)
plt.title("Green Mask")
plt.imshow(mask_blur, cmap='gray')

plt.show()
