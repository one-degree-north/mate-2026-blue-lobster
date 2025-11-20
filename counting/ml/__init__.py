# ML
# pip install ultralytics

from ultralytics import YOLO
import cv2

# Load YOLOv8 segmentation model
model = YOLO("yolov8x-seg.pt")   # segmentation model

# Read image
img_path = "Image_20251113_160910_132.jpeg"
img = cv2.imread(img_path)

# Run segmentation
results = model(img_path)    # or model(img)

# Show segmentation result
segmented_img = results[0].plot()  # draws masks + boxes on the image

cv2.imshow("YOLO Segmentation", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the result
results[0].save("leaves_segmented.jpg")
