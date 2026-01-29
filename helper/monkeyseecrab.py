import cv2
import numpy as np

class ImageRecognizer:
    def __init__(self, template_path: str):
        # Load the template
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError("Failed to load template")
        self.h, self.w = self.template.shape

        # Extract edges/contours from template for shape matching
        self.template_edges = cv2.Canny(self.template, 50, 150)
        contours, _ = cv2.findContours(self.template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Template has no contours")
        # Assume largest contour is the object
        self.template_contour = max(contours, key=cv2.contourArea)

    def outline_object(self, frame_rgb: np.ndarray) -> np.ndarray:
        return frame_rgb
        output = frame_rgb.copy()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # ---------- Step 1: Rough template matching ----------
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.5:  # tune threshold
            return output

        top_left = max_loc
        bottom_right = (top_left[0] + self.w, top_left[1] + self.h)

        # ---------- Step 2: Extract edges/contours in candidate region ----------
        candidate = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        candidate_edges = cv2.Canny(candidate, 50, 150)
        contours, _ = cv2.findContours(candidate_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return output
        candidate_contour = max(contours, key=cv2.contourArea)

        # ---------- Step 3: Compare shape with template ----------
        similarity = cv2.matchShapes(self.template_contour, candidate_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        print(similarity)

        if similarity > 25:  # tune threshold, lower is better
            return output  # shape does not match
        # ---------- Step 4: Compute homography ----------
        # Map template corners to candidate region corners
        template_corners = np.float32([[0,0],[self.w,0],[self.w,self.h],[0,self.h]]).reshape(-1,1,2)
        candidate_corners = np.float32([[top_left[0], top_left[1]],
                                        [bottom_right[0], top_left[1]],
                                        [bottom_right[0], bottom_right[1]],
                                        [top_left[0], bottom_right[1]]]).reshape(-1,1,2)
        H, _ = cv2.findHomography(template_corners, candidate_corners)

        # Draw outline
        proj = cv2.perspectiveTransform(template_corners, H)
        cv2.polylines(output, [np.int32(proj)], True, (0, 255, 0), 3)

        return output
