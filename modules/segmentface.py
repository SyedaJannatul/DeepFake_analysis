import cv2
import mediapipe as mp
import numpy as np
from rembg import remove
from PIL import Image
import io

class FaceSegmenter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    def segment_face(self, image_input):
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            # Load from file path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError("Image not found or unable to load.")
        elif isinstance(image_input, np.ndarray):
            # Use numpy array directly (BGR format)
            image = image_input.copy()
        else:
            raise ValueError("Input must be file path string or numpy array")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(rgb_image)

        if not face_results.detections:
            # Use rembg with numpy array input
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            output_image = remove(pil_image)  # rembg handles PIL Images
            
            # Convert to numpy array and remove alpha channel
            output_image = np.array(output_image)
            if output_image.shape[2] == 4:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2RGB)
            return output_image

        # Existing face segmentation logic
        detection = face_results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y, width, height = (
            int(bboxC.xmin * w), int(bboxC.ymin * h),
            int(bboxC.width * w), int(bboxC.height * h)
        )

        segmentation_results = self.selfie_segmentation.process(rgb_image)
        mask = (segmentation_results.segmentation_mask > self.threshold).astype(np.uint8)
        
        face_mask = np.zeros_like(mask)
        face_mask[y:y+height, x:x+width] = mask[y:y+height, x:x+width]
        segmented_face = cv2.bitwise_and(image, image, mask=face_mask)

        return segmented_face

    # Updated helper methods to handle numpy arrays
    def save_segmented_face(self, image_input, output_path):
        segmented_face = self.segment_face(image_input)
        cv2.imwrite(output_path, segmented_face)

    def show_segmented_face(self, image_input):
        segmented_face = self.segment_face(image_input)
        cv2.imshow("Segmented Face", segmented_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()