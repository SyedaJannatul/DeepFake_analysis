import cv2
import mediapipe as mp
import numpy as np
from rembg import remove  
from PIL import Image  

class FaceSegmenter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Initialize face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 for general use, 0 for close-up faces
            min_detection_confidence=0.5
        )
        # Initialize selfie segmentation (for background removal)
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 1 for general use, 0 for close-up faces
        )

    def segment_face(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Convert to RGB (MediaPipe requires RGB input)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Detect the face
        face_results = self.face_detection.process(rgb_image)
        if not face_results.detections:
            # Use rembg to remove the background
            with open(image_path, "rb") as input_file:
                input_image = input_file.read()
                output_image = remove(input_image)  
            # Convert the output image to a numpy array
            output_image = np.array(Image.open(io.BytesIO(output_image)))
            # Convert RGBA to RGB (remove alpha channel)
            if output_image.shape[2] == 4:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2RGB)
            return output_image

        # Get the bounding box of the first detected face
        detection = face_results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                              int(bboxC.width * w), int(bboxC.height * h)

        # Step 2: Segment the foreground (selfie segmentation)
        segmentation_results = self.selfie_segmentation.process(rgb_image)
        if segmentation_results.segmentation_mask is None:
            raise ValueError("Segmentation failed.")

        # Create a binary mask
        mask = (segmentation_results.segmentation_mask > self.threshold).astype(np.uint8)

        # Step 3: Crop the face using the bounding box
        face_mask = np.zeros_like(mask)
        face_mask[y:y+height, x:x+width] = mask[y:y+height, x:x+width]

        # Apply the mask to the original image
        segmented_face = cv2.bitwise_and(image, image, mask=face_mask)

        return segmented_face

    def save_segmented_face(self, image_path, output_path):
        segmented_face = self.segment_face(image_path)
        cv2.imwrite(output_path, segmented_face)

    def show_segmented_face(self, image_path):
        segmented_face = self.segment_face(image_path)
        cv2.imshow("Segmented Face", segmented_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()