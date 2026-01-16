import cv2
import numpy as np
from imagededup.methods import PHash
from Util.config import AppConfig
class ImagePreprocessor:
    """Handles initial image analysis like blur and content detection."""
    def analyze(self, img: np.ndarray) -> dict:
        """Analyzes an image for blur and crop field content."""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (BGR) image.")

        # 1. Blur detection using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Green/Yellow detection for crop fields
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
        green_yellow_ratio = (np.sum(green_mask | yellow_mask) / img.size) * 100

        return {
            "is_blurred": bool(lap_var <= AppConfig.BLUR_THRESHOLD),
            "is_crop_field": bool(green_yellow_ratio >= AppConfig.CROP_FIELD_THRESHOLD),
        }

    def compute_phash(self, image: np.ndarray) -> str:
        """Computes the perceptual hash for an image."""
        phasher = PHash()
        return phasher.encode_image(image_array=image)