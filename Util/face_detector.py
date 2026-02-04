import sys
import os
from insightface.app import FaceAnalysis
import numpy as np
from Util.config import AppConfig
from Util.database import DatabaseManager


class FaceDuplicateDetector:
    """Detects faces and identifies duplicates against a database."""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.app = self._initialize_insightface()
        print("FaceEmbedder (InsightFace) initialized.")

    @staticmethod
    def _suppress_stdout(func):
        """A decorator to suppress stdout during function execution."""
        def wrapper(*args, **kwargs):
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                try:
                    return func(*args, **kwargs)
                finally:
                    sys.stdout = old_stdout
        return wrapper

    @_suppress_stdout
    def _initialize_insightface(self):
        """Initializes and prepares the FaceAnalysis model."""
        ctx_id = 0 if AppConfig.GPU_MODE else -1
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_thresh=AppConfig.FACE_DETECTION_CONFIDENCE)
        return app

    def _extract_faces(self, img: np.ndarray) -> list:
        """Extracts face embeddings and locations from an image."""
        faces = self.app.get(img)
        extracted_data = []
        if not faces:
            return []

        h, w = img.shape[:2]
        for face in faces:
            if face.det_score < AppConfig.FACE_DETECTION_CONFIDENCE:
                continue

            bbox = np.clip(face.bbox.astype(int), 0, [w, h, w, h])
            left, top, right, bottom = bbox
            
            if (right - left < AppConfig.FACE_MIN_SIZE) or (bottom - top < AppConfig.FACE_MIN_SIZE):
                continue

            extracted_data.append({
                'face_location': (top, right, bottom, left),
                'embedding': face.embedding.astype(np.float32)
            })
        return extracted_data
    
    def process_image(self, image_array: np.ndarray, filename: str) -> bool:
        """
        Processes an image to find and save faces, and checks for duplicates.
        Returns True if any duplicate face is found, False otherwise.
        """
        if self.db_manager.is_image_processed_for_faces(filename):
            print(f"Info: Image '{filename}' already processed for faces. Skipping.")
            return False 

        extracted_faces = self._extract_faces(image_array)
        if not extracted_faces:
            print(f"No faces detected in '{filename}'.")
            return False

        all_existing_faces = self.db_manager.get_all_faces()
        has_duplicates = False

        for new_face in extracted_faces:
            # The _is_face_duplicate method returns a numpy.bool_, which works fine in a
            # Python 'if' statement. The 'has_duplicates' variable will be a standard bool.
            if self._is_face_duplicate(new_face['embedding'], all_existing_faces):
                has_duplicates = True
                print(f"Duplicate face detected in '{filename}'.")
                # Could add more info here about which face it matched

            # Save the new face regardless of duplication for future checks
            self.db_manager.save_face(filename, new_face['face_location'], new_face['embedding'])

        return has_duplicates

    @staticmethod
    def _is_face_duplicate(new_embedding: np.ndarray, existing_faces: list) -> np.bool_:
        """Compares a new face embedding against a list of existing ones."""
        if not existing_faces:
            return np.bool_(False)
        
        existing_embeddings = np.array([face['embedding'] for face in existing_faces])
        distances = np.linalg.norm(existing_embeddings - new_embedding, axis=1)
        
        return np.any(distances < AppConfig.FACE_DISTANCE_THRESHOLD)
