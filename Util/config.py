import torch
class AppConfig:
    """Application-wide configurations."""
    # Database
    DB_PATH = './millet_analysis.db'

    # Image Pre-processing
    BLUR_THRESHOLD = 100 # Laplacian variance threshold for blur
    CROP_FIELD_THRESHOLD = 10 # Minimum percentage of green/yellow

    # Perceptual Hash Duplicate Detection
    PHASH_DUPLICATE_THRESHOLD = 5 # threshold for hamming distance

    # Face-based Duplicate Detection
    FACE_DETECTION_CONFIDENCE = 0.8
    FACE_DISTANCE_THRESHOLD = 20.0 # Euclidean distance for duplicate faces
    FACE_MIN_SIZE = 20 # Minimum pixel size (width or height) for a face

    # Model & ML
    MODEL_PATH = './model/millet_model.pth'

    CLASS_NAMES = ['jowar', 'non_millet', 'ragi']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_MODE = torch.cuda.is_available()