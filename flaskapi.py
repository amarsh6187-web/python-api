from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Import custom utility modules
from Util.config import AppConfig
from Util.database import DatabaseManager
from Util.classifier import MilletClassifier
from Util.preprocessor import ImagePreprocessor
from Util.face_detector import FaceDuplicateDetector
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
db_manager = DatabaseManager(AppConfig.DB_PATH)
millet_classifier = MilletClassifier(AppConfig.MODEL_PATH, AppConfig.CLASS_NAMES, AppConfig.DEVICE)
face_detector = FaceDuplicateDetector(db_manager)
preprocessor = ImagePreprocessor()

# --- Utility Functions ---
def read_uploaded_file(file_storage):
    """Convert uploaded file to PIL and OpenCV image formats."""
    contents = file_storage.read()
    pil_image = Image.open(BytesIO(contents)).convert('RGB')
    np_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, np_image_bgr

picture_category = {
    'blur': 0,
    'crop_field': 1,
    'face': 2,
    'duplicate': 3
}

# For final classification labels
millet_category = {
    'jowar': 4,
    'non_millet': 5,
    'ragi': 6
}


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # Or your domain
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# --- API Endpoints ---
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Millet Image Analysis API"})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'imageFile' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['imageFile']
    filename = secure_filename(file.filename)

    try:
        # Step 1: Read Image
        pil_img, np_img = read_uploaded_file(file)

        # Step 2: Preprocess - check blur and crop field
        pre_analysis = preprocessor.analyze(np_img)
        if pre_analysis["is_blurred"]:
           return jsonify({
                "status_code": 422,
                "category": picture_category["blur"],
                "error": "Image is too blurry."
            }), 422


        if not pre_analysis["is_crop_field"]:
            return jsonify({
                "status_code": 400,
                "category": picture_category["crop_field"],
                "error": "Image does not appear to contain a crop field."
            }), 400

        # Step 3: Perceptual hash check for duplicates
        p_hash = preprocessor.compute_phash(np_img)
        if db_manager.is_phash_duplicate(p_hash):
            return jsonify({
                "status_code": 409,
                "status": picture_category["duplicate"],
                "error": "Duplicate image detected (perceptual hash)."
            }), 409

        db_manager.add_phash(filename, p_hash)

        # Step 4: Face duplicate detection
        face_duplicate_found = face_detector.process_image(np_img, filename)

        # Step 5: Classification
        predicted_class,predict_confidence = millet_classifier.predict(pil_img)

        return jsonify({
            "filename": filename,
            "is_blurred": pre_analysis["is_blurred"],
            "is_crop_field": pre_analysis["is_crop_field"],
            "face_duplicate_found": face_duplicate_found,
            "predicted_class": predicted_class,
            "millet_class_code": millet_category.get(predicted_class, -1), 
            "confidence": predict_confidence
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run Application ---
if __name__ == "__main__":
    app.run(host="10.44.106.116", port=5000, debug=True)

