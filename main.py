from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

# Custom utilities
from Util.config import AppConfig
from Util.database import DatabaseManager
from Util.classifier import MilletClassifier
from Util.preprocessor import ImagePreprocessor
from Util.face_detector import FaceDuplicateDetector

# Initialize FastAPI app
app = FastAPI(
    title="Millet Image Analysis API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager(AppConfig.DB_PATH)
millet_classifier = MilletClassifier(AppConfig.MODEL_PATH,AppConfig.CLASS_NAMES,AppConfig.DEVICE)
face_detector = FaceDuplicateDetector(db_manager)
preprocessor = ImagePreprocessor()

# --- Utility Functions ---
def read_uploaded_file(file: UploadFile):
    contents = file.file.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    np_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, np_image_bgr

picture_category = {
    "blur": 0,
    "crop_field": 1,
    "face": 2,
    "duplicate": 3
}

millet_category = {
    "jowar": 4,
    "non_millet": 5,
    "ragi": 6
}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Example endpoint with parameter
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}

# Example "prediction" endpoint (dummy)
@app.get("/predict")
def predict(x: int, y: int):
    result = x + y
    return {"x": x, "y": y, "prediction": result}


@app.post("/analyze")
async def analyze_image(imageFile: UploadFile = File(...)):
    if not imageFile:
        raise HTTPException(status_code=400, detail="No file provided")

    filename = secure_filename(imageFile.filename)

    try:
        # Step 1: Read Image
        pil_img, np_img = read_uploaded_file(imageFile)

        # Step 2: Preprocessing
        pre_analysis = preprocessor.analyze(np_img)
        if pre_analysis["is_blurred"]:
            return JSONResponse(
                status_code=422,
                content={
                    "status_code": 422,
                    "category": picture_category["blur"],
                    "error": "Image is too blurry."
                }
            )

        if not pre_analysis["is_crop_field"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status_code": 400,
                    "category": picture_category["crop_field"],
                    "error": "Image does not appear to contain a crop field."
                }
            )

        # Step 3: Perceptual hash duplicate check
        p_hash = preprocessor.compute_phash(np_img)
        if db_manager.is_phash_duplicate(p_hash):
            return JSONResponse(
                status_code=409,
                content={
                    "status_code": 409,
                    "category": picture_category["duplicate"],
                    "error": "Duplicate image detected (perceptual hash)."
                }
            )

        db_manager.add_phash(filename, p_hash)

        # Step 4: Face duplicate detection
        face_duplicate_found = face_detector.process_image(np_img, filename)

        # Step 5: Classification
        predicted_class, predict_confidence = millet_classifier.predict(pil_img)

        return {
            "filename": filename,
            "is_blurred": pre_analysis["is_blurred"],
            "is_crop_field": pre_analysis["is_crop_field"],
            "face_duplicate_found": face_duplicate_found,
            "predicted_class": predicted_class,
            "millet_class_code": millet_category.get(predicted_class, -1),
            "confidence": predict_confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
