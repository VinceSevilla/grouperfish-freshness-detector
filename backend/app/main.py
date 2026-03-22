from fastapi import UploadFile, File
import shutil
"""
FastAPI Application for Fish Freshness Detection
Handles image upload and real-time predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from pathlib import Path
import io
from PIL import Image
import base64


from app.detection.eye_detector import EyeDetector
from app.detection.gills_detector import GillsDetector
from app.models_service.model_loader import ModelLoader
from app.config import MODELS_PATH

# Initialize FastAPI app
app = FastAPI(
    title="Fish Freshness Detection API",
    description="Real-time fish freshness classification using eyes and gills",
    version="1.0.0"
)


class CameraRequest(BaseModel):
    base64_image: str

# Add CORS middleware
# Allow all origins for debugging (not for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://grouperfish-freshness-detector.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TEMPORARY: Admin endpoint to upload .h5 model files
@app.post("/admin/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    # Only allow .h5 files
    if not file.filename.endswith(".h5"):
        return {"error": "Only .h5 files are allowed."}
    # Save to backend/results/
    save_path = Path("results") / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filename": file.filename}


# Initialize models and detectors
MODELS_DIR = Path(MODELS_PATH)
print(f"[INIT] MODELS_PATH = {MODELS_PATH}")
print(f"[INIT] MODELS_DIR = {MODELS_DIR}")
print(f"[INIT] MODELS_DIR.exists() = {MODELS_DIR.exists()}")
if MODELS_DIR.exists():
    print(f"[INIT] Files in MODELS_DIR: {list(MODELS_DIR.glob('*.h5'))}")
eye_detector = EyeDetector()
gills_detector = GillsDetector()
model_loader = ModelLoader(str(MODELS_DIR))


def detect_and_extract_regions(image: np.ndarray) -> dict:
    """Detect and extract both eye and gill regions using separate detectors"""
    print(f"[MAIN] Starting detection on image: {image.shape}")
    
    # Detect eyes
    print("[MAIN] Calling eye detector...")
    eye_result = eye_detector.detect_and_extract(image)
    print(f"[MAIN] Eye detection result: {eye_result['eye_detected']}")
    
    # Detect gills (with eye anchoring for anatomical accuracy)
    print("[MAIN] Calling gills detector...")
    gill_result = gills_detector.detect_and_extract(image, eye_box=eye_result['eye_bbox'])
    print(f"[MAIN] Gill detection result: {gill_result['gill_detected']}")
    
    # Combine results
    return {
        'eye': eye_result['eye'],
        'gill': gill_result['gill'],
        'eye_bbox': eye_result['eye_bbox'],
        'gill_bbox': gill_result['gill_bbox'],
        'full_image': image,
        'eye_detected': eye_result['eye_detected'],
        'gill_detected': gill_result['gill_detected']
    }


def convert_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def convert_base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "eye_model_loaded": model_loader.eye_model is not None,
        "gill_model_loaded": model_loader.gill_model is not None
    }


@app.post("/predict/upload")
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Predict freshness from uploaded image
    Detects eyes and gills, returns predictions for both
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect regions
        regions = detect_and_extract_regions(image_cv)
        
        # Prepare predictions
        predictions = {
            "eye_detected": regions['eye_detected'],
            "gill_detected": regions['gill_detected'],
            "eye_prediction": None,
            "gill_prediction": None,
            "integrated_prediction": None,
            "annotated_image": None,
            "original_image_base64": convert_image_to_base64(image_cv)
        }
        
        # Predict from eye if detected
        print(f"[PREDICT] Eye detected: {regions['eye_detected']}, Eye ROI is None: {regions['eye'] is None}")
        if regions['eye'] is not None:
            print(f"[PREDICT] Eye ROI shape: {regions['eye'].shape}")
        # Robust check: eye must be detected, not None, and shape must be (224, 224, 3)
        if regions['eye_detected'] and regions['eye'] is not None:
            if isinstance(regions['eye'], np.ndarray) and regions['eye'].ndim == 3 and regions['eye'].shape[0] == 224 and regions['eye'].shape[1] == 224:
                print(f"[PREDICT] Calling predict_eye...")
                try:
                    predictions['eye_prediction'] = model_loader.predict_eye(regions['eye'], include_glcm=True)
                    print(f"[PREDICT] Eye prediction result: {predictions['eye_prediction']}")
                except Exception as e:
                    print(f"[PREDICT] ERROR in predict_eye: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    predictions['eye_prediction'] = None
            else:
                print(f"[PREDICT] Eye ROI shape invalid: {regions['eye'].shape if hasattr(regions['eye'], 'shape') else 'unknown'}")
        else:
            print(f"[PREDICT] Eye not detected or ROI is None")
        
        # Predict from gill if detected
        print(f"[PREDICT] Gill detected: {regions['gill_detected']}, Gill ROI is None: {regions['gill'] is None}")
        if regions['gill'] is not None:
            print(f"[PREDICT] Gill ROI shape: {regions['gill'].shape}")
        if regions['gill_detected'] and regions['gill'] is not None:
            print(f"[PREDICT] Calling predict_gill...")
            predictions['gill_prediction'] = model_loader.predict_gill(regions['gill'], include_glcm=True)
            print(f"[PREDICT] Gill prediction result: {predictions['gill_prediction']}")
        
        # Calculate overall prediction using conservative approach (worse prediction)
        if predictions['eye_prediction'] and predictions['gill_prediction']:
            # Map class names to severity (higher = worse)
            severity_map = {'fresh': 0, 'less_fresh': 1, 'starting_to_rot': 2, 'rotten': 3}
            
            eye_class = predictions['eye_prediction']['class']
            gill_class = predictions['gill_prediction']['class']
            eye_severity = severity_map.get(eye_class, 0)
            gill_severity = severity_map.get(gill_class, 0)
            
            # Select the worse prediction (higher severity)
            if eye_severity >= gill_severity:
                predictions['integrated_prediction'] = predictions['eye_prediction']
            else:
                predictions['integrated_prediction'] = predictions['gill_prediction']
        else:
            # If only one is available, use that one
            predictions['integrated_prediction'] = predictions['eye_prediction'] or predictions['gill_prediction']
        
        # Create annotated image
        annotated = image_cv.copy()
        
        if regions['eye_bbox']:
            x, y, w, h = regions['eye_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Eye", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if regions['gill_bbox']:
            x, y, w, h = regions['gill_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated, "Gill", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        predictions['annotated_image'] = convert_image_to_base64(annotated)
        
        return JSONResponse(predictions)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/camera")
async def predict_from_camera(payload: CameraRequest):
    """
    Predict freshness from camera frame (base64 encoded)
    """
    try:
        # Decode image
        image_cv = convert_base64_to_image(payload.base64_image)
        
        # Detect regions
        regions = detect_and_extract_regions(image_cv)
        
        # Prepare predictions
        predictions = {
            "eye_detected": regions['eye_detected'],
            "gill_detected": regions['gill_detected'],
            "eye_prediction": None,
            "gill_prediction": None,
            "integrated_prediction": None,
            "annotated_image": None,
        }
        
        # Predict from eye if detected
        print(f"[PREDICT-CAM] Eye detected: {regions['eye_detected']}, Eye ROI is None: {regions['eye'] is None}")
        if regions['eye'] is not None:
            print(f"[PREDICT-CAM] Eye ROI shape: {regions['eye'].shape}")
        if regions['eye_detected'] and regions['eye'] is not None:
            print(f"[PREDICT-CAM] Calling predict_eye...")
            predictions['eye_prediction'] = model_loader.predict_eye(regions['eye'], include_glcm=True)
            print(f"[PREDICT-CAM] Eye prediction result: {predictions['eye_prediction']}")
        
        # Predict from gill if detected
        print(f"[PREDICT-CAM] Gill detected: {regions['gill_detected']}, Gill ROI is None: {regions['gill'] is None}")
        if regions['gill'] is not None:
            print(f"[PREDICT-CAM] Gill ROI shape: {regions['gill'].shape}")
        if regions['gill_detected'] and regions['gill'] is not None:
            print(f"[PREDICT-CAM] Calling predict_gill...")
            predictions['gill_prediction'] = model_loader.predict_gill(regions['gill'], include_glcm=True)
            print(f"[PREDICT-CAM] Gill prediction result: {predictions['gill_prediction']}")
        
        # Calculate overall prediction using conservative approach (worse prediction)
        if predictions['eye_prediction'] and predictions['gill_prediction']:
            # Map class names to severity (higher = worse)
            severity_map = {'fresh': 0, 'less_fresh': 1, 'starting_to_rot': 2, 'rotten': 3}
            
            eye_class = predictions['eye_prediction']['class']
            gill_class = predictions['gill_prediction']['class']
            eye_severity = severity_map.get(eye_class, 0)
            gill_severity = severity_map.get(gill_class, 0)
            
            # Select the worse prediction (higher severity)
            if eye_severity >= gill_severity:
                predictions['integrated_prediction'] = predictions['eye_prediction']
            else:
                predictions['integrated_prediction'] = predictions['gill_prediction']
        else:
            # If only one is available, use that one
            predictions['integrated_prediction'] = predictions['eye_prediction'] or predictions['gill_prediction']
        
        # Create annotated image
        annotated = image_cv.copy()
        
        if regions['eye_bbox']:
            x, y, w, h = regions['eye_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Eye", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if regions['gill_bbox']:
            x, y, w, h = regions['gill_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated, "Gill", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        predictions['annotated_image'] = convert_image_to_base64(annotated)
        
        return JSONResponse(predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect_regions(file: UploadFile = File(...)):
    """
    Detect eyes and gills regions in uploaded image
    Returns bounding boxes and extracted ROIs
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        regions = detect_and_extract_regions(image_cv)
        
        result = {
            "eye_detected": regions['eye_detected'],
            "gill_detected": regions['gill_detected'],
            "eye_bbox": regions['eye_bbox'],
            "gill_bbox": regions['gill_bbox'],
            "original_image": convert_image_to_base64(image_cv),
            "annotated_image": None,
            "eye_roi": None,
            "gill_roi": None
        }
        
        # Create annotated image
        annotated = image_cv.copy()
        
        if regions['eye_bbox']:
            x, y, w, h = regions['eye_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Eye", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if regions['eye'] is not None:
                result['eye_roi'] = convert_image_to_base64(regions['eye'])
        
        if regions['gill_bbox']:
            x, y, w, h = regions['gill_bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated, "Gill", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if regions['gill'] is not None:
                result['gill_roi'] = convert_image_to_base64(regions['gill'])
        
        result['annotated_image'] = convert_image_to_base64(annotated)
        
        return JSONResponse(result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
