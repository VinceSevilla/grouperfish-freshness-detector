"""
Model Loader Service
Loads and manages the trained models
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from app.models_service.glcm_extractor import GLCMExtractor
import traceback


class ModelLoader:
    """Loads and manages fish freshness models"""
    
    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    def __init__(self, models_dir: str):
        """
        Initialize model loader
        
        Args:
            models_dir: Path to directory containing .h5 model files
        """
        self.models_dir = Path(models_dir)
        self.eye_model: Optional[tf.keras.Model] = None
        self.gill_model: Optional[tf.keras.Model] = None
        print(f"[DEBUG] ModelLoader will load models from: {self.models_dir.resolve()}")
        
        # Create ResNet50 feature extractor (2048 features)
        self.resnet_model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        # Create MobileNetV1 feature extractor (1024 features)
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        self._load_models()
    
    def _load_models(self):
        """Load eye and gill models only"""
        try:
            eye_path = self.models_dir / 'hybrid_eyes_model.h5'
            print(f"[DEBUG] Eye model path: {eye_path.resolve()}")
            if eye_path.exists():
                self.eye_model = load_model(str(eye_path), compile=False)
                print(f"✓ Loaded eye model")
            else:
                print(f"⚠ Eye model not found: {eye_path}")
            gill_path = self.models_dir / 'hybrid_gills_model.h5'
            print(f"[DEBUG] Gill model path: {gill_path.resolve()}")
            if gill_path.exists():
                self.gill_model = load_model(str(gill_path), compile=False)
                print(f"✓ Loaded gill model")
            else:
                print(f"⚠ Gill model not found: {gill_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _flatten_glcm_features(self, glcm_dict: dict) -> np.ndarray:
        """Flatten GLCM feature dict into a 1D vector of exactly 29 features"""
        features = []
        
        # Extract 6 basic features (averaged across angles)
        if 'basic' in glcm_dict and glcm_dict['basic']:
            basic = glcm_dict['basic']
            features.extend([
                basic.get('contrast', 0.0),
                basic.get('dissimilarity', 0.0),
                basic.get('homogeneity', 0.0),
                basic.get('energy', 0.0),
                basic.get('correlation', 0.0),
                basic.get('ASM', 0.0)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Extract 18 multi-scale features (3 scales × 6 properties)
        if 'multi_scale' in glcm_dict and glcm_dict['multi_scale']:
            for scale_key in ['scale_1', 'scale_2', 'scale_3']:
                if scale_key in glcm_dict['multi_scale']:
                    scale_data = glcm_dict['multi_scale'][scale_key]
                    features.extend([
                        scale_data.get('contrast', 0.0),
                        scale_data.get('dissimilarity', 0.0),
                        scale_data.get('homogeneity', 0.0),
                        scale_data.get('energy', 0.0),
                        scale_data.get('correlation', 0.0),
                        scale_data.get('ASM', 0.0)
                    ])
                else:
                    features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 18)
        
        # Extract 5 directional variance features (variance across 4 directions)
        if 'directional' in glcm_dict and glcm_dict['directional']:
            dir_data = glcm_dict['directional']
            # Calculate variance of key properties across directions
            for prop in ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']:
                values = [dir_data[d].get(prop, 0.0) for d in ['0°', '45°', '90°', '135°'] if d in dir_data]
                if values:
                    features.append(float(np.var(values)))
                else:
                    features.append(0.0)
        else:
            features.extend([0.0] * 5)
        
        # Ensure exactly 29 features
        features = features[:29]
        while len(features) < 29:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def preprocess_image_resnet(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ResNet50"""
        # Ensure uint8 first (0-255 range)
        if image.dtype != np.uint8:
            image = (image.clip(0, 255)).astype(np.uint8)
        
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = np.stack([image] * 3, axis=2)
        
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            import cv2
            image = cv2.resize(image, (224, 224))
        
        # Apply ResNet50 preprocessing (expects uint8 0-255 or float32 0-1)
        image = preprocess_input(image.astype(np.float32))
        
        return image
    
    def preprocess_image_mobilenet(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MobileNetV1"""
        # Ensure uint8 first (0-255 range)
        if image.dtype != np.uint8:
            image = (image.clip(0, 255)).astype(np.uint8)
        
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = np.stack([image] * 3, axis=2)
        
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            import cv2
            image = cv2.resize(image, (224, 224))
        
        # Apply MobileNet preprocessing (expects uint8 0-255 or float32 0-1)
        image = mobilenet_preprocess(image.astype(np.float32))
        
        return image
    
    def apply_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Minimal white balance - preserve natural color signature for real fish"""
        import cv2
        # Just return the image as-is - preserve natural colors
        # Aggressive white balance was corrupting real-world fish images
        return img
    
    def predict_eye(self, eye_image: np.ndarray, include_glcm: bool = False, eye_bbox: tuple = None) -> Optional[dict]:
        """
        Predict freshness from eye image using new hybrid model (expects 224x224x3 image input)
        Args:
            eye_image: Full fish image or eye ROI
            include_glcm: If True, include GLCM texture features in output
            eye_bbox: Optional bounding box (x, y, w, h) for eye ROI. If provided, will use EyeDetector's extract_eye_roi.
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        if self.eye_model is None:
            return None
        try:
            import cv2
            from app.detection.eye_detector import EyeDetector
            if eye_bbox is not None:
                detector = EyeDetector()
                roi = detector.extract_eye_roi(eye_image, eye_bbox)
                if roi is None:
                    print("[ERROR][EYE] Eye ROI extraction failed.")
                    return None
                # ROI from detector is already RGB uint8, just convert to float32
                image = roi.astype(np.float32)
            else:
                # Fallback: eye_image is already RGB from extract_eye_roi
                image_rgb = eye_image if len(eye_image.shape) == 3 and eye_image.shape[2] == 3 else eye_image
                if image_rgb.shape[:2] != (224, 224):
                    image_rgb = cv2.resize(image_rgb, (224, 224))
                # Already RGB, just convert to float32 (0-255 range)
                image = image_rgb.astype(np.float32)

            # Debug: Save preprocessed image to disk for inspection
            import os
            debug_dir = './debug_eye_detector/preprocessed/'
            os.makedirs(debug_dir, exist_ok=True)
            
            # SAVE: Original extracted ROI (before any preprocessing)
            roi_path = os.path.join(debug_dir, 'eye_roi_ORIGINAL.png')
            # eye_image is RGB from extract_eye_roi, convert to BGR for cv2.imwrite
            if len(eye_image.shape) == 3 and eye_image.shape[2] == 3:
                cv2.imwrite(roi_path, cv2.cvtColor(eye_image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(roi_path, eye_image)
            print(f"[DEBUG][EYE] Saved original ROI to {roi_path}")
            
            debug_img_path = os.path.join(debug_dir, 'eye_preprocessed.png')
            img_to_save = image.astype('uint8')
            cv2.imwrite(debug_img_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG][EYE] Saved preprocessed image to {debug_img_path}")
            
            # DEBUG: Also save as heatmap showing brightness distribution
            gray_version = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray_version, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug_dir, 'eye_brightness_heatmap.png'), heatmap)
            print(f"[DEBUG][EYE] Image brightness stats - min:{gray_version.min()}, max:{gray_version.max()}, mean:{gray_version.mean():.1f}")

            # --- Extract features as in training (no additional preprocessing for eyes) ---
            resnet_feat = self.resnet_model.predict(np.expand_dims(self.preprocess_image_resnet(image), axis=0), verbose=0)[0]
            mobilenet_feat = self.mobilenet_model.predict(np.expand_dims(self.preprocess_image_mobilenet(image), axis=0), verbose=0)[0]
            # CRITICAL FIX: Compute GLCM from uint8 image (0-255), not float32 (0-1)
            glcm_dict = GLCMExtractor.compute_glcm_summary(img_to_save)
            glcm_feat = self._flatten_glcm_features(glcm_dict)
            cnn_features = np.concatenate([resnet_feat, mobilenet_feat])
            batch_cnn = np.expand_dims(cnn_features, axis=0)
            batch_glcm = np.expand_dims(glcm_feat, axis=0)
            # Debug: print feature stats
            print(f"[DEBUG][EYE] CNN features mean: {cnn_features.mean():.4f}, std: {cnn_features.std():.4f}, min: {cnn_features.min():.4f}, max: {cnn_features.max():.4f}")
            print(f"[DEBUG][EYE] GLCM features mean: {glcm_feat.mean():.4f}, std: {glcm_feat.std():.4f}, min: {glcm_feat.min():.4f}, max: {glcm_feat.max():.4f}, non-zero: {np.count_nonzero(glcm_feat)}/29")
            print(f"[DEBUG][EYE] ResNet50 specific: mean={resnet_feat.mean():.4f}, top5: {np.sort(resnet_feat)[-5:]}")
            print(f"[DEBUG][EYE] MobileNetV1 specific: mean={mobilenet_feat.mean():.4f}, top5: {np.sort(mobilenet_feat)[-5:]}")
            predictions = self.eye_model.predict([batch_cnn, batch_glcm], verbose=0)
            print(f"[DEBUG][EYE] Raw prediction probabilities: {predictions[0]}")
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            result = {
                'class': self.FRESHNESS_CLASSES[class_idx],
                'confidence': confidence,
                'probabilities': {
                    self.FRESHNESS_CLASSES[i]: float(predictions[0][i])
                    for i in range(len(self.FRESHNESS_CLASSES))
                }
            }
            if include_glcm:
                result['glcm_features'] = glcm_dict
            return result
        except Exception as e:
            print(f"[ERROR] Error predicting eye: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            return None
    
    def normalize_gill_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Minimal gill preprocessing - preserve natural color for real fish.
        
        NOTE: Previous aggressive white balance was corrupting real-world images.
        The training data had unnatural characteristics; we should NOT
        try to match those unnatural patterns on real fish.
        """
        import cv2
        # Return image as-is - preserve natural gilt color signature
        # Models were trained with aggressive aug which makes them robust enough
        return image
    
    def predict_gill(self, gill_image: np.ndarray, include_glcm: bool = False) -> Optional[dict]:
        """
        Predict freshness from gill image using new hybrid model (expects 224x224x3 image input)
        Args:
            gill_image: Gill ROI image
            include_glcm: If True, include GLCM texture features in output
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        if self.gill_model is None:
            return None
        try:
            import cv2
            # Ensure image is 224x224 and 3 channels
            if gill_image.shape[:2] != (224, 224):
                gill_image_resized = cv2.resize(gill_image, (224, 224))
            else:
                gill_image_resized = gill_image
            if len(gill_image_resized.shape) == 2 or gill_image_resized.shape[2] == 1:
                gill_image_resized = np.stack([gill_image_resized]*3, axis=2)
            
            # No white balance - training didn't use it either
            # ROI should be BGR uint8, convert to RGB (KEEP 0-255 range)
            print("[DEBUG][GILL] Converting BGR ROI to RGB (no white balance)...")
            image = cv2.cvtColor(gill_image_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # DEBUG: Save preprocessed gill image
            import os
            debug_dir = './debug_gill_detector/preprocessed/'
            os.makedirs(debug_dir, exist_ok=True)
            
            # SAVE: Original extracted ROI (before any preprocessing)
            roi_path = os.path.join(debug_dir, 'gill_roi_ORIGINAL.png')
            # gill_image is BGR from extraction, save directly
            cv2.imwrite(roi_path, gill_image if len(gill_image.shape) == 3 else cv2.cvtColor(gill_image, cv2.COLOR_GRAY2BGR))
            print(f"[DEBUG][GILL] Saved original ROI to {roi_path} - this should be tight on gill tissue only")
            
            debug_img_path = os.path.join(debug_dir, 'gill_preprocessed.png')
            img_to_save = image.astype('uint8')
            # image is BGR (from apply_white_balance), so save directly without conversion
            cv2.imwrite(debug_img_path, img_to_save)
            print(f"[DEBUG][GILL] Saved preprocessed image to {debug_img_path}")
            
            # DEBUG: Also save as heatmap showing color distribution (R channel)
            r_channel = img_to_save[:, :, 2]  # Red is at index 2 in BGR
            heatmap = cv2.applyColorMap(r_channel, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug_dir, 'gill_red_heatmap.png'), heatmap)
            print(f"[DEBUG][GILL] Image color stats - R_mean:{img_to_save[:,:,2].mean():.1f}, G_mean:{img_to_save[:,:,1].mean():.1f}, B_mean:{img_to_save[:,:,0].mean():.1f}")
            
            # --- Extract features as in training ---
            resnet_feat = self.resnet_model.predict(np.expand_dims(self.preprocess_image_resnet(image), axis=0), verbose=0)[0]
            mobilenet_feat = self.mobilenet_model.predict(np.expand_dims(self.preprocess_image_mobilenet(image), axis=0), verbose=0)[0]
            # CRITICAL FIX: Compute GLCM from uint8 image (0-255), not float32 (0-1)
            glcm_dict = GLCMExtractor.compute_glcm_summary(img_to_save)
            glcm_feat = self._flatten_glcm_features(glcm_dict)
            cnn_features = np.concatenate([resnet_feat, mobilenet_feat])
            batch_cnn = np.expand_dims(cnn_features, axis=0)
            batch_glcm = np.expand_dims(glcm_feat, axis=0)
            # Debug: print feature stats
            print(f"[DEBUG][GILL] CNN features mean: {cnn_features.mean():.4f}, std: {cnn_features.std():.4f}, min: {cnn_features.min():.4f}, max: {cnn_features.max():.4f}")
            print(f"[DEBUG][GILL] GLCM features mean: {glcm_feat.mean():.4f}, std: {glcm_feat.std():.4f}, min: {glcm_feat.min():.4f}, max: {glcm_feat.max():.4f}, non-zero: {np.count_nonzero(glcm_feat)}/29")
            print(f"[DEBUG][GILL] ResNet50 specific: mean={resnet_feat.mean():.4f}, top5: {np.sort(resnet_feat)[-5:]}")
            print(f"[DEBUG][GILL] MobileNetV1 specific: mean={mobilenet_feat.mean():.4f}, top5: {np.sort(mobilenet_feat)[-5:]}")
            predictions = self.gill_model.predict([batch_cnn, batch_glcm], verbose=0)
            print(f"[DEBUG][GILL] Raw prediction probabilities: {predictions[0]}")
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            result = {
                'class': self.FRESHNESS_CLASSES[class_idx],
                'confidence': confidence,
                'probabilities': {
                    self.FRESHNESS_CLASSES[i]: float(predictions[0][i])
                    for i in range(len(self.FRESHNESS_CLASSES))
                }
            }
            if include_glcm:
                result['glcm_features'] = glcm_dict
            return result
        except Exception as e:
            print(f"[ERROR] Error predicting gill: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            return None
    
    def predict_eyes_gills(self, full_image: np.ndarray, include_glcm: bool = False) -> Optional[dict]:
        """
        Predict freshness from full fish image (eyes and gills integrated)
        
        Args:
            full_image: Full fish image or integrated ROI
            include_glcm: If True, include GLCM texture features in output
        
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        # Eyes+Gills model removed. This function is now deprecated.
        return None