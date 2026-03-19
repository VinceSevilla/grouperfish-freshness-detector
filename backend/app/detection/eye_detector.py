import cv2
import numpy as np
from typing import List, Tuple

class EyeDetector:
    """Detects and extracts eyes from fish images"""
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance eye detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def detect_eyes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Single-stage: Use Cb channel dark circle detection (pupil-based, robust across all freshness levels)"""
        try:
            original_h, original_w = image.shape[:2]
            print(f"[EYE] Original image size: {original_w}x{original_h}")
            
            # Downsample for speed
            max_detection_size = 1280
            if max(original_w, original_h) > max_detection_size:
                scale_down = max_detection_size / max(original_w, original_h)
                detect_w = int(original_w * scale_down)
                detect_h = int(original_h * scale_down)
                img_small = cv2.resize(image, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
            else:
                img_small = image.copy()
                detect_w, detect_h = original_w, original_h
                scale_down = 1.0
            
            # Detect eye using Cb channel (dark pupil detection)
            eye_center, eye_radius = self.detect_eye_cb_channel(img_small)
            
            if eye_center is None or eye_radius is None:
                print("[EYE] No eye detected via Cb channel")
                return []
            
            cx, cy = eye_center
            radius = eye_radius
            print(f"[EYE] Detected eye at ({cx}, {cy}), radius={radius}")
            
            # Create bbox centered on detected eye with 4:3 aspect ratio
            box_size = int(radius * 5.5)
            bbox_h = box_size
            bbox_w = int(bbox_h * 4 / 3)  # width = 4/3 of height for 4:3 ratio
            bbox_x = max(0, cx - bbox_w // 2)
            bbox_y = max(0, cy - bbox_h // 2)
            bbox_w = min(detect_w - bbox_x, bbox_w)
            bbox_h = min(detect_h - bbox_y, bbox_h)
            
            # Scale back to original image size
            scale_up = 1.0 / scale_down
            bbox_x_orig = int(bbox_x * scale_up)
            bbox_y_orig = int(bbox_y * scale_up)
            bbox_w_orig = int(bbox_w * scale_up)
            bbox_h_orig = int(bbox_h * scale_up)
            print(f"[EYE] Scaled to original: bbox=({bbox_x_orig}, {bbox_y_orig}, {bbox_w_orig}x{bbox_h_orig})")
            
            return [(bbox_x_orig, bbox_y_orig, bbox_w_orig, bbox_h_orig)]
            
        except Exception as e:
            print(f"[EYE] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_eye_cb_channel(self, image: np.ndarray) -> Tuple:
        """Detect eye from Cb channel dark circle detection
        
        Returns:
            Tuple of (center, radius) where center=(cx, cy) and radius=r
            or (None, None) if no eye found
        """
        try:
            # Convert BGR to YCbCr and extract Cb channel
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            cb_channel = ycbcr[:, :, 1]
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cb_enhanced = clahe.apply(cb_channel)
            
            # Create binary mask of dark regions (darkness < 110) - pupils are very dark
            dark_mask = cv2.threshold(cb_enhanced, 110, 255, cv2.THRESH_BINARY_INV)[1]
            
            # Detect contours in dark mask
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[CB_EYE] Found {len(contours)} contours")
            
            candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Skip small noise
                    continue
                
                # Fit circle to contour
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                if radius < 10:  # Minimum eye radius
                    continue
                
                # Score: prefer circular and reasonably sized
                circularity = area / (np.pi * radius * radius) if radius > 0 else 0
                
                if circularity < 0.5:  # Must be reasonably circular
                    continue
                
                score = (circularity * 100) + (area / 100)
                candidates.append({
                    'cx': cx, 'cy': cy, 'radius': radius,
                    'area': area, 'circularity': circularity,
                    'score': score
                })
                print(f"[CB_EYE] Candidate at ({cx:.0f}, {cy:.0f}), r={radius:.0f}, circularity={circularity:.2f}, score={score:.1f}")
            
            # Return best candidate
            if candidates:
                candidates.sort(key=lambda c: c['score'], reverse=True)
                best = candidates[0]
                print(f"[CB_EYE] Best eye at ({best['cx']:.0f}, {best['cy']:.0f}), radius={best['radius']:.0f}")
                return ((int(best['cx']), int(best['cy'])), int(best['radius']))
            else:
                print("[CB_EYE] No suitable circles found")
                return (None, None)
                
        except Exception as e:
            print(f"[CB_EYE] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None, None)

    def extract_eye_roi(self, image: np.ndarray, eye_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract square eye ROI to avoid aspect ratio distortion during resize"""
        x, y, w, h = eye_rect
        # Calculate center of detection bbox
        cx = x + w // 2
        cy = y + h // 2
        
        # Extract SQUARE crop (1.5x the detected size to get context)
        size = int(max(w, h) * 1.5)
        half_size = size // 2
        
        # Clip to image bounds
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(image.shape[1], cx + half_size)
        y2 = min(image.shape[0], cy + half_size)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Resize to 224x224 (now square crop, no distortion)
        roi_resized = cv2.resize(roi, (224, 224))
        # Convert from BGR to RGB for model input
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        return roi_rgb

    def apply_white_balance(self, img):
        """Simple white balance using Gray World Assumption"""
        import cv2
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def convert_to_cb_channel(self, image: np.ndarray, eye_region: Tuple[int, int, int, int] = None, return_ycbcr: bool = False, save_debug: bool = True) -> np.ndarray:
        """Convert image to Cb channel (chrominance blue) for debugging color information
        
        Args:
            image: Input BGR image
            eye_region: (x, y, w, h) ROI to extract Cb channel for circle detection
            return_ycbcr: If True, return full YCbCr image; if False, return only Cb channel
            save_debug: If True, save enhanced Cb channel with detected eye circles as debug image
        
        Returns:
            Cb channel (single channel) or full YCbCr image depending on return_ycbcr flag
        """
        try:
            # Convert BGR to YCbCr
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            if return_ycbcr:
                return ycbcr
            else:
                # Extract only Cb channel (index 1 in YCbCr/YCrCb)
                cb_channel = ycbcr[:, :, 1]
                print(f"[CB_CHANNEL] Extracted Cb channel, shape: {cb_channel.shape}, range: {cb_channel.min()}-{cb_channel.max()}")
                
                # Save debug image if requested
                if save_debug:
                    import os
                    debug_dir = './debug_eye_detector'
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Keep channel as-is (don't invert) - pupil is naturally dark
                    # Enhance contrast using CLAHE
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cb_enhanced = clahe.apply(cb_channel)
                    
                    # Extract eye region if provided
                    if eye_region is not None:
                        x, y, w, h = eye_region
                        # Expand region a bit for better detection
                        expand = int(max(w, h) * 0.3)
                        x1 = max(0, x - expand)
                        y1 = max(0, y - expand)
                        x2 = min(cb_enhanced.shape[1], x + w + expand)
                        y2 = min(cb_enhanced.shape[0], y + h + expand)
                        cb_roi = cb_enhanced[y1:y2, x1:x2]
                        print(f"[CB_CHANNEL] Extracted eye region: ({x1}, {y1}) to ({x2}, {y2})")
                    else:
                        cb_roi = cb_enhanced
                        x1, y1 = 0, 0
                    
                    # Detect dark circles in the ROI
                    h_roi, w_roi = cb_roi.shape[:2]
                    min_eye_radius = int(0.02 * w_roi)
                    max_eye_radius = int(0.15 * w_roi)
                    
                    # Apply Canny edge detection with STRICT thresholds (only strong edges = circular boundaries)
                    edges = cv2.Canny(cb_roi, 100, 200)
                    
                    # Save edge map for debugging
                    edge_path = f"{debug_dir}/08_cb_edges.png"
                    cv2.imwrite(edge_path, edges)
                    print(f"[CB_CHANNEL] Saved edge map to {edge_path}")
                    
                    # Create binary mask of dark regions (darkness < 110) from FULL image - only very dark regions (pupils)
                    dark_mask_full = cv2.threshold(cb_enhanced, 110, 255, cv2.THRESH_BINARY_INV)[1]
                    
                    # Save masked image showing dark regions in FULL image
                    mask_path = f"{debug_dir}/07_cb_masked.png"
                    cv2.imwrite(mask_path, dark_mask_full)
                    print(f"[CB_CHANNEL] Saved dark mask to {mask_path}")
                    
                    # Detect circles directly from the FULL dark mask
                    contours, _ = cv2.findContours(dark_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    print(f"[CB_CHANNEL] Found {len(contours)} contours in full dark mask")
                    
                    # Create marked image using full enhanced Cb channel (convert to BGR for visualization)
                    cb_marked = cv2.cvtColor(cb_enhanced, cv2.COLOR_GRAY2BGR)
                    
                    best_circle = None
                    best_score = -float('inf')
                    
                    # Score based ONLY on Cb channel characteristics (independent of iris detection)
                    candidates = []  # Store all candidates
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 200:  # Skip small noise - eyes have significant area
                            continue
                        
                        # Fit circle to contour
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        
                        if radius < 10:  # Minimum eye radius
                            continue
                        
                        # Score: circularity is the primary criterion
                        circularity = area / (np.pi * radius * radius) if radius > 0 else 0
                        
                        # Only consider reasonably circular contours
                        if circularity < 0.5:  # Must be at least 50% circular
                            continue
                        
                        # Score based purely on shape quality and size (independent of iris)
                        score = (circularity * 100) + (area / 100)  # Prefer circular and reasonably sized
                        
                        candidates.append({
                            'cx': cx, 'cy': cy, 'radius': radius,
                            'area': area, 'circularity': circularity,
                            'score': score
                        })
                        
                        print(f"[CB_CHANNEL] Candidate at ({cx:.0f}, {cy:.0f}), r={radius:.0f}, area={area:.0f}, circularity={circularity:.2f}, score={score:.1f}")
                    
                    # Pick best candidate based on score
                    best_circle = None
                    best_score = -float('inf')
                    if candidates:
                        candidates.sort(key=lambda c: c['score'], reverse=True)
                        best = candidates[0]
                        best_circle = (int(best['cx']), int(best['cy']), int(best['radius']))
                        best_score = best['score']
                        print(f"[CB_CHANNEL] Best candidate: ({best['cx']:.0f}, {best['cy']:.0f}), r={best['radius']:.0f}, circularity={best['circularity']:.2f}, score={best['score']:.1f}")
                    
                    # Draw the best circle on full image
                    if best_circle is not None:
                        cx, cy, radius = best_circle
                        cv2.circle(cb_marked, (cx, cy), radius, (255, 0, 0), 3)
                        print(f"[CB_CHANNEL] Selected eye at ({cx}, {cy}), radius={radius}, score={best_score:.0f}")
                    else:
                        print(f"[CB_CHANNEL] No suitable circles found in dark regions")
                    
                    cb_path = f"{debug_dir}/06_cb_channel.png"
                    cv2.imwrite(cb_path, cb_marked)
                    print(f"[CB_CHANNEL] Saved Cb channel to {cb_path}")
                
                return cb_channel
        except Exception as e:
            print(f"[CB_CHANNEL] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def detect_and_extract(self, image: np.ndarray) -> dict:
        """Detect and extract eye region"""
        try:
            result = {
                'eye': None,
                'eye_bbox': None,
                'eye_detected': False
            }
            print(f"\n[EYE DETECTOR] ===== NEW DETECTION REQUEST =====")
            eyes = self.detect_eyes(image)
            if len(eyes) > 0:
                print(f"[EYE DETECTOR] Eye detected!")
                eye = max(eyes, key=lambda e: e[2] * e[3])
                result['eye_bbox'] = tuple(eye)
                result['eye'] = self.extract_eye_roi(image, eye)
                result['eye_detected'] = result['eye'] is not None
            else:
                print(f"[EYE DETECTOR] No eye detected")
            print(f"[EYE DETECTOR] Final: eye={result['eye_detected']}")
            return result
        except Exception as e:
            print(f"[EYE DETECTOR] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'eye': None,
                'eye_bbox': None,
                'eye_detected': False
            }
