
"""
Gills Detection Module
Detects and extracts gills from fish images using multi-method approach:
- HSV color-based detection (red/brown tissues)
- Edge-based detection (gill filament structures)
- Anatomical anchoring (gills near eye region)
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import os


class GillsDetector:
    """
    Detects and extracts gills from fish images using multiple methods:
    1. HSV color-based detection for red/maroon gill tissue
    2. Edge-based detection for gill filament structures
    3. Optional eye-region anchoring for anatomical accuracy
    """
    
    def __init__(self, debug_mode: bool = True, debug_dir: str = "debug_gill_detector",
                 min_gill_area: int = 2000, min_gill_width: int = 40, min_gill_height: int = 40):
        """
        Initialize detector with HSV color ranges optimized for gill detection
        
        Args:
            debug_mode: If True, saves debug images of detection process
            debug_dir: Directory to save debug images
            min_gill_area: Minimum pixel area for a valid gill region (rejects small noise)
            min_gill_width: Minimum bounding box width for a valid gill
            min_gill_height: Minimum bounding box height for a valid gill
        """
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.min_gill_area = min_gill_area
        self.min_gill_width = min_gill_width
        self.min_gill_height = min_gill_height
        
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[GILL] Debug mode enabled. Saving to {os.path.abspath(self.debug_dir)}")
            print(f"[GILL] Minimum gill area: {self.min_gill_area} px²")
            print(f"[GILL] Minimum gill dimensions: {self.min_gill_width}x{self.min_gill_height} px")
        
        # Improved HSV ranges for various gill colors across freshness levels
        # Deep red (fresh gills)
        self.lower_red_deep = np.array([0, 100, 60])
        self.upper_red_deep = np.array([12, 255, 255])
        
        # Pink/light red (slightly aged)
        self.lower_red_light = np.array([355, 80, 80])
        self.upper_red_light = np.array([5, 255, 255])
        
        # Brown (aged/less fresh)
        self.lower_brown = np.array([10, 40, 30])
        self.upper_brown = np.array([25, 200, 200])
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance gill detection and edge visibility"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def save_debug_image(self, image: np.ndarray, stage_name: str, is_mask: bool = False) -> None:
        """Save debug image if debug mode is enabled"""
        if not self.debug_mode or image is None:
            return
        
        debug_path = os.path.join(self.debug_dir, f"{stage_name}.png")
        if is_mask:
            # For masks, apply colormap for better visualization
            image_colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            cv2.imwrite(debug_path, image_colored)
        else:
            cv2.imwrite(debug_path, image)
        print(f"[GILL DEBUG] Saved: {debug_path}")
    
    def detect_gills_by_color(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect gills using HSV color-based masking.
        Targets red/maroon gill tissue across different freshness levels.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Broader HSV ranges to catch gills at all freshness levels
            # Gills range from bright red (fresh) to dark maroon (aged) to brownish (old)
            
            # Range 1: Pure red - hue 0-15 (bright to medium red)
            mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([15, 255, 255]))
            
            # Range 2: Dark red/maroon - hue 160-180 (wrapping around)
            # GOAL: Catch ALL gill freshness levels - bright maroon to very dark desaturated maroon
            # Fresh gills: sat 50+, value 40+
            # Less-fresh gills: sat 10+, value 19+ (very desaturated, dark)
            mask_red2 = cv2.inRange(hsv, np.array([160, 10, 19]), np.array([180, 255, 255]))
            
            # Range 3: Reddish-brown for aged gills - hue 8-25
            mask_red3 = cv2.inRange(hsv, np.array([8, 45, 30]), np.array([25, 220, 220]))
            
            # Save individual masks for debugging
            self.save_debug_image(mask_red1, "02a_mask_red1_pure_red_hue0-15", is_mask=True)
            
            # Clean mask_red2 before saving: remove noise dots, keep solid gill
            mask_red2_cleaned = mask_red2.copy()
            # Close small gaps within gill
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_red2_cleaned = cv2.morphologyEx(mask_red2_cleaned, cv2.MORPH_CLOSE, kernel_close)
            # Remove noise dots (open removes small scattered pixels)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_red2_cleaned = cv2.morphologyEx(mask_red2_cleaned, cv2.MORPH_OPEN, kernel_open)
            
            self.save_debug_image(mask_red2_cleaned, "02b_mask_red2_dark_maroon_hue160-180", is_mask=True)
            self.save_debug_image(mask_red3, "02c_mask_red3_reddish_brown_hue8-25", is_mask=True)
            
            # Create composite view of all three masks
            h, w = image.shape[:2]
            composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            mask_red1_colored = cv2.applyColorMap(mask_red1, cv2.COLORMAP_HOT)
            mask_red2_colored = cv2.applyColorMap(mask_red2_cleaned, cv2.COLORMAP_HOT)
            mask_red3_colored = cv2.applyColorMap(mask_red3, cv2.COLORMAP_HOT)
            
            composite[0:h, 0:w] = cv2.resize(mask_red1_colored, (w, h))
            composite[0:h, w:w*2] = cv2.resize(mask_red2_colored, (w, h))
            composite[h:h*2, 0:w] = cv2.resize(mask_red3_colored, (w, h))
            
            # Add text labels
            cv2.putText(composite, "Red (0-15)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(composite, "Maroon (160-180)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(composite, "Brown (8-25)", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.save_debug_image(composite, "02_redness_masks_composite")
            
            # Create saturation heatmap (shows how red/saturated each pixel is)
            saturation = hsv[:, :, 1]
            saturation_heatmap = cv2.applyColorMap(saturation, cv2.COLORMAP_JET)
            self.save_debug_image(saturation_heatmap, "02d_hsv_saturation_heatmap")
            
            # Create hue channel visualization (0=red, wraps around)
            hue = hsv[:, :, 0]
            hue_normalized = (hue * 255 / 180).astype(np.uint8)
            hue_heatmap = cv2.applyColorMap(hue_normalized, cv2.COLORMAP_HSV)
            self.save_debug_image(hue_heatmap, "02e_hsv_hue_channel")
            
            # Combine all red/maroon masks - ONLY use mask2!
            # Mask2 (hue 160-180) is the ACTUAL GILL DETECTOR
            # Do NOT add mask1 (hue 0-15) - it picks up fish SCALES, not gills!
            # Ignore mask3 (hue 8-25) - too noisy
            
            # Clean mask2: remove noise dots, keep solid gill
            mask_red = mask_red2.copy()
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)
            
            print(f"[GILL COLOR] Using ONLY mask2 (hue 160-180) - cleaned!")
            print(f"[GILL COLOR] NOT using mask1 (hue 0-15) - picks up fish scales")
            print(f"[GILL COLOR] NOT using mask3 (hue 8-25) - too noisy")
            
            self.save_debug_image(mask_red, "02f_mask_combined_before_morph", is_mask=True)
            
            # Use cleaned mask directly for contour detection
            mask = mask_red.copy()
            
            self.save_debug_image(mask, "02g_mask_combined_after_morph", is_mask=True)
            
            # Print pixel statistics
            red_pixels = np.sum(mask_red > 0)
            total_pixels = mask_red.shape[0] * mask_red.shape[1]
            print(f"[GILL COLOR] Red pixels detected: {red_pixels} / {total_pixels} ({100*red_pixels/total_pixels:.2f}%)")
            print(f"[GILL COLOR] Mask1 (hue 0-15) pixels: {np.sum(mask_red1 > 0)}")
            print(f"[GILL COLOR] Mask2 (hue 160-180) pixels: {np.sum(mask_red2 > 0)} *** PRIMARY GILL DETECTOR ***")
            print(f"[GILL COLOR] Mask3 (hue 8-25) pixels: {np.sum(mask_red3 > 0)}")
            print(f"[GILL COLOR] After morphology: {np.sum(mask > 0)} pixels remain")
            
            return mask
        except Exception as e:
            print(f"[GILL COLOR] Error in color detection: {e}")
            return None
    
    def detect_gills_by_desaturated_color(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback detector for STARTING_TO_ROT and ROTTEN gills.
        Uses HSV ranges CALIBRATED from 23 starting_to_rot and 8 rotten manual selections.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            print(f"[GILL DESATURATED] Fallback detector for aged/rotten gills...")
            
            # STARTING_TO_ROT: Measured from 23 manual selections (Q10-Q90 percentiles)
            # Multi-colored, partially oxidized gill tissue
            # H: 0-177, S: 48-158, V: 33-70
            mask_starting_rot = cv2.inRange(hsv, np.array([0, 48, 33]), np.array([177, 158, 70]))
            
            # ROTTEN: Measured from 8 manual selections (Q10-Q90 percentiles)
            # Dark, heavily oxidized gill tissue (narrow hue range)
            # H: 5-13, S: 63-187, V: 29-60
            mask_rotten = cv2.inRange(hsv, np.array([5, 63, 29]), np.array([13, 187, 60]))
            
            self.save_debug_image(mask_starting_rot, "02a_fallback_starting_rot_calibrated", is_mask=True)
            self.save_debug_image(mask_rotten, "02b_fallback_rotten_calibrated", is_mask=True)
            
            # Combine both masks
            mask_desaturated = cv2.bitwise_or(mask_starting_rot, mask_rotten)
            
            self.save_debug_image(mask_desaturated, "02c_fallback_combined_mask", is_mask=True)
            
            # Apply morphological cleanup
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_desaturated = cv2.morphologyEx(mask_desaturated, cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_desaturated = cv2.morphologyEx(mask_desaturated, cv2.MORPH_OPEN, kernel_open)
            
            desaturated_pixels = np.sum(mask_desaturated > 0)
            starting_rot_pixels = np.sum(mask_starting_rot > 0)
            rotten_pixels = np.sum(mask_rotten > 0)
            total_pixels = mask_desaturated.shape[0] * mask_desaturated.shape[1]
            
            print(f"[GILL DESATURATED] Starting_to_rot (H:0-177, S:48-158, V:33-70): {starting_rot_pixels} pixels")
            print(f"[GILL DESATURATED] Rotten (H:5-13, S:63-187, V:29-60): {rotten_pixels} pixels")
            print(f"[GILL DESATURATED] Combined desaturated pixels: {desaturated_pixels} / {total_pixels} "
                  f"({100*desaturated_pixels/total_pixels:.2f}%)")
            
            return mask_desaturated
        except Exception as e:
            print(f"[GILL DESATURATED] Error in desaturated color detection: {e}")
            return None
    
    def detect_gills_by_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Edge-based supplementary detection for gill filament structures.
        Used as fallback when color detection is weak.
        """
        try:
            # Convert to grayscale and enhance edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection with moderate thresholds
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to make them more prominent
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            return dilated
        except Exception as e:
            print(f"[GILL EDGES] Error in edge detection: {e}")
            return None
    
    def detect_head_side(self, image: np.ndarray) -> str:
        """Determine if fish head is on left or right side based on darkness."""
        h, w = image.shape[:2]
        left_region = image[:, :w//2]
        right_region = image[:, w//2:]
        
        left_gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
        
        _, left_dark = cv2.threshold(left_gray, 60, 255, cv2.THRESH_BINARY_INV)
        _, right_dark = cv2.threshold(right_gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        left_score = np.sum(left_dark)
        right_score = np.sum(right_dark)
        
        return "left" if left_score > right_score else "right"
    
    def select_best_contour(self, contours, image_shape, head_side: str = None, 
                           eye_box: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Select the best contour by finding the LARGEST red area that meets minimum size thresholds.
        ALSO validates that gill is appropriately sized relative to eye and positioned near it.
        
        Anatomical constraints:
        - Gill box max width/height: 4x the eye box dimensions
        - Gill position: must be within reasonable distance of eye location
        """
        if not contours:
            print("[GILL SELECTION] No contours to select from")
            return None
        
        h, w = image_shape[:2]
        best_gill = None
        best_area = 0
        rejected_count = 0
        
        print(f"\n[GILL SELECTION] Finding red areas with minimum area={self.min_gill_area}px², "
              f"dimensions >= {self.min_gill_width}x{self.min_gill_height}px")
        
        # Print eye box info if available
        if eye_box:
            eye_x, eye_y, eye_w, eye_h = eye_box
            print(f"[GILL SELECTION] Eye box: ({eye_x}, {eye_y}, {eye_w}x{eye_h})")
            print(f"[GILL SELECTION] Max gill dimensions: {eye_w*4}x{eye_h*4} (4x eye size)")
            print(f"[GILL SELECTION] Gills must be adjacent to eye region\n")
        
        # Find the LARGEST valid contour by area - that's the gill!
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            
            # Check size criteria
            size_valid = area >= self.min_gill_area and w_box >= self.min_gill_width and h_box >= self.min_gill_height
            aspect_valid = not (aspect_ratio > 5.0 or aspect_ratio < 0.15)
            eye_size_valid = True  # default to valid if no eye box
            eye_position_valid = True  # default to valid if no eye box
            
            reason = ""
            
            # Reject if too small for a gill region
            if area < self.min_gill_area:
                reason = f"Area {area} < minimum {self.min_gill_area}"
                rejected_count += 1
            
            # Reject if bounding box too small
            elif w_box < self.min_gill_width or h_box < self.min_gill_height:
                reason = f"Dimensions {w_box}x{h_box} < minimum {self.min_gill_width}x{self.min_gill_height}"
                rejected_count += 1
            
            # Reject if too elongated/weird aspect ratio
            elif aspect_ratio > 5.0 or aspect_ratio < 0.15:
                reason = f"Bad aspect ratio {aspect_ratio:.2f} (must be 0.15-5.0)"
                rejected_count += 1
            
            # ANATOMICAL VALIDATION: Check against eye box if available
            elif eye_box:
                eye_x, eye_y, eye_w, eye_h = eye_box
                
                # Constraint 1: Gill box should be at most 4x the eye box dimensions
                max_gill_width = eye_w * 4
                max_gill_height = eye_h * 4
                
                if w_box > max_gill_width or h_box > max_gill_height:
                    reason = f"Gill size {w_box}x{h_box} exceeds max {max_gill_width}x{max_gill_height} (4x eye)"
                    eye_size_valid = False
                    rejected_count += 1
                
                # Constraint 2: Gills must be adjacent to eye (not far away)
                # Expected: Gill box slightly overlaps or is immediately next to eye
                # Calculate center of both regions
                gill_center_x = x + w_box // 2
                gill_center_y = y + h_box // 2
                eye_center_x = eye_x + eye_w // 2
                eye_center_y = eye_y + eye_h // 2
                
                # Distance tolerance: gills should be within ~1.5x of eye size away
                distance_tolerance = max(eye_w, eye_h) * 2.5
                distance = np.sqrt((gill_center_x - eye_center_x)**2 + (gill_center_y - eye_center_y)**2)
                
                if not eye_size_valid:
                    pass  # Already rejected above
                elif distance > distance_tolerance:
                    reason = f"Gill center at ({gill_center_x}, {gill_center_y}) too far from eye center ({eye_center_x}, {eye_center_y}), distance={distance:.0f} > {distance_tolerance:.0f}"
                    eye_position_valid = False
                    rejected_count += 1
            
            status = "✓" if (size_valid and aspect_valid and eye_size_valid and eye_position_valid and not reason) else "✗"
            print(f"  Contour {i:3d}: area={area:7.0f}, bbox=({x:4d},{y:4d},{w_box:4d},{h_box:4d}), "
                  f"aspect={aspect_ratio:.2f} {status}")
            
            if reason:
                print(f"           → REJECTED: {reason}")
                continue
            
            # This is a valid candidate - is it bigger than our current best?
            if area > best_area:
                best_area = area
                best_gill = (x, y, w_box, h_box)
                print(f"           → NEW BEST!")
        
        if best_gill:
            print(f"\n[GILL SELECTION] ✓ FOUND VALID GILL: {best_gill} (area {best_area:.0f})")
            print(f"[GILL SELECTION] Rejected {rejected_count} contours (size/aspect/anatomy failed)\n")
        else:
            print(f"\n[GILL SELECTION] ✗ NO VALID GILL FOUND")
            print(f"[GILL SELECTION] All {len(contours)} contours were rejected\n")
        
        return best_gill
    
    def detect_gills(self, image: np.ndarray, eye_box: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Main gill detection method combining color and edge-based approaches.
        
        Args:
            image: Input fish image
            eye_box: Optional eye bounding box (x, y, w, h) for anatomical anchoring
        
        Returns:
            Bounding box (x, y, w, h) of detected gills or None
        """
        try:
            h, w = image.shape[:2]
            print(f"[GILL] Original image size: {w}x{h}")
            
            # Save original image
            self.save_debug_image(image, "01_original_image")
            
            # Detect head side
            head_side = self.detect_head_side(image)
            print(f"[GILL] Detected head side: {head_side}")
            
            # Method 1: Color-based detection
            color_mask = self.detect_gills_by_color(image)
            self.save_debug_image(color_mask, "02_color_mask", is_mask=True)
            
            # Method 2: Edge-based detection
            edge_mask = self.detect_gills_by_edges(image)
            self.save_debug_image(edge_mask, "03_edge_mask", is_mask=True)
            
            # Combine masks: color detection is primary, edges supplement where color is weak
            if color_mask is not None and np.sum(color_mask) > 0:
                combined_mask = color_mask
                # If color detection is too weak, reinforce with edges
                if np.sum(color_mask) < 500:  # Very few pixels detected
                    if edge_mask is not None:
                        combined_mask = cv2.bitwise_or(color_mask, edge_mask // 3)
                        print("[GILL] Color mask weak - reinforcing with edges")
            elif edge_mask is not None:
                combined_mask = edge_mask
                print("[GILL] Using edge detection as primary method")
            else:
                print("[GILL] Failed to generate detection masks")
                return None
            
            self.save_debug_image(combined_mask, "04_combined_mask", is_mask=True)
            
            # Find contours in combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[GILL] Found {len(contours)} contours in mask")
            
            if len(contours) == 0:
                print("[GILL] No contours found - color mask may be too strict")
                # Save mask for inspection
                self.save_debug_image(combined_mask, "04_combined_mask_EMPTY", is_mask=True)
            else:
                self.save_debug_image(combined_mask, "04_combined_mask", is_mask=True)
            
            # Select best contour
            gill_box = self.select_best_contour(contours, image.shape, head_side, eye_box)
            
            # FALLBACK: If primary detection failed, try desaturated color detector for aged/rotten gills
            if not gill_box:
                print("\n[GILL] ⚠ PRIMARY DETECTION FAILED - Starting fallback for AGED/ROTTEN gills...")
                desaturated_mask = self.detect_gills_by_desaturated_color(image)
                
                if desaturated_mask is not None and np.sum(desaturated_mask) > 0:
                    # Find contours in fallback mask
                    fallback_contours, _ = cv2.findContours(desaturated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    print(f"[GILL FALLBACK] Found {len(fallback_contours)} contours in desaturated mask")
                    
                    if len(fallback_contours) > 0:
                        # Try to find valid gill in fallback contours
                        gill_box = self.select_best_contour(fallback_contours, image.shape, head_side, eye_box)
                        if gill_box:
                            print(f"[GILL FALLBACK] ✓ Fallback successful! Gill detected via aged/rotten detector")
                        else:
                            print(f"[GILL FALLBACK] ✗ Fallback contours too small - no valid gill found")
                    else:
                        print(f"[GILL FALLBACK] ✗ No contours in fallback mask")
                else:
                    print(f"[GILL FALLBACK] ✗ No desaturated regions detected")
            
            # Save final detection
            if gill_box:
                final_image = image.copy()
                x, y, w_box, h_box = gill_box
                cv2.rectangle(final_image, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)
                cv2.putText(final_image, f"Gill Area: {w_box*h_box}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.save_debug_image(final_image, "06_final_detection_bbox")
                print(f"[GILL] Detected gill at {gill_box}")
            else:
                self.save_debug_image(image, "06_final_detection_NONE")
                print("[GILL] No valid gill contour found (both primary and fallback failed)")
            
            return gill_box
            
        except Exception as e:
            print(f"[GILL] ERROR in detect_gills: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_gill_roi(self, image: np.ndarray, gill_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract gill region of interest from image.
        Tight crop on just the gill tissue - minimal padding!
        
        Args:
            image: Input image
            gill_rect: Bounding box (x, y, w, h)
        
        Returns:
            Cropped and resized gill region (224x224) or None
        """
        try:
            x, y, w, h = gill_rect
            
            # TIGHT CROP: Minimal padding to match training data (training gills fill the frame)
            # Use negative padding (inset) to crop tighter on just the gill tissue
            padding = int(max(w, h) * -0.05)  # Negative = inset/crop tighter
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Resize to standard size (224x224 for model input)
            roi_resized = cv2.resize(roi, (224, 224))
            print(f"[GILL ROI] Extracted tight crop: bbox=({x1},{y1},{x2-x1},{y2-y1}) -> resized to (224,224)")
            return roi_resized
        except Exception as e:
            print(f"[GILL ROI] Error extracting ROI: {e}")
            return None
    
    def detect_and_extract(self, image: np.ndarray, eye_box: Tuple[int, int, int, int] = None) -> dict:
        """
        Complete gill detection and extraction pipeline.
        
        Args:
            image: Input fish image
            eye_box: Optional eye bounding box for anatomical anchoring
        
        Returns:
            Dictionary with detection results
        """
        try:
            result = {
                'gill': None,
                'gill_bbox': None,
                'gill_detected': False
            }
            
            print(f"\n[GILLS DETECTOR] ===== NEW DETECTION REQUEST =====")
            
            # Detect gills
            gill_rect = self.detect_gills(image, eye_box)
            if gill_rect is not None:
                print(f"[GILLS DETECTOR] Gill detected!")
                result['gill_bbox'] = gill_rect
                result['gill'] = self.extract_gill_roi(image, gill_rect)
                result['gill_detected'] = result['gill'] is not None
            else:
                print(f"[GILLS DETECTOR] No gill detected")
            
            print(f"[GILLS DETECTOR] Final: gill={result['gill_detected']}")
            return result
        except Exception as e:
            print(f"[GILLS DETECTOR] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'gill': None,
                'gill_bbox': None,
                'gill_detected': False
            }
