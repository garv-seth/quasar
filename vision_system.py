"""
Computer Vision System for Autonomous Agent

This module provides screen understanding capabilities for true agentic behavior,
allowing the agent to "see" and understand UI elements and make decisions based on
visual information.
"""

import os
import base64
import logging
import json
import time
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-system")

# Optional dependencies with proper error handling
try:
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Computer vision libraries not available. Using fallback mechanisms.")

class VisionSystem:
    """
    Computer vision system for screen understanding and UI element detection
    
    This component allows the agent to visually analyze interfaces, identify
    UI elements, and make decisions based on visual context.
    """
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize the vision system
        
        Args:
            use_ocr: Whether to use OCR for text recognition
        """
        self.use_ocr = use_ocr and VISION_AVAILABLE
        self.templates = self._load_templates()
        
        # Metrics
        self.processed_images = 0
        self.detected_elements = 0
        self.ocr_operations = 0
        
        # Status tracking
        self.system_ready = VISION_AVAILABLE
        self.last_processing_time = 0
        
        # In a real implementation, we would initialize OCR and element detection models here
        
    def _load_templates(self):
        """Load element detection templates"""
        # In a real implementation, these would be a comprehensive set of UI element templates
        templates = {
            "button": {
                "patterns": ["btn", "button", "submit", "ok", "cancel", "next", "prev", "previous", "close"],
                "css_selectors": ["button", ".btn", "input[type='button']", "input[type='submit']"],
                "visual_features": {
                    "aspect_ratio_range": [1.5, 5.0],  # width/height
                    "min_size": 20,  # min height in pixels
                    "max_size": 100,  # max height in pixels
                    "typical_colors": ["#007bff", "#28a745", "#dc3545", "#ffc107", "#17a2b8", "#6c757d", "#343a40"]
                }
            },
            "input": {
                "patterns": ["input", "textbox", "field", "text field", "form field"],
                "css_selectors": ["input[type='text']", "input[type='password']", "input[type='email']", "textarea"],
                "visual_features": {
                    "aspect_ratio_range": [3.0, 10.0],
                    "min_size": 20,
                    "max_size": 60,
                    "typical_colors": ["#ffffff", "#f8f9fa", "#e9ecef"]
                }
            },
            "checkbox": {
                "patterns": ["checkbox", "check", "option"],
                "css_selectors": ["input[type='checkbox']"],
                "visual_features": {
                    "aspect_ratio_range": [0.8, 1.2],
                    "min_size": 10,
                    "max_size": 30,
                    "typical_colors": ["#ffffff", "#f8f9fa"]
                }
            },
            "dropdown": {
                "patterns": ["dropdown", "select", "menu", "combobox"],
                "css_selectors": ["select", ".dropdown"],
                "visual_features": {
                    "aspect_ratio_range": [2.0, 8.0],
                    "min_size": 20,
                    "max_size": 60,
                    "typical_colors": ["#ffffff", "#f8f9fa", "#e9ecef"]
                }
            },
            "link": {
                "patterns": ["link", "hyperlink", "a href"],
                "css_selectors": ["a", ".link"],
                "visual_features": {
                    "aspect_ratio_range": [1.0, 20.0],
                    "min_size": 10,
                    "max_size": 50,
                    "typical_colors": ["#007bff", "#0056b3", "#6610f2"]
                }
            },
            "image": {
                "patterns": ["image", "img", "picture", "photo"],
                "css_selectors": ["img", ".image", "picture"],
                "visual_features": {
                    "aspect_ratio_range": [0.5, 2.0],
                    "min_size": 30,
                    "max_size": 800,
                    "typical_colors": []  # Any colors
                }
            }
        }
        
        return templates
        
    def analyze_screenshot(self, screenshot_base64: str) -> Dict[str, Any]:
        """
        Analyze a screenshot to detect UI elements and understand the interface
        
        Args:
            screenshot_base64: Base64 encoded screenshot image
            
        Returns:
            Dict with detected elements and analysis
        """
        start_time = time.time()
        
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Computer vision libraries not available",
                "elements": [],
                "text": "Vision processing unavailable",
                "processing_time": 0
            }
            
        try:
            # Decode the base64 image
            image_data = base64.b64decode(screenshot_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to numpy array for OpenCV
            image_np = np.array(image)
            
            # Process the image to detect UI elements
            elements = self._detect_ui_elements(image_np)
            
            # Create visualization
            visualization_base64 = self._create_visualization(image, elements)
            
            # Extract text if OCR is enabled
            text = "OCR not enabled"
            if self.use_ocr:
                text = self._extract_text(image_np)
                self.ocr_operations += 1
                
            # Update metrics
            self.processed_images += 1
            self.detected_elements += len(elements)
            self.last_processing_time = time.time() - start_time
            
            return {
                "success": True,
                "elements": elements,
                "element_count": len(elements),
                "text": text,
                "visualization": visualization_base64,
                "processing_time": self.last_processing_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "elements": [],
                "text": "Error processing image",
                "processing_time": time.time() - start_time
            }
            
    def _detect_ui_elements(self, image) -> List[Dict[str, Any]]:
        """
        Detect UI elements in the image
        
        In a real implementation, this would use more sophisticated models
        for accurate UI element detection. This is a simplified placeholder.
        """
        elements = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours - a simple approach to locate potential UI elements
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        element_id = 0
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small or very large elements
            if w < 10 or h < 10 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue
                
            # Analyze the element area to determine its type
            element_type = self._classify_element(image, x, y, w, h)
            
            # Add to the elements list
            element_id += 1
            elements.append({
                "id": f"element_{element_id}",
                "type": element_type,
                "confidence": 0.7,  # Placeholder confidence value
                "bounds": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "center": {
                    "x": int(x + w/2),
                    "y": int(y + h/2)
                }
            })
            
        return elements
        
    def _classify_element(self, image, x, y, w, h) -> str:
        """
        Classify a UI element based on visual characteristics
        
        In a real implementation, this would use CNN-based classification
        or template matching with a comprehensive set of UI element templates.
        This is a simplified placeholder.
        """
        # Get aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Simple classification based on aspect ratio and size
        if 0.8 <= aspect_ratio <= 1.2 and 10 <= h <= 30:
            return "checkbox"
        elif aspect_ratio >= 3.0 and 20 <= h <= 60:
            return "input"
        elif 1.5 <= aspect_ratio <= 5.0 and 20 <= h <= 60:
            return "button"
        elif aspect_ratio >= 2.0 and h < 60:
            return "dropdown"
        elif 0.5 <= aspect_ratio <= 2.0 and h >= 50:
            return "image"
        else:
            return "unknown"
            
    def _extract_text(self, image) -> str:
        """
        Extract text from the image using OCR
        
        In a real implementation, this would use a proper OCR engine like
        Tesseract or a more advanced model. This is a simplified placeholder.
        """
        # Placeholder OCR implementation
        text = "Sample text detected in the interface"
        
        # In a real implementation, we would use OCR to extract text:
        # Example with tesseract:
        # text = pytesseract.image_to_string(image)
        
        return text
        
    def _create_visualization(self, image, elements) -> str:
        """Create a visualization of detected UI elements"""
        # Create a copy of the image to draw on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Draw rectangles around detected elements
        for element in elements:
            bounds = element["bounds"]
            x, y, w, h = bounds["x"], bounds["y"], bounds["width"], bounds["height"]
            
            # Choose color based on element type
            color_map = {
                "button": (0, 255, 0),    # Green
                "input": (255, 0, 0),     # Red
                "checkbox": (0, 0, 255),  # Blue
                "dropdown": (255, 255, 0), # Yellow
                "link": (255, 0, 255),    # Magenta
                "image": (0, 255, 255),   # Cyan
                "unknown": (128, 128, 128) # Gray
            }
            
            color = color_map.get(element["type"], (128, 128, 128))
            
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
            
            # Draw element type label
            draw.text((x, y - 10), element["type"], fill=color)
            
        # Convert back to base64
        buffer = BytesIO()
        draw_image.save(buffer, format="PNG")
        visualization_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return visualization_base64
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the vision system"""
        return {
            "processed_images": self.processed_images,
            "detected_elements": self.detected_elements,
            "ocr_operations": self.ocr_operations,
            "average_processing_time": (self.last_processing_time if self.processed_images == 1 else
                                     self.last_processing_time / self.processed_images if self.processed_images > 0 else 0),
            "system_ready": self.system_ready,
            "vision_libraries_available": VISION_AVAILABLE
        }
        
    def get_element_at_coordinates(self, elements: List[Dict[str, Any]], x: int, y: int) -> Optional[Dict[str, Any]]:
        """Find the UI element at the given screen coordinates"""
        for element in elements:
            bounds = element["bounds"]
            if (bounds["x"] <= x <= bounds["x"] + bounds["width"] and
                bounds["y"] <= y <= bounds["y"] + bounds["height"]):
                return element
        return None
        
    def get_elements_by_type(self, elements: List[Dict[str, Any]], element_type: str) -> List[Dict[str, Any]]:
        """Get all elements of a specific type"""
        return [e for e in elements if e["type"] == element_type]