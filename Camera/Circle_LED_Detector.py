import numpy as np
import cv2
import time
import json
from datetime import datetime

# ================================
# CONFIGURABLE PARAMETERS
# ================================
CONFIG = {
    # Camera settings (fixed - no need for real-time changes)
    "camera_ip": "169.254.160.162",
    "camera_port": 8554,
    "camera_username": "fgcam",
    "camera_password": "admin",
    "camera_stream_path": "/0/unicast",
    
    # Real-time adjustable parameters
    "binary_threshold": 255,  # Adjustable with +/- keys
    "reference_x": 770,       # Adjustable with arrow keys
    "reference_y": 310,       # Adjustable with arrow keys
    "min_radius": 1,          # Adjustable with 1/2 keys
    "max_radius": 10,         # Adjustable with 3/4 keys
    "roi_x_min": 150,         # Adjustable with u/i/o/p keys
    "roi_x_max": 1100,        # Adjustable with u/i/o/p keys
    "roi_y_min": 280,         # Adjustable with j/k/l/; keys
    "roi_y_max": 700,         # Adjustable with j/k/l/; keys
    
    # HoughCircles parameters (adjustable with z/x/c/v/b/n keys)
    "hough_dp": 1.0,          # Resolution ratio (z/x keys, step 0.2)
    "hough_min_dist": 20,     # Min distance between circles (c/v keys, step 5)
    "hough_param1": 50,       # Canny edge threshold (b/n keys, step 5)
    
    # Frame rotation (adjustable with [ ] keys)
    "rotation_angle": 180,    # 0, 90, 180, 270 degrees
    
    # Frequency analysis parameters
    "target_frequency": 2.0,         # Target LED frequency in Hz (2Hz = 0.5s ON, 0.5s OFF)
    "frequency_tolerance": 0.3,      # Frequency tolerance (±0.3 Hz for more robustness)
    "analysis_duration": 5.0,        # Duration in seconds to analyze frequency
    "min_detection_count": 6,        # Minimum detections needed to track a circle (increased for 2Hz)
    "led_roi_padding": 30,           # Padding around detected LEDs for focused ROI
    "brightness_threshold": 0.7,     # Minimum brightness consistency for LED validation
    "size_consistency_threshold": 3, # Maximum radius variation allowed for LEDs
    "stage1_roi_count": 3,           # Number of ROI areas to create in stage 1
    "stage1_roi_size": 100,          # Size of each stage 1 ROI (pixels)
    "stage2_analysis_duration": 5.0, # Duration for stage 2 analysis
    
    # Fixed parameters
    "accumulator_threshold": 10,  # Reduced from 20 to be less strict
    "display_width": 1280,
    "display_height": 720,
    "show_binary_frame": True,
    "show_detection_info": True,
    "save_data": True,
    "data_filename": "circle_detection_data.json"
}

class CircleDetector:
    def __init__(self, config):
        """Initialize the circle LED detector with enhanced error handling"""
        try:
            self.config = config
            self.rtsp_url = f"rtsp://{config['camera_username']}:{config['camera_password']}@{config['camera_ip']}:{config['camera_port']}{config['camera_stream_path']}"
            self.cap = None
            self.data_log = []
            self.last_save_time = time.time()
            
            # Mouse position tracking
            self.mouse_x = 0
            self.mouse_y = 0
            self.mouse_clicked = False
            
            # Frame counter for debugging
            self.frame_count = 0
            
            # Frequency analysis tracking with enhanced safety
            self.analysis_mode = "stage1_analysis"  # "stage1_analysis", "stage2_analysis", or "final_tracking"
            self.analysis_start_time = None
            self.circle_history = {}  # Track circles over time
            self.next_circle_id = 1
            self.confirmed_leds = []  # List of confirmed LED positions
            self.led_rois = []  # List of focused ROI areas around LEDs
            self.stage1_rois = []  # 3 ROI areas from stage 1
            self.stage2_start_time = None
            
            # Initialize camera with error handling
            self.init_camera()
            
        except Exception as e:
            print(f"❌ Critical error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def init_camera(self):
        """Initialize camera connection"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to camera")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function to track cursor position"""
        # Convert mouse coordinates back to original frame coordinates
        # Since we resize the display, we need to scale back
        original_x = int(x * (1280 / self.config["display_width"]))
        original_y = int(y * (720 / self.config["display_height"]))
        
        # If we're showing both original and binary frames side by side
        if self.config["show_binary_frame"]:
            # The display shows original + binary side by side
            # So if x > half width, it's on the binary side
            display_half_width = self.config["display_width"] // 2
            if x > display_half_width:
                # Mouse is on binary frame side
                original_x = int((x - display_half_width) * (1280 / display_half_width))
            else:
                # Mouse is on original frame side
                original_x = int(x * (1280 / display_half_width))
        
        self.mouse_x = original_x
        self.mouse_y = original_y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
            print(f"Mouse clicked at: ({self.mouse_x}, {self.mouse_y})")
            print(f"Use these coordinates to update reference position in config file")
    
    def apply_binary_threshold(self, frame):
        """Apply binary threshold to frame with ROI cropping - delegates to LED ROI method"""
        return self.apply_binary_threshold_with_led_rois(frame)
    
    def rotate_frame(self, frame):
        """Rotate frame based on rotation_angle setting"""
        angle = self.config["rotation_angle"]
        if angle == 0:
            return frame
        elif angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def detect_circles(self, binary_frame):
        """Detect circle-like shapes in binary frame"""
        circles = cv2.HoughCircles(
            binary_frame,
            cv2.HOUGH_GRADIENT,
            dp=self.config["hough_dp"],
            minDist=self.config["hough_min_dist"],
            param1=self.config["hough_param1"],
            param2=self.config["accumulator_threshold"],
            minRadius=self.config["min_radius"],
            maxRadius=self.config["max_radius"]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return []
    
    def match_circles_to_history(self, current_circles, current_time, binary_frame):
        """Match detected circles to historical circles for tracking"""
        max_distance = 30  # Maximum pixel distance to consider same circle
        
        # Add current detections to history
        for circle in current_circles:
            x, y, r = circle
            best_match_id = None
            min_distance = float('inf')
            
            # Find closest historical circle
            for circle_id, history in self.circle_history.items():
                if len(history['positions']) > 0:
                    last_pos = history['positions'][-1]
                    distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                    if distance < min_distance and distance < max_distance:
                        min_distance = distance
                        best_match_id = circle_id
            
            # Calculate brightness at circle location
            brightness = self.calculate_circle_brightness(binary_frame, x, y, r)
            
            # Update existing circle or create new one
            if best_match_id:
                self.circle_history[best_match_id]['positions'].append((x, y))
                self.circle_history[best_match_id]['timestamps'].append(current_time)
                self.circle_history[best_match_id]['radius'] = r
                self.circle_history[best_match_id]['brightness_values'].append(brightness)
                self.circle_history[best_match_id]['radii'].append(r)
            else:
                # Create new circle track
                self.circle_history[self.next_circle_id] = {
                    'positions': [(x, y)],
                    'timestamps': [current_time],
                    'radius': r,
                    'radii': [r],
                    'brightness_values': [brightness],
                    'frequency': 0.0,
                    'is_led': False,
                    'confidence': 0.0
                }
                self.next_circle_id += 1
    
    def calculate_circle_brightness(self, binary_frame, x, y, radius):
        """Calculate average brightness in circle area"""
        try:
            # Ensure coordinates are within frame bounds
            h, w = binary_frame.shape
            x = max(0, min(w-1, int(x)))
            y = max(0, min(h-1, int(y)))
            radius = max(1, min(50, int(radius)))
            
            # Create a mask for the circle
            mask = np.zeros_like(binary_frame)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Calculate mean brightness in the circle area
            circle_area = cv2.bitwise_and(binary_frame, mask)
            if np.any(circle_area > 0):
                mean_brightness = np.mean(circle_area[circle_area > 0])
                return mean_brightness / 255.0  # Normalize to 0-1
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️  Error calculating brightness: {e}")
            return 0.0
    
    def calculate_circle_frequencies(self):
        """Calculate blinking frequency for each tracked circle with improved validation"""
        try:
            current_time = time.time()
            
            for circle_id, history in self.circle_history.items():
                try:
                    timestamps = history.get('timestamps', [])
                    brightness_values = history.get('brightness_values', [])
                    radii = history.get('radii', [])
                    
                    # Need sufficient detections to calculate frequency
                    if len(timestamps) < self.config["min_detection_count"]:
                        continue
                    
                    # Remove old timestamps (keep only analysis duration)
                    analysis_start = current_time - self.config["analysis_duration"]
                    recent_indices = [i for i, t in enumerate(timestamps) if t >= analysis_start]
                    
                    if len(recent_indices) < self.config["min_detection_count"]:
                        continue
                    
                    recent_timestamps = [timestamps[i] for i in recent_indices]
                    recent_brightness = [brightness_values[i] for i in recent_indices] if brightness_values else [1.0] * len(recent_indices)
                    recent_radii = [radii[i] for i in recent_indices] if radii else [history.get('radius', 5)] * len(recent_indices)
                    
                    # Calculate frequency based on detection pattern
                    detection_intervals = []
                    for i in range(1, len(recent_timestamps)):
                        interval = recent_timestamps[i] - recent_timestamps[i-1]
                        if interval > 0.05:  # Ignore very short intervals (noise)
                            detection_intervals.append(interval)
                    
                    if detection_intervals:
                        # For 2Hz frequency, we expect average interval of ~0.5s between detections
                        avg_interval = np.mean(detection_intervals)
                        
                        # Calculate frequency: For 2Hz, LED is ON for 0.25s, OFF for 0.25s
                        if avg_interval > 0.1:  # Sanity check
                            frequency = 1.0 / avg_interval  # This should give us ~2Hz for 0.5s intervals
                        else:
                            frequency = 0.0
                        
                        history['frequency'] = frequency
                        
                        # Calculate validation metrics with safety checks
                        if len(recent_brightness) > 1:
                            brightness_consistency = 1.0 - (np.std(recent_brightness) / (np.mean(recent_brightness) + 0.001))
                        else:
                            brightness_consistency = 0.5
                        
                        if len(recent_radii) > 1:
                            size_consistency = 1.0 - (np.std(recent_radii) / (np.mean(recent_radii) + 0.001))
                        else:
                            size_consistency = 0.5
                        
                        # Check if frequency matches LED target with improved validation
                        target_freq = self.config["target_frequency"]
                        tolerance = self.config["frequency_tolerance"]
                        
                        frequency_match = abs(frequency - target_freq) <= tolerance
                        brightness_ok = brightness_consistency >= self.config["brightness_threshold"]
                        size_ok = np.std(recent_radii) <= self.config["size_consistency_threshold"] if len(recent_radii) > 1 else True
                        
                        # Calculate confidence score
                        freq_score = max(0, 1.0 - abs(frequency - target_freq) / tolerance) if tolerance > 0 else 0
                        brightness_score = max(0, min(1, brightness_consistency))
                        size_score = max(0, min(1, size_consistency))
                        detection_score = min(1.0, len(recent_timestamps) / (self.config["analysis_duration"] * target_freq)) if target_freq > 0 else 0
                        
                        confidence = (freq_score * 0.4 + brightness_score * 0.2 + size_score * 0.2 + detection_score * 0.2)
                        history['confidence'] = max(0, min(1, confidence))
                        
                        # More lenient LED detection - focus on frequency primarily
                        if frequency_match and confidence > 0.5:
                            history['is_led'] = True
                        else:
                            history['is_led'] = False
                    else:
                        history['frequency'] = 0.0
                        history['confidence'] = 0.0
                        history['is_led'] = False
                        
                except Exception as e:
                    print(f"⚠️  Error processing circle {circle_id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error in frequency calculation: {e}")
            # Reset to safe state
            for history in self.circle_history.values():
                history['frequency'] = 0.0
                history['confidence'] = 0.0
                history['is_led'] = False
    
    def generate_stage1_rois(self):
        """Generate 3 ROI areas based on most promising LED detections from stage 1"""
        self.stage1_rois = []
        
        # Get all circles with reasonable confidence and frequency
        candidate_circles = []
        for circle_id, history in self.circle_history.items():
            if (history.get('confidence', 0) > 0.3 and 
                len(history['positions']) >= self.config["min_detection_count"] // 2):
                
                # Get average position for this circle
                positions = history['positions']
                avg_x = int(np.mean([pos[0] for pos in positions]))
                avg_y = int(np.mean([pos[1] for pos in positions]))
                
                candidate_circles.append({
                    'x': avg_x,
                    'y': avg_y,
                    'confidence': history.get('confidence', 0),
                    'frequency': history.get('frequency', 0),
                    'detection_count': len(positions),
                    'circle_id': circle_id
                })
        
        # Sort by confidence score
        candidate_circles.sort(key=lambda c: c['confidence'], reverse=True)
        
        # Create ROI areas around the most promising locations
        roi_size = self.config["stage1_roi_size"]
        max_rois = self.config["stage1_roi_count"]
        
        created_rois = []
        
        for candidate in candidate_circles:
            if len(created_rois) >= max_rois:
                break
                
            x, y = candidate['x'], candidate['y']
            
            # Check if this location is too close to existing ROIs
            too_close = False
            for existing_roi in created_rois:
                center_distance = np.sqrt((x - existing_roi['center_x'])**2 + (y - existing_roi['center_y'])**2)
                if center_distance < roi_size:  # ROIs would overlap significantly
                    too_close = True
                    break
            
            if not too_close:
                # Create ROI around this candidate
                roi = {
                    'x_min': max(0, x - roi_size // 2),
                    'x_max': min(1280, x + roi_size // 2),
                    'y_min': max(0, y - roi_size // 2),
                    'y_max': min(720, y + roi_size // 2),
                    'center_x': x,
                    'center_y': y,
                    'confidence': candidate['confidence'],
                    'roi_id': len(created_rois) + 1
                }
                created_rois.append(roi)
        
        # If we don't have enough ROIs, create them in areas with most detections
        if len(created_rois) < max_rois:
            print(f"⚠️  Only found {len(created_rois)} promising areas, creating additional ROIs...")
            
            # Fill remaining ROIs by dividing the original ROI into sections
            original_width = self.config["roi_x_max"] - self.config["roi_x_min"]
            original_height = self.config["roi_y_max"] - self.config["roi_y_min"]
            
            sections_per_row = 2
            sections_per_col = 2
            section_width = original_width // sections_per_row
            section_height = original_height // sections_per_col
            
            for i in range(sections_per_row):
                for j in range(sections_per_col):
                    if len(created_rois) >= max_rois:
                        break
                    
                    center_x = self.config["roi_x_min"] + (i + 0.5) * section_width
                    center_y = self.config["roi_y_min"] + (j + 0.5) * section_height
                    
                    # Check if this overlaps with existing ROIs
                    too_close = False
                    for existing_roi in created_rois:
                        center_distance = np.sqrt((center_x - existing_roi['center_x'])**2 + (center_y - existing_roi['center_y'])**2)
                        if center_distance < roi_size:
                            too_close = True
                            break
                    
                    if not too_close:
                        roi = {
                            'x_min': max(0, int(center_x - roi_size // 2)),
                            'x_max': min(1280, int(center_x + roi_size // 2)),
                            'y_min': max(0, int(center_y - roi_size // 2)),
                            'y_max': min(720, int(center_y + roi_size // 2)),
                            'center_x': int(center_x),
                            'center_y': int(center_y),
                            'confidence': 0.1,  # Low confidence for fallback ROIs
                            'roi_id': len(created_rois) + 1
                        }
                        created_rois.append(roi)
        
        self.stage1_rois = created_rois[:max_rois]  # Ensure we don't exceed max count
        
        print(f"📍 Generated {len(self.stage1_rois)} Stage 1 ROI areas:")
        for i, roi in enumerate(self.stage1_rois):
            print(f"   ROI{roi['roi_id']}: Center({roi['center_x']}, {roi['center_y']}) - Confidence: {roi['confidence']:.2f}")
    
    def generate_led_rois(self):
        """Generate focused ROI areas around confirmed LEDs"""
        self.confirmed_leds = []
        self.led_rois = []
        
        for circle_id, history in self.circle_history.items():
            if history['is_led'] and len(history['positions']) > 0:
                # Get latest position
                last_pos = history['positions'][-1]
                x, y = last_pos
                padding = self.config["led_roi_padding"]
                
                # Create ROI around LED
                roi = {
                    'x_min': max(0, x - padding),
                    'x_max': min(1280, x + padding),
                    'y_min': max(0, y - padding),
                    'y_max': min(720, y + padding),
                    'center': (x, y),
                    'circle_id': circle_id
                }
                
                self.led_rois.append(roi)
                self.confirmed_leds.append({
                    'x': x,
                    'y': y,
                    'radius': history['radius'],
                    'frequency': history['frequency'],
                    'confidence': history.get('confidence', 0.0),
                    'circle_id': circle_id
                })
    
    def apply_binary_threshold_with_led_rois(self, frame):
        """Apply binary threshold to frame with focused LED ROI areas"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.analysis_mode == "stage1_analysis":
            # Use original ROI during stage 1 analysis
            roi_mask = np.zeros_like(gray)
            roi_mask[self.config["roi_y_min"]:self.config["roi_y_max"], 
                     self.config["roi_x_min"]:self.config["roi_x_max"]] = 255
        elif self.analysis_mode == "stage2_analysis":
            # Use 3 ROI areas during stage 2 analysis
            roi_mask = np.zeros_like(gray)
            for stage1_roi in self.stage1_rois:
                roi_mask[stage1_roi['y_min']:stage1_roi['y_max'], 
                         stage1_roi['x_min']:stage1_roi['x_max']] = 255
        else:
            # Use final focused LED ROIs during tracking phase
            roi_mask = np.zeros_like(gray)
            for led_roi in self.led_rois:
                roi_mask[led_roi['y_min']:led_roi['y_max'], 
                         led_roi['x_min']:led_roi['x_max']] = 255
        
        # Apply mask to grayscale image
        gray_roi = cv2.bitwise_and(gray, roi_mask)
        
        # Apply binary threshold
        threshold_value = self.config["binary_threshold"]
        _, binary = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def calculate_distance(self, circle_center):
        """Calculate distance from circle center to reference position"""
        ref_x = self.config["reference_x"]
        ref_y = self.config["reference_y"]
        
        distance_pixels = np.sqrt((circle_center[0] - ref_x)**2 + (circle_center[1] - ref_y)**2)
        
        return {
            "distance_pixels": float(distance_pixels),
            "distance_units": float(distance_pixels),  # Using pixels as units
            "circle_x": int(circle_center[0]),
            "circle_y": int(circle_center[1]),
            "ref_x": ref_x,
            "ref_y": ref_y
        }
    
    def save_data(self, detection_data):
        """Save detection data to file"""
        if not self.config["save_data"]:
            return
            
        current_time = time.time()
        if current_time - self.last_save_time >= 1.0:  # Save every second
            # Add timestamp to data
            timestamped_data = {
                "timestamp": datetime.now().isoformat(),
                "unix_time": current_time,
                **detection_data
            }
            
            self.data_log.append(timestamped_data)
            
            # Save to file
            try:
                with open(self.config["data_filename"], 'w') as f:
                    json.dump(self.data_log, f, indent=2)
                # Removed excessive save printing
            except Exception as e:
                print(f"Error saving data: {e}")
            
            self.last_save_time = current_time
    
    def draw_detection_info(self, frame, binary_frame, circles, detection_data):
        """Draw detection information on frames"""
        # Get frame dimensions for center calculation
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Draw ROI boundaries based on current mode
        if self.analysis_mode == "stage1_analysis":
            # Draw original ROI
            cv2.rectangle(frame, 
                         (self.config["roi_x_min"], self.config["roi_y_min"]), 
                         (self.config["roi_x_max"], self.config["roi_y_max"]), 
                         (255, 255, 0), 2)  # Yellow original ROI rectangle
        elif self.analysis_mode == "stage2_analysis":
            # Draw 3 Stage 1 ROI areas
            colors = [(255, 165, 0), (255, 20, 147), (0, 191, 255)]  # Orange, Deep Pink, Deep Sky Blue
            for i, stage1_roi in enumerate(self.stage1_rois):
                color = colors[i % len(colors)]
                cv2.rectangle(frame,
                             (stage1_roi['x_min'], stage1_roi['y_min']),
                             (stage1_roi['x_max'], stage1_roi['y_max']),
                             color, 3)  # Thick colored rectangles
                # Add ROI label
                cv2.putText(frame, f"ROI{stage1_roi['roi_id']}", 
                           (stage1_roi['x_min'] + 5, stage1_roi['y_min'] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Draw final LED ROI areas
            for led_roi in self.led_rois:
                cv2.rectangle(frame,
                             (led_roi['x_min'], led_roi['y_min']),
                             (led_roi['x_max'], led_roi['y_max']),
                             (0, 255, 0), 2)  # Green final LED ROI rectangles
        
        # Draw camera center
        cv2.circle(frame, (center_x, center_y), 8, (255, 0, 255), 2)  # Magenta circle
        cv2.putText(frame, "CENTER", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw reference position
        ref_x = self.config["reference_x"]
        ref_y = self.config["reference_y"]
        cv2.circle(frame, (ref_x, ref_y), 10, (0, 255, 0), 2)  # Green circle
        cv2.putText(frame, "REF", (ref_x + 15, ref_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw line from center to reference and calculate distance components
        cv2.line(frame, (center_x, center_y), (ref_x, ref_y), (255, 0, 255), 2)  # Magenta line
        center_to_ref_x = ref_x - center_x  # X distance (positive = right)
        center_to_ref_y = ref_y - center_y  # Y distance (positive = down)
        
        # Draw detected circles with individual information
        circle_info_list = []
        for i, circle in enumerate(circles):
            x, y, r = circle
            
            # Check if this circle is a confirmed LED
            is_confirmed_led = False
            circle_frequency = 0.0
            circle_confidence = 0.0
            for circle_id, history in self.circle_history.items():
                if len(history['positions']) > 0:
                    last_pos = history['positions'][-1]
                    distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                    if distance < 20:  # Close enough to be same circle
                        circle_frequency = history['frequency']
                        circle_confidence = history.get('confidence', 0.0)
                        is_confirmed_led = history['is_led']
                        break
            
            # Draw circle with color based on LED status
            if is_confirmed_led:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)  # Green for confirmed LEDs
                cv2.circle(frame, (x, y), 2, (0, 255, 0), 3)   # Green center dot
                cv2.putText(frame, f"LED{i+1}", (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)  # Red for unconfirmed circles
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)   # Red center dot
                cv2.putText(frame, f"C{i+1}", (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw line from circle to reference
            cv2.line(frame, (x, y), (ref_x, ref_y), (255, 255, 0), 1)  # Yellow line
            
            # Calculate distance components from circle to reference
            circle_to_ref_x = ref_x - x
            circle_to_ref_y = ref_y - y
            
            # Store circle information
            circle_info_list.append({
                'id': i+1,
                'x': x,
                'y': y,
                'radius': r,
                'to_ref_x': circle_to_ref_x,
                'to_ref_y': circle_to_ref_y,
                'frequency': circle_frequency,
                'confidence': circle_confidence,
                'is_led': is_confirmed_led
            })
        
        # Display detection information
        if self.config["show_detection_info"]:
            info_y = 30
            
            # Show current mode and analysis progress
            if self.analysis_mode == "stage1_analysis":
                elapsed_time = time.time() - self.analysis_start_time if self.analysis_start_time else 0
                remaining_time = max(0, self.config["analysis_duration"] - elapsed_time)
                cv2.putText(frame, f"STAGE 1: INITIAL ANALYSIS... {remaining_time:.1f}s", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                info_y += 30
            elif self.analysis_mode == "stage2_analysis":
                elapsed_time = time.time() - self.stage2_start_time if self.stage2_start_time else 0
                remaining_time = max(0, self.config["stage2_analysis_duration"] - elapsed_time)
                cv2.putText(frame, f"STAGE 2: ANALYZING 3 ROIs... {remaining_time:.1f}s", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                info_y += 30
            else:
                cv2.putText(frame, f"FINAL: LED TRACKING - {len(self.confirmed_leds)} LEDs", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                info_y += 30
            
            cv2.putText(frame, f"Circles: {len(circles)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Target Freq: {self.config['target_frequency']}Hz", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Binary threshold: {self.config['binary_threshold']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Reference: ({ref_x}, {ref_y})", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Radius: {self.config['min_radius']}-{self.config['max_radius']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"ROI: X({self.config['roi_x_min']}-{self.config['roi_x_max']}) Y({self.config['roi_y_min']}-{self.config['roi_y_max']})", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 20
            cv2.putText(frame, f"HoughDP: {self.config['hough_dp']:.1f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 20
            cv2.putText(frame, f"HoughDist: {self.config['hough_min_dist']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 20
            cv2.putText(frame, f"HoughParam1: {self.config['hough_param1']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 20
            cv2.putText(frame, f"Rotation: {self.config['rotation_angle']}°", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display information for each detected circle
            for circle_info in circle_info_list:
                info_y += 25
                status = "LED" if circle_info['is_led'] else "CIRCLE"
                cv2.putText(frame, f"{status}{circle_info['id']}: ({circle_info['x']},{circle_info['y']}) R={circle_info['radius']}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if circle_info['is_led'] else (255, 255, 255), 1)
                
                info_y += 20
                if circle_info['frequency'] > 0:
                    cv2.putText(frame, f"  Freq:{circle_info['frequency']:.2f}Hz Conf:{circle_info['confidence']:.2f} X:{circle_info['to_ref_x']:+.0f} Y:{circle_info['to_ref_y']:+.0f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(frame, f"  ->Ref X:{circle_info['to_ref_x']:+.0f} Y:{circle_info['to_ref_y']:+.0f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display mouse cursor position (always show)
        info_y = frame.shape[0] - 60  # Position near bottom of frame
        cv2.putText(frame, f"Mouse: ({self.mouse_x}, {self.mouse_y})", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Cyan color
        
        info_y += 20
        cv2.putText(frame, "Click to get coordinates for config", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("Starting circle detection with frequency analysis...")
        print(f"Target LED frequency: {self.config['target_frequency']}Hz")
        print(f"Analysis duration: {self.config['analysis_duration']}s")
        print(f"Expected to find: 8 LEDs blinking at 2Hz")
        print(f"Initial settings - Reference: ({self.config['reference_x']}, {self.config['reference_y']}), Binary threshold: {self.config['binary_threshold']}")
        print("="*50)
        print("THREE-STAGE ANALYSIS WORKFLOW:")
        print("1. STAGE 1 (5s): Analyze entire ROI, identify 3 most promising areas")
        print("2. STAGE 2 (5s): Focus on 3 ROI areas, find best LED candidates")
        print("3. FINAL: Single ROI with confirmed LEDs for precise tracking")
        print("="*50)
        print("KEYBOARD CONTROLS:")
        print("  +/- : Adjust binary threshold")
        print("  Arrow keys or T/F/G/H : Move reference position")
        print("  1/2 : Adjust min radius (decrease/increase)")
        print("  3/4 : Adjust max radius (decrease/increase)")
        print("  u/i/o/p : Adjust ROI X borders")
        print("  j/k/l/; : Adjust ROI Y borders")
        print("  z/x : Adjust HoughCircles dp (resolution ratio)")
        print("  c/v : Adjust HoughCircles minDist (min distance between circles)")
        print("  b/n : Adjust HoughCircles param1 (Canny edge threshold)")
        print("  [ ] : Rotate frame 90° clockwise")
        print("  SPACE : Restart frequency analysis")
        print("  s : Save frames")
        print("  r : Reload config file")
        print("  q/ESC : Quit")
        print("="*50)
        print("*** CLICK ON THE VIDEO WINDOW TO ENABLE KEYBOARD CONTROLS ***")
        print("="*50)
        
        # Initialize analysis phase
        self.analysis_start_time = time.time()
        self.analysis_mode = "stage1_analysis"
        
        # Create window and set mouse callback
        window_name = "Circle Detection - Frequency Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Ensure window is focused
        cv2.moveWindow(window_name, 100, 100)  # Move window to ensure it's visible
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # Apply frame rotation if needed
                try:
                    frame = self.rotate_frame(frame)
                except Exception as e:
                    print(f"⚠️  Frame rotation error: {e}")
                
                # Apply binary threshold
                try:
                    binary_frame = self.apply_binary_threshold(frame)
                except Exception as e:
                    print(f"⚠️  Binary threshold error: {e}")
                    binary_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                
                # Detect circles
                try:
                    circles = self.detect_circles(binary_frame)
                except Exception as e:
                    print(f"⚠️  Circle detection error: {e}")
                    circles = []
                
                # Handle three-stage analysis workflow
                try:
                    if self.analysis_mode == "stage1_analysis":
                        # Stage 1: Track circles during initial analysis phase
                        self.match_circles_to_history(circles, current_time, binary_frame)
                        
                        # Check if stage 1 analysis period is complete
                        elapsed_time = current_time - self.analysis_start_time
                        if elapsed_time >= self.config["analysis_duration"]:
                            print(f"\n🔍 Stage 1 complete! Generating 3 ROI areas...")
                            
                            # Calculate frequencies and generate 3 ROI areas
                            self.calculate_circle_frequencies()
                            self.generate_stage1_rois()
                            
                            # Switch to stage 2 analysis
                            self.analysis_mode = "stage2_analysis"
                            self.stage2_start_time = time.time()
                            
                            # Reset tracking for stage 2
                            self.circle_history = {}
                            self.next_circle_id = 1
                            
                            print(f"✅ Stage 1 → Stage 2: Analyzing {len(self.stage1_rois)} ROI areas")
                            
                    elif self.analysis_mode == "stage2_analysis":
                        # Stage 2: Track circles in the 3 ROI areas
                        self.match_circles_to_history(circles, current_time, binary_frame)
                        
                        # Check if stage 2 analysis period is complete
                        elapsed_time = current_time - self.stage2_start_time
                        if elapsed_time >= self.config["stage2_analysis_duration"]:
                            print(f"\n🎯 Stage 2 complete! Identifying final LED locations...")
                            
                            # Calculate frequencies and identify final LEDs
                            self.calculate_circle_frequencies()
                            self.generate_led_rois()
                            
                            # Switch to final tracking mode
                            self.analysis_mode = "final_tracking"
                            
                            print(f"🏆 Final Stage: Found {len(self.confirmed_leds)} confirmed LEDs at 2Hz")
                            for i, led in enumerate(self.confirmed_leds):
                                print(f"   LED{i+1}: ({led['x']}, {led['y']}) - Frequency: {led['frequency']:.2f}Hz - Confidence: {led.get('confidence', 0):.2f}")
                            
                            if len(self.confirmed_leds) == 0:
                                print("⚠️  No LEDs found in Stage 2. Restarting analysis...")
                                # Restart from stage 1
                                self.analysis_start_time = time.time()
                                self.analysis_mode = "stage1_analysis"
                                self.circle_history = {}
                                self.next_circle_id = 1
                                self.stage1_rois = []
                                
                except Exception as e:
                    print(f"⚠️  Stage analysis error: {e}")
                
                # Process detected circles
                try:
                    detection_data = {
                        "total_circles": len(circles),
                        "circles": [],
                        "analysis_mode": self.analysis_mode,
                        "confirmed_leds": len(self.confirmed_leds)
                    }
                    
                    if len(circles) > 0:
                        ref_x = self.config["reference_x"]
                        ref_y = self.config["reference_y"]
                        
                        # Process each circle
                        for i, circle in enumerate(circles):
                            circle_data = self.calculate_distance(circle)
                            circle_data["circle_id"] = i + 1
                            
                            # Add frequency information if available
                            circle_frequency = 0.0
                            is_led = False
                            for circle_id, history in self.circle_history.items():
                                if len(history['positions']) > 0:
                                    last_pos = history['positions'][-1]
                                    distance = np.sqrt((circle[0] - last_pos[0])**2 + (circle[1] - last_pos[1])**2)
                                    if distance < 20:  # Close enough to be same circle
                                        circle_frequency = history['frequency']
                                        is_led = history['is_led']
                                        break
                            
                            circle_data["frequency"] = circle_frequency
                            circle_data["is_led"] = is_led
                            detection_data["circles"].append(circle_data)
                        
                        # For backward compatibility, also include the closest circle as main detection
                        distances_to_ref = [np.sqrt((c[0] - ref_x)**2 + (c[1] - ref_y)**2) for c in circles]
                        closest_circle_idx = np.argmin(distances_to_ref)
                        closest_circle = circles[closest_circle_idx]
                        
                        # Calculate distance data for closest circle (for compatibility)
                        main_detection = self.calculate_distance(closest_circle)
                        detection_data.update(main_detection)  # Add main detection data
                        
                        # Save data
                        self.save_data(detection_data)
                        
                except Exception as e:
                    print(f"⚠️  Detection data processing error: {e}")
                    detection_data = {"total_circles": 0, "circles": [], "analysis_mode": self.analysis_mode, "confirmed_leds": 0}
                
                # Draw detection information
                try:
                    display_frame = self.draw_detection_info(frame, binary_frame, circles, detection_data)
                except Exception as e:
                    print(f"⚠️  Drawing detection info error: {e}")
                    display_frame = frame
                
                # Prepare display
                try:
                    if self.config["show_binary_frame"]:
                        # Convert binary to color for display
                        binary_color = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
                        # Concatenate original and binary frames
                        combined_frame = np.hstack((display_frame, binary_color))
                    else:
                        combined_frame = display_frame
                    
                    # Resize for display
                    display_resized = cv2.resize(combined_frame, 
                                               (self.config["display_width"], self.config["display_height"]))
                    
                    # Show frame
                    cv2.imshow(window_name, display_resized)
                    
                except Exception as e:
                    print(f"⚠️  Display preparation error: {e}")
                    try:
                        cv2.imshow(window_name, frame)
                    except:
                        print("❌ Cannot display frame at all")
                
                # Handle keyboard input with comprehensive controls
                try:
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                        break
                        
                    # Restart frequency analysis
                    elif key == ord(' '):  # SPACE to restart analysis
                        print("\n🔄 Restarting three-stage analysis...")
                        self.analysis_start_time = time.time()
                        self.analysis_mode = "stage1_analysis"
                        self.circle_history = {}
                        self.next_circle_id = 1
                        self.confirmed_leds = []
                        self.led_rois = []
                        self.stage1_rois = []
                        self.stage2_start_time = None
                        
                    # Binary threshold controls
                    elif key == ord('+') or key == ord('='):  # '+' to increase threshold
                        self.config["binary_threshold"] = min(255, self.config["binary_threshold"] + 5)
                    elif key == ord('-') or key == ord('_'):  # '-' to decrease threshold
                        self.config["binary_threshold"] = max(0, self.config["binary_threshold"] - 5)
                        
                    # Reference position controls (Arrow keys + TFGH alternative for compatibility)
                    elif key == 82 or key == 0 or key == ord('t') or key == ord('T'):  # Up arrow or T
                        self.config["reference_y"] = max(0, self.config["reference_y"] - 5)
                    elif key == 84 or key == 1 or key == ord('g') or key == ord('G'):  # Down arrow or G
                        self.config["reference_y"] = min(960, self.config["reference_y"] + 5)
                    elif key == 81 or key == 2 or key == ord('f') or key == ord('F'):  # Left arrow or F
                        self.config["reference_x"] = max(0, self.config["reference_x"] - 5)
                    elif key == 83 or key == 3 or key == ord('h') or key == ord('H'):  # Right arrow or H
                        self.config["reference_x"] = min(1280, self.config["reference_x"] + 5)
                        
                    # Radius controls
                    elif key == ord('1'):  # Decrease min radius
                        self.config["min_radius"] = max(1, self.config["min_radius"] - 1)
                    elif key == ord('2'):  # Increase min radius
                        self.config["min_radius"] = min(self.config["max_radius"] - 1, self.config["min_radius"] + 1)
                    elif key == ord('3'):  # Decrease max radius
                        self.config["max_radius"] = max(self.config["min_radius"] + 1, self.config["max_radius"] - 1)
                    elif key == ord('4'):  # Increase max radius
                        self.config["max_radius"] = min(100, self.config["max_radius"] + 1)
                        
                    # ROI controls (updated to avoid conflicts)
                    elif key == ord('u'):  # Decrease ROI x_min
                        self.config["roi_x_min"] = max(0, self.config["roi_x_min"] - 10)
                    elif key == ord('i'):  # Increase ROI x_min
                        self.config["roi_x_min"] = min(self.config["roi_x_max"] - 50, self.config["roi_x_min"] + 10)
                    elif key == ord('o'):  # Decrease ROI x_max
                        self.config["roi_x_max"] = max(self.config["roi_x_min"] + 50, self.config["roi_x_max"] - 10)
                    elif key == ord('p'):  # Increase ROI x_max
                        self.config["roi_x_max"] = min(1280, self.config["roi_x_max"] + 10)
                    elif key == ord('j'):  # Decrease ROI y_min
                        self.config["roi_y_min"] = max(0, self.config["roi_y_min"] - 10)
                    elif key == ord('k'):  # Increase ROI y_min
                        self.config["roi_y_min"] = min(self.config["roi_y_max"] - 50, self.config["roi_y_min"] + 10)
                    elif key == ord('l'):  # Decrease ROI y_max
                        self.config["roi_y_max"] = max(self.config["roi_y_min"] + 50, self.config["roi_y_max"] - 10)
                    elif key == ord(';'):  # Increase ROI y_max
                        self.config["roi_y_max"] = min(960, self.config["roi_y_max"] + 10)
                        
                    # HoughCircles parameter controls
                    elif key == ord('z'):  # Decrease dp (resolution ratio)
                        self.config["hough_dp"] = max(0.2, self.config["hough_dp"] - 0.2)
                    elif key == ord('x'):  # Increase dp (resolution ratio)
                        self.config["hough_dp"] = min(3.0, self.config["hough_dp"] + 0.2)
                    elif key == ord('c'):  # Decrease minDist
                        self.config["hough_min_dist"] = max(5, self.config["hough_min_dist"] - 5)
                    elif key == ord('v'):  # Increase minDist
                        self.config["hough_min_dist"] = min(100, self.config["hough_min_dist"] + 5)
                    elif key == ord('b'):  # Decrease param1 (Canny threshold)
                        self.config["hough_param1"] = max(10, self.config["hough_param1"] - 5)
                    elif key == ord('n'):  # Increase param1 (Canny threshold)
                        self.config["hough_param1"] = min(200, self.config["hough_param1"] + 5)
                        
                    # Frame rotation controls
                    elif key == ord('['):  # Rotate 90° clockwise
                        self.config["rotation_angle"] = (self.config["rotation_angle"] + 90) % 360
                        print(f"🔄 Frame rotated to {self.config['rotation_angle']}°")
                    elif key == ord(']'):  # Rotate 90° clockwise (same as [)
                        self.config["rotation_angle"] = (self.config["rotation_angle"] + 90) % 360
                        print(f"🔄 Frame rotated to {self.config['rotation_angle']}°")
                        
                    # Save frames and Config reload
                    elif key == ord('s'):  # 's' to save current frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        frame_filename = f"frame_{timestamp}.jpg"
                        binary_filename = f"binary_{timestamp}.jpg"
                        
                        cv2.imwrite(frame_filename, frame)
                        cv2.imwrite(binary_filename, binary_frame)
                        
                        # Get current working directory to show full path
                        import os
                        current_dir = os.getcwd()
                        print(f"📸 Frames saved:")
                        print(f"   Original: {os.path.join(current_dir, frame_filename)}")
                        print(f"   Binary:   {os.path.join(current_dir, binary_filename)}")
                        
                    elif key == ord('r'):  # 'r' to reload config (changed from 'c' to avoid conflict)
                        try:
                            with open("circle_detection_config.json", 'r') as f:
                                loaded_config = json.load(f)
                                self.config.update(loaded_config)
                            print("🔄 Configuration reloaded from file")
                        except Exception as e:
                            print(f"❌ Error reloading config: {e}")
                            
                except Exception as e:
                    print(f"⚠️  Keyboard input error: {e}")
                    
            except Exception as e:
                print(f"❌ Critical error in main loop: {e}")
                import traceback
                traceback.print_exc()
                # Add a small delay to prevent rapid error loops
                time.sleep(0.1)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final parameter summary
        print("\n" + "="*50)
        print("FINAL LED DETECTION RESULTS:")
        print("="*50)
        print(f"Analysis mode: {self.analysis_mode}")
        print(f"Confirmed LEDs: {len(self.confirmed_leds)}")
        for i, led in enumerate(self.confirmed_leds):
            print(f"  LED{i+1}: Position({led['x']}, {led['y']}) - Frequency: {led['frequency']:.2f}Hz")
        print(f"Total tracked circles: {len(self.circle_history)}")
        print(f"Total data entries saved: {len(self.data_log)}")
        print("Detection stopped")

def load_config(config_file="circle_detection_config.json"):
    """Load configuration from file or create default"""
    try:
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            # Update default config with loaded values
            CONFIG.update(loaded_config)
    except FileNotFoundError:
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        print(f"Error loading config: {e}, using default values")
    
    return CONFIG

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Create and run detector
    try:
        detector = CircleDetector(config)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
