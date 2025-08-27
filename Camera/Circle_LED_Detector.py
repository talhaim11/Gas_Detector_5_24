import numpy as np
import cv2
import time
import json
from datetime import datetime

# ================================
# CONFIGURABLE PARAMETERS
# ================================
CONFIG = {
    # Camera settings
    "camera_ip": "169.254.160.162",
    "camera_port": 8554,
    "camera_username": "fgcam",
    "camera_password": "admin",
    "camera_stream_path": "/0/unicast",
    
    # Image processing
    "binary_threshold": 255,  # Threshold for binary conversion (0-255)
    "binary_threshold_type": cv2.THRESH_BINARY,  # cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV
    
    # Field of Interest (ROI) - crop area for processing
    "field_of_interest": {
        "x_min": 150,    # Left boundary
        "x_max": 1100,   # Right boundary  
        "y_min": 150,    # Top boundary
        "y_max": 600,    # Bottom boundary
        "enabled": True  # Enable/disable cropping
    },
    
    # Reference position (where the LED should be detected)
    "reference_position": {"x": 770, "y": 310},  # Center of typical 1280x960 frame
    
    # Camera center (for coordinate system)
    "camera_center": {"x": 640, "y": 480},
    
    # Circle detection parameters
    "circle_detection": {
        "min_radius": 5,
        "max_radius": 50,
        "min_dist_between_circles": 30,
        "canny_high_threshold": 100,
        "accumulator_threshold": 20
    },
    
    # Distance measurement
    "pixels_per_unit": 1.0,  # Conversion factor from pixels to real units (mm, cm, etc.)
    "distance_unit": "pixels",  # Unit name for display
    
    # Data saving
    "save_data": True,
    "data_filename": "circle_detection_data.json",
    "save_interval_seconds": 1.0,  # Save data every N seconds
    
    # Display settings
    "display_width": 1280,
    "display_height": 720,
    "show_binary_frame": True,
    "show_detection_info": True
}

class CircleDetector:
    def __init__(self, config):
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
        
        # Initialize camera
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera connection"""
        print(f"Connecting to camera: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to camera")
        
        print("✅ Camera connected successfully")
    
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
        """Apply binary threshold to frame with optional ROI cropping"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply field of interest cropping if enabled
        if self.config["field_of_interest"]["enabled"]:
            x_min = self.config["field_of_interest"]["x_min"]
            x_max = self.config["field_of_interest"]["x_max"]
            y_min = self.config["field_of_interest"]["y_min"]
            y_max = self.config["field_of_interest"]["y_max"]
            
            # Create a mask for the ROI
            roi_mask = np.zeros_like(gray)
            roi_mask[y_min:y_max, x_min:x_max] = 255
            
            # Apply mask to grayscale image
            gray_roi = cv2.bitwise_and(gray, roi_mask)
        else:
            gray_roi = gray
        
        # Apply binary threshold - Fixed to use the actual parameter value
        threshold_value = self.config["binary_threshold"]
        _, binary = cv2.threshold(gray_roi, 
                                 threshold_value, 
                                 255, 
                                 self.config["binary_threshold_type"])
        
        # Debug: Print threshold value occasionally to verify it's changing
        if hasattr(self, 'frame_count') and self.frame_count % 60 == 0:  # Every 60 frames
            print(f"Current binary threshold: {threshold_value}")
        
        return binary
    
    def detect_circles(self, binary_frame):
        """Detect circle-like shapes in binary frame"""
        circles = cv2.HoughCircles(
            binary_frame,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.config["circle_detection"]["min_dist_between_circles"],
            param1=self.config["circle_detection"]["canny_high_threshold"],
            param2=self.config["circle_detection"]["accumulator_threshold"],
            minRadius=self.config["circle_detection"]["min_radius"],
            maxRadius=self.config["circle_detection"]["max_radius"]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return []
    
    def calculate_distance(self, circle_center):
        """Calculate distance from circle center to reference position"""
        ref_x = self.config["reference_position"]["x"]
        ref_y = self.config["reference_position"]["y"]
        
        distance_pixels = np.sqrt((circle_center[0] - ref_x)**2 + (circle_center[1] - ref_y)**2)
        distance_units = distance_pixels * self.config["pixels_per_unit"]
        
        return {
            "distance_pixels": float(distance_pixels),
            "distance_units": float(distance_units),
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
        if current_time - self.last_save_time >= self.config["save_interval_seconds"]:
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
                print(f"Data saved: {len(self.data_log)} entries")
            except Exception as e:
                print(f"Error saving data: {e}")
            
            self.last_save_time = current_time
    
    def draw_detection_info(self, frame, binary_frame, circles, detection_data):
        """Draw detection information on frames"""
        # Draw field of interest (ROI) boundaries if enabled
        if self.config["field_of_interest"]["enabled"]:
            x_min = self.config["field_of_interest"]["x_min"]
            x_max = self.config["field_of_interest"]["x_max"]
            y_min = self.config["field_of_interest"]["y_min"]
            y_max = self.config["field_of_interest"]["y_max"]
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)  # Magenta rectangle
            cv2.putText(frame, "ROI", (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw reference position
        ref_x = self.config["reference_position"]["x"]
        ref_y = self.config["reference_position"]["y"]
        cv2.circle(frame, (ref_x, ref_y), 10, (0, 255, 0), 2)  # Green circle
        cv2.putText(frame, "REF", (ref_x + 15, ref_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw camera center
        center_x = self.config["camera_center"]["x"]
        center_y = self.config["camera_center"]["y"]
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), 2)  # Blue circle
        cv2.putText(frame, "CENTER", (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw detected circles
        for circle in circles:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)  # Red circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)   # Red center dot
            
            # Draw line from circle to reference
            cv2.line(frame, (x, y), (ref_x, ref_y), (255, 255, 0), 1)  # Yellow line
        
        # Display detection information
        if self.config["show_detection_info"]:
            info_y = 30
            cv2.putText(frame, f"Circles detected: {len(circles)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Binary threshold: {self.config['binary_threshold']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if detection_data and "distance_units" in detection_data:
                info_y += 25
                cv2.putText(frame, f"Distance: {detection_data['distance_units']:.2f} {self.config['distance_unit']}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                info_y += 25
                cv2.putText(frame, f"Circle pos: ({detection_data['circle_x']}, {detection_data['circle_y']})", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
        print("Starting circle detection...")
        print(f"Reference position: ({self.config['reference_position']['x']}, {self.config['reference_position']['y']})")
        print(f"Binary threshold: {self.config['binary_threshold']}")
        print("Press 'q' or ESC to quit")
        print("Click on the video window to get coordinates for configuration")
        print("Keyboard controls: '+' / '-' to adjust binary threshold, 'r' to reload config")
        
        # Create window and set mouse callback
        cv2.namedWindow("Circle Detection - Live Stream", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Circle Detection - Live Stream", self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Apply binary threshold
            binary_frame = self.apply_binary_threshold(frame)
            
            # Detect circles
            circles = self.detect_circles(binary_frame)
            
            # Process detected circles
            detection_data = {}
            if len(circles) > 0:
                # Use the circle closest to reference position
                ref_x = self.config["reference_position"]["x"]
                ref_y = self.config["reference_position"]["y"]
                
                distances_to_ref = [np.sqrt((c[0] - ref_x)**2 + (c[1] - ref_y)**2) for c in circles]
                closest_circle_idx = np.argmin(distances_to_ref)
                closest_circle = circles[closest_circle_idx]
                
                # Calculate distance data
                detection_data = self.calculate_distance(closest_circle[:2])
                
                # Save data
                self.save_data(detection_data)
            
            # Draw detection information
            display_frame = self.draw_detection_info(frame, binary_frame, circles, detection_data)
            
            # Prepare display
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
            cv2.imshow("Circle Detection - Live Stream", display_resized)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # 's' to save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"frame_{timestamp}.jpg", frame)
                cv2.imwrite(f"binary_{timestamp}.jpg", binary_frame)
                print(f"Frames saved with timestamp {timestamp}")
            elif key == ord('+') or key == ord('='):  # '+' to increase threshold
                self.config["binary_threshold"] = min(255, self.config["binary_threshold"] + 5)
                print(f"Binary threshold increased to: {self.config['binary_threshold']}")
            elif key == ord('-') or key == ord('_'):  # '-' to decrease threshold
                self.config["binary_threshold"] = max(0, self.config["binary_threshold"] - 5)
                print(f"Binary threshold decreased to: {self.config['binary_threshold']}")
            elif key == ord('r'):  # 'r' to reload config
                try:
                    with open("circle_detection_config.json", 'r') as f:
                        loaded_config = json.load(f)
                        self.config.update(loaded_config)
                    print(f"Configuration reloaded. Binary threshold: {self.config['binary_threshold']}")
                except Exception as e:
                    print(f"Error reloading config: {e}")
                print(f"Frames saved with timestamp {timestamp}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

def load_config(config_file="circle_detection_config.json"):
    """Load configuration from file or create default"""
    try:
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            # Update default config with loaded values
            CONFIG.update(loaded_config)
            print(f"Configuration loaded from {config_file}")
    except FileNotFoundError:
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        print(f"Default configuration saved to {config_file}")
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
