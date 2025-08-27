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
    "min_radius": 5,          # Adjustable with 1/2 keys
    "max_radius": 50,         # Adjustable with 3/4 keys
    "roi_x_min": 150,         # Adjustable with q/w keys
    "roi_x_max": 1100,        # Adjustable with e/r keys
    "roi_y_min": 150,         # Adjustable with a/s keys
    "roi_y_max": 600,         # Adjustable with d/f keys
    
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
        """Apply binary threshold to frame with ROI cropping"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI cropping
        roi_mask = np.zeros_like(gray)
        roi_mask[self.config["roi_y_min"]:self.config["roi_y_max"], 
                 self.config["roi_x_min"]:self.config["roi_x_max"]] = 255
        
        # Apply mask to grayscale image
        gray_roi = cv2.bitwise_and(gray, roi_mask)
        
        # Apply binary threshold
        threshold_value = self.config["binary_threshold"]
        _, binary = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Debug: Print threshold value occasionally to verify it's changing
        # Removed excessive debug printing
        
        return binary
    
    def detect_circles(self, binary_frame):
        """Detect circle-like shapes in binary frame"""
        circles = cv2.HoughCircles(
            binary_frame,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,  # Reduced from 30 - allow closer circles
            param1=50,   # Reduced from 100 - less strict edge detection
            param2=self.config["accumulator_threshold"],
            minRadius=self.config["min_radius"],
            maxRadius=self.config["max_radius"]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return []
    
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
        
        # Draw ROI boundaries
        cv2.rectangle(frame, 
                     (self.config["roi_x_min"], self.config["roi_y_min"]), 
                     (self.config["roi_x_max"], self.config["roi_y_max"]), 
                     (255, 255, 0), 2)  # Yellow ROI rectangle
        
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
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)  # Red circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)   # Red center dot
            
            # Draw line from circle to reference
            cv2.line(frame, (x, y), (ref_x, ref_y), (255, 255, 0), 1)  # Yellow line
            
            # Calculate distance components from circle to reference
            circle_to_ref_x = ref_x - x
            circle_to_ref_y = ref_y - y
            
            # Label the circle with its number
            cv2.putText(frame, f"C{i+1}", (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Store circle information
            circle_info_list.append({
                'id': i+1,
                'x': x,
                'y': y,
                'radius': r,
                'to_ref_x': circle_to_ref_x,
                'to_ref_y': circle_to_ref_y
            })
        
        # Display detection information
        if self.config["show_detection_info"]:
            info_y = 30
            cv2.putText(frame, f"Circles: {len(circles)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Binary threshold: {self.config['binary_threshold']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Reference: ({ref_x}, {ref_y})", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Center-to-Ref X: {center_to_ref_x:+.0f} px", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Center-to-Ref Y: {center_to_ref_y:+.0f} px", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Radius: {self.config['min_radius']}-{self.config['max_radius']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display information for each detected circle
            for circle_info in circle_info_list:
                info_y += 25
                cv2.putText(frame, f"C{circle_info['id']}: ({circle_info['x']},{circle_info['y']}) R={circle_info['radius']}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                info_y += 20
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
        print("Starting circle detection...")
        print(f"Initial settings - Reference: ({self.config['reference_x']}, {self.config['reference_y']}), Binary threshold: {self.config['binary_threshold']}")
        print("="*50)
        print("KEYBOARD CONTROLS:")
        print("  +/- : Adjust binary threshold")
        print("  Arrow keys or T/F/G/H : Move reference position")
        print("  1/2 : Adjust min radius (decrease/increase)")
        print("  3/4 : Adjust max radius (decrease/increase)")
        print("  u/i/o/p : Adjust ROI X borders")
        print("  j/k/l/; : Adjust ROI Y borders")
        print("  s : Save frames")
        print("  c : Reload config file")
        print("  q/ESC : Quit")
        print("="*50)
        print("*** CLICK ON THE VIDEO WINDOW TO ENABLE KEYBOARD CONTROLS ***")
        print("="*50)
        
        # Create window and set mouse callback
        window_name = "Circle Detection - Live Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Ensure window is focused
        cv2.moveWindow(window_name, 100, 100)  # Move window to ensure it's visible
        
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
            detection_data = {
                "total_circles": len(circles),
                "circles": []
            }
            
            if len(circles) > 0:
                ref_x = self.config["reference_x"]
                ref_y = self.config["reference_y"]
                
                # Process each circle
                for i, circle in enumerate(circles):
                    circle_data = self.calculate_distance(circle)
                    circle_data["circle_id"] = i + 1
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
            cv2.imshow(window_name, display_resized)
            
            # Handle keyboard input with comprehensive controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
                
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
                self.config["reference_x"] = min(1280, self.config["reference_x"] + 5)            # Radius controls
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
                
            elif key == ord('c'):  # 'c' to reload config
                try:
                    with open("circle_detection_config.json", 'r') as f:
                        loaded_config = json.load(f)
                        self.config.update(loaded_config)
                    print("🔄 Configuration reloaded from file")
                except Exception as e:
                    print(f"❌ Error reloading config: {e}")
                print(f"Frames saved with timestamp {timestamp}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final parameter summary
        print("\n" + "="*50)
        print("FINAL PARAMETER SETTINGS:")
        print("="*50)
        print(f"Binary threshold: {self.config['binary_threshold']}")
        print(f"Reference position: ({self.config['reference_x']}, {self.config['reference_y']})")
        print(f"Circle radius range: {self.config['min_radius']} - {self.config['max_radius']}")
        print(f"ROI boundaries: X({self.config['roi_x_min']}-{self.config['roi_x_max']}), Y({self.config['roi_y_min']}-{self.config['roi_y_max']})")
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
