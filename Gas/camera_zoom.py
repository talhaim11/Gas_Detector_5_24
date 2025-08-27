"""
🎥 FAST LIVE CAMERA ZOOM INTERFACE
High-performance camera streaming with instant click-to-zoom functionality
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
from PIL import Image, ImageTk

class FastCameraZoom:
    def __init__(self):
        self.setup_window()
        self.setup_variables()
        self.setup_interface()
        
    def setup_window(self):
        """Create main window"""
        self.root = tk.Tk()
        self.root.title("🎥 Fast Live Camera Zoom Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
    def setup_variables(self):
        """Initialize all variables"""
        # Camera settings (your camera details)
        self.ip = "169.254.188.158"
        self.port = "8554"
        self.username = "fgcam"
        self.password = "admin"
        self.stream_path = "/0/unicast"
        
        # Camera variables
        self.cap = None
        self.frame = None
        self.display_frame = None
        self.is_running = False
        self.is_connected = False
        
        # Zoom variables
        self.zoom_factor = 1.0
        self.zoom_center_x = 0
        self.zoom_center_y = 0
        self.original_width = 0
        self.original_height = 0
        self.zoom_target_active = False  # New: Track if zoom target is active
        self.zoom_target_x = 0  # New: Target coordinates for zoom
        self.zoom_target_y = 0
        
        # Enhanced zoom variables for full-resolution scrolling
        self.scroll_offset_x = 0  # Current scroll position in zoomed frame
        self.scroll_offset_y = 0  # Current scroll position in zoomed frame
        self.zoomed_frame_width = 0  # Current zoomed frame dimensions
        self.zoomed_frame_height = 0
        
        # Contour detection variables
        self.contours_enabled = False
        self.detected_contours = []
        self.contour_measurements = []
        self.contour_sensitivity = 50  # Contour detection sensitivity (1-100)
        self.min_contour_area = 100    # Minimum area to detect as contour
        
        # Display variables
        self.display_width = 800
        self.display_height = 600
        self.maintain_aspect_ratio = True
        
        # Frame rate control
        self.target_fps = 30  # Target frames per second
        self.frame_delay = 1.0 / self.target_fps  # Delay between frames
        
    def setup_interface(self):
        """Create the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = tk.Label(control_frame, text="🎥 FAST LIVE CAMERA ZOOM", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#34495e')
        title_label.pack(pady=10)
        
        # Camera info
        self.info_label = tk.Label(control_frame, 
                                  text=f"📹 Camera: {self.ip}:{self.port} | User: {self.username} | Status: Ready", 
                                  font=('Arial', 10), fg='#ecf0f1', bg='#34495e')
        self.info_label.pack(pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(control_frame, bg='#34495e')
        buttons_frame.pack(pady=10)
        
        # Connect button
        self.connect_btn = tk.Button(buttons_frame, text="🚀 CONNECT CAMERA", 
                                   command=self.connect_camera, bg='#27ae60', fg='white',
                                   font=('Arial', 12, 'bold'), width=15, height=2)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        # Disconnect button
        self.disconnect_btn = tk.Button(buttons_frame, text="📴 DISCONNECT", 
                                      command=self.disconnect_camera, bg='#e74c3c', fg='white',
                                      font=('Arial', 12, 'bold'), width=15, height=2, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset zoom button
        self.reset_btn = tk.Button(buttons_frame, text="🔄 RESET ZOOM", 
                                 command=self.reset_zoom, bg='#3498db', fg='white',
                                 font=('Arial', 12, 'bold'), width=15, height=2, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Save frame button
        self.save_btn = tk.Button(buttons_frame, text="📷 SAVE FRAME", 
                                command=self.save_frame, bg='#f39c12', fg='white',
                                font=('Arial', 12, 'bold'), width=15, height=2, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        self.settings_btn = tk.Button(buttons_frame, text="⚙️ SETTINGS", 
                                    command=self.open_settings, bg='#9b59b6', fg='white',
                                    font=('Arial', 12, 'bold'), width=15, height=2)
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Contour detection button
        self.contour_btn = tk.Button(buttons_frame, text="📐 CONTOURS", 
                                   command=self.toggle_contours, bg='#16a085', fg='white',
                                   font=('Arial', 12, 'bold'), width=15, height=2, state=tk.DISABLED)
        self.contour_btn.pack(side=tk.LEFT, padx=5)
        
        # Contour settings button (initially hidden)
        self.contour_settings_btn = tk.Button(buttons_frame, text="⚙️ CONTOUR", 
                                            command=self.open_contour_settings, bg='#8e44ad', fg='white',
                                            font=('Arial', 12, 'bold'), width=15, height=2)
        self.contour_settings_btn.pack(side=tk.LEFT, padx=5)
        self.contour_settings_btn.pack_forget()  # Hide initially
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Click anywhere on video to set zoom target! Use mouse wheel/arrow keys to scroll when zoomed. Middle-click drag to pan smoothly.", 
                                   font=('Arial', 11), fg='#f1c40f', bg='#34495e')
        self.status_label.pack(pady=5)
        
        # Zoom info
        self.zoom_label = tk.Label(control_frame, text="Zoom: 1.0x | Ready to zoom", 
                                 font=('Arial', 10), fg='#bdc3c7', bg='#34495e')
        self.zoom_label.pack(pady=2)
        
        # Video display frame with scrollable area
        video_frame = tk.Frame(main_frame, bg='#2c3e50', relief=tk.SUNKEN, bd=2)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable canvas container
        self.scroll_canvas = tk.Canvas(video_frame, bg='#2c3e50')
        self.scroll_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        v_scrollbar = tk.Scrollbar(video_frame, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(video_frame, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.scroll_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Video canvas inside scrollable area
        self.video_canvas = tk.Canvas(self.scroll_canvas, bg='black', width=self.display_width, height=self.display_height)
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.video_canvas, anchor="center")
        
        # Bind canvas resize
        self.video_canvas.bind('<Configure>', self.on_canvas_configure)
        self.scroll_canvas.bind('<Configure>', self.on_scroll_canvas_configure)
        
        # Create zoom control panel (initially hidden)
        self.zoom_control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        self.zoom_control_frame.pack(fill=tk.X, pady=(5, 0))
        self.zoom_control_frame.pack_forget()  # Hide initially
        
        # Zoom slider and controls
        zoom_title = tk.Label(self.zoom_control_frame, text="🔍 ZOOM CONTROL", 
                             font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        zoom_title.pack(pady=5)
        
        slider_frame = tk.Frame(self.zoom_control_frame, bg='#34495e')
        slider_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(slider_frame, text="1x", fg='white', bg='#34495e', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.zoom_slider = tk.Scale(slider_frame, from_=1.0, to=10.0, resolution=0.1,
                                   orient=tk.HORIZONTAL, bg='#3498db', fg='white',
                                   activebackground='#2980b9', font=('Arial', 9),
                                   command=self.on_zoom_slider_change)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(slider_frame, text="10x", fg='white', bg='#34495e', font=('Arial', 10)).pack(side=tk.RIGHT)
        
        # Zoom info and reset button
        zoom_info_frame = tk.Frame(self.zoom_control_frame, bg='#34495e')
        zoom_info_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.zoom_coords_label = tk.Label(zoom_info_frame, text="Click on video to activate zoom", 
                                         fg='#bdc3c7', bg='#34495e', font=('Arial', 10))
        self.zoom_coords_label.pack(side=tk.LEFT)
        
        self.hide_zoom_btn = tk.Button(zoom_info_frame, text="❌ Hide Zoom", 
                                      command=self.hide_zoom_controls, bg='#e74c3c', fg='white',
                                      font=('Arial', 9), relief=tk.FLAT)
        self.hide_zoom_btn.pack(side=tk.RIGHT)
        
        # Zoom help instructions
        zoom_help_frame = tk.Frame(self.zoom_control_frame, bg='#34495e')
        zoom_help_frame.pack(fill=tk.X, padx=20, pady=(0, 5))
        
        help_text = "🔄 Navigation: Mouse Wheel/Arrow Keys=Pan | Middle-Click+Drag=Smooth Pan | Shift+Wheel=Horizontal"
        self.zoom_help_label = tk.Label(zoom_help_frame, text=help_text, 
                                       fg='#95a5a6', bg='#34495e', font=('Arial', 9))
        self.zoom_help_label.pack(side=tk.LEFT)
        
        # Bind mouse events for zoom and contours
        self.video_canvas.bind("<Button-1>", self.on_canvas_click)  # Left click - activate zoom or detect contour
        self.video_canvas.bind("<Button-3>", self.on_right_click)   # Right click - clear contours
        
        # Bind scroll events for panning when zoomed
        self.video_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.video_canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.video_canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Bind keyboard events for precise panning
        self.video_canvas.bind("<KeyPress>", self.on_key_press)
        self.video_canvas.focus_set()  # Allow keyboard focus
        
        # Bind drag events for smooth panning
        self.video_canvas.bind("<ButtonPress-2>", self.on_drag_start)     # Middle click start
        self.video_canvas.bind("<B2-Motion>", self.on_drag_motion)        # Middle click drag
        self.video_canvas.bind("<ButtonRelease-2>", self.on_drag_end)     # Middle click end
        
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_last_scroll_x = 0
        self.drag_last_scroll_y = 0
        
        # Initialize canvas image reference
        self.canvas_image_id = None
        
    def open_settings(self):
        """Open camera settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Camera Settings")
        settings_window.geometry("450x400")  # Made larger to fit all fields
        settings_window.configure(bg='#34495e')
        settings_window.grab_set()  # Make it modal
        
        # Center the window
        settings_window.transient(self.root)
        
        # Center the window on screen
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (450 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (400 // 2)
        settings_window.geometry(f"450x400+{x}+{y}")
        
        tk.Label(settings_window, text="📹 Camera Configuration", font=('Arial', 14, 'bold'), 
                fg='white', bg='#34495e').pack(pady=15)
        
        # Create a frame for all input fields
        fields_frame = tk.Frame(settings_window, bg='#34495e')
        fields_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # IP Address
        tk.Label(fields_frame, text="IP Address:", fg='white', bg='#34495e', font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        ip_entry = tk.Entry(fields_frame, width=35, font=('Arial', 11))
        ip_entry.insert(0, self.ip)
        ip_entry.pack(pady=(0,10), fill=tk.X)
        
        # Port
        tk.Label(fields_frame, text="Port:", fg='white', bg='#34495e', font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        port_entry = tk.Entry(fields_frame, width=35, font=('Arial', 11))
        port_entry.insert(0, self.port)
        port_entry.pack(pady=(0,10), fill=tk.X)
        
        # Username
        tk.Label(fields_frame, text="Username:", fg='white', bg='#34495e', font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        username_entry = tk.Entry(fields_frame, width=35, font=('Arial', 11))
        username_entry.insert(0, self.username)
        username_entry.pack(pady=(0,10), fill=tk.X)
        
        # Password
        tk.Label(fields_frame, text="Password:", fg='white', bg='#34495e', font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        password_entry = tk.Entry(fields_frame, width=35, font=('Arial', 11), show="*")
        password_entry.insert(0, self.password)
        password_entry.pack(pady=(0,10), fill=tk.X)
        
        # Frame Rate
        tk.Label(fields_frame, text="Frame Rate (FPS):", fg='white', bg='#34495e', font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        fps_entry = tk.Entry(fields_frame, width=35, font=('Arial', 11))
        fps_entry.insert(0, str(self.target_fps))
        fps_entry.pack(pady=(0,15), fill=tk.X)
        
        # Buttons frame at the bottom
        button_frame = tk.Frame(settings_window, bg='#34495e')
        button_frame.pack(side=tk.BOTTOM, pady=20)
        
        def save_settings():
            try:
                self.ip = ip_entry.get().strip()
                self.port = port_entry.get().strip()
                self.username = username_entry.get().strip()
                self.password = password_entry.get()
                
                # Validate inputs
                if not self.ip:
                    messagebox.showwarning("Invalid Input", "IP Address cannot be empty")
                    return
                if not self.port:
                    messagebox.showwarning("Invalid Input", "Port cannot be empty")
                    return
                if not self.username:
                    messagebox.showwarning("Invalid Input", "Username cannot be empty")
                    return
                
                # Update frame rate
                try:
                    new_fps = float(fps_entry.get().strip())
                    if new_fps > 0 and new_fps <= 120:
                        self.target_fps = new_fps
                        self.frame_delay = 1.0 / self.target_fps
                    else:
                        messagebox.showwarning("Invalid FPS", "Frame rate must be between 1 and 120")
                        return
                except ValueError:
                    messagebox.showwarning("Invalid FPS", "Please enter a valid number for frame rate")
                    return
                
                self.update_info_label()
                settings_window.destroy()
                messagebox.showinfo("✅ Settings Saved", 
                                  f"Settings updated successfully!\n\n📹 Camera: {self.ip}:{self.port}\n👤 User: {self.username}\n🎬 FPS: {self.target_fps}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
        
        def cancel_settings():
            settings_window.destroy()
        
        # Larger, more visible buttons
        save_button = tk.Button(button_frame, text="💾 SAVE SETTINGS", command=save_settings, 
                               bg='#27ae60', fg='white', font=('Arial', 12, 'bold'), 
                               width=15, height=2, relief=tk.RAISED, bd=3)
        save_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="❌ CANCEL", command=cancel_settings, 
                                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'), 
                                 width=15, height=2, relief=tk.RAISED, bd=3)
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Make the Save button the default (Enter key)
        save_button.focus_set()
        settings_window.bind('<Return>', lambda event: save_settings())
        settings_window.bind('<Escape>', lambda event: cancel_settings())
    
    def update_info_label(self):
        """Update camera info display"""
        status = "🟢 CONNECTED" if self.is_connected else "🔴 DISCONNECTED"
        self.info_label.config(text=f"📹 Camera: {self.ip}:{self.port} | User: {self.username} | Status: {status}")
    
    def connect_camera(self):
        """Connect to the camera"""
        try:
            # Build RTSP URL with authentication
            camera_url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}{self.stream_path}"
            
            self.status_label.config(text="🔄 Connecting to camera...", fg='#f39c12')
            self.root.update()
            
            # Try to connect
            self.cap = cv2.VideoCapture(camera_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera stream")
            
            # Test if we can read a frame
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Could not read from camera")
            
            # Success!
            self.frame = test_frame
            self.original_height, self.original_width = test_frame.shape[:2]
            self.is_connected = True
            self.is_running = True
            
            # Reset zoom
            self.zoom_factor = 1.0
            self.zoom_center_x = self.original_width // 2
            self.zoom_center_y = self.original_height // 2
            
            # Update UI
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.contour_btn.config(state=tk.NORMAL)
            self.contour_settings_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text="✅ Connected! Click anywhere on video to set zoom target. When zoomed: scroll with mouse wheel or arrow keys, middle-click drag to pan.", fg='#27ae60')
            self.update_info_label()
            self.update_zoom_label()
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect to camera:\n{str(e)}")
            self.status_label.config(text="❌ Connection failed. Check camera settings.", fg='#e74c3c')
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def disconnect_camera(self):
        """Disconnect from camera"""
        self.is_running = False
        self.is_connected = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear canvas and reset image reference
        self.video_canvas.delete("all")
        self.canvas_image_id = None
        
        # Update UI
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.contour_btn.config(state=tk.DISABLED)
        self.contour_settings_btn.config(state=tk.DISABLED)
        
        self.status_label.config(text="📴 Camera disconnected", fg='#e74c3c')
        self.update_info_label()
    
    def video_loop(self):
        """Main video capture and display loop"""
        while self.is_running and self.cap:
            frame_start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.update_display()
                else:
                    print("Failed to read frame")
                    break
            except Exception as e:
                print(f"Video loop error: {e}")
                break
            
            # Frame rate control
            frame_process_time = time.time() - frame_start_time
            if frame_process_time < self.frame_delay:
                time.sleep(self.frame_delay - frame_process_time)
        
        # Cleanup if loop exits
        if self.cap:
            self.cap.release()
    
    def update_display(self):
        """Update the video display with current frame - NO FLICKERING"""
        if self.frame is None:
            return
        
        try:
            # Apply zoom
            zoomed_frame = self.apply_zoom(self.frame)
            
            # Draw zoom target if active
            if self.zoom_target_active and not self.contours_enabled:
                zoomed_frame = self.draw_zoom_target(zoomed_frame)
            
            # Apply contours if enabled
            if self.contours_enabled and len(self.detected_contours) > 0:
                zoomed_frame = self.draw_contours_on_frame(zoomed_frame)
            
            # Calculate display size to maintain aspect ratio and handle zoom
            display_frame = self.prepare_display_frame(zoomed_frame)
            
            # Convert BGR to RGB for tkinter
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(display_frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas size for full scrolling capability
            visible_width = display_frame.shape[1]
            visible_height = display_frame.shape[0]
            
            if self.zoom_factor > 1.0:
                # When zoomed: enable full scrolling through the zoomed image
                # The video canvas shows only the visible portion
                self.video_canvas.configure(width=visible_width, height=visible_height)
                
                # Set scroll region to the full zoomed frame size
                self.scroll_canvas.configure(scrollregion=(0, 0, self.zoomed_frame_width, self.zoomed_frame_height))
                
                # Update scrollbar positions based on current scroll offset
                if self.zoomed_frame_width > visible_width:
                    scroll_fraction_x = self.scroll_offset_x / (self.zoomed_frame_width - visible_width)
                    self.scroll_canvas.xview_moveto(scroll_fraction_x)
                
                if self.zoomed_frame_height > visible_height:
                    scroll_fraction_y = self.scroll_offset_y / (self.zoomed_frame_height - visible_height)
                    self.scroll_canvas.yview_moveto(scroll_fraction_y)
            else:
                # When not zoomed: standard display with no scrolling needed
                self.video_canvas.configure(width=visible_width, height=visible_height)
                self.scroll_canvas.configure(scrollregion=(0, 0, visible_width, visible_height))
            
            # Update canvas WITHOUT deleting - prevents flickering
            if self.canvas_image_id is None:
                # First time - create the image centered in the visible area
                self.canvas_image_id = self.video_canvas.create_image(
                    visible_width//2, visible_height//2, image=photo)
            else:
                # Update existing image - no flickering!
                self.video_canvas.itemconfig(self.canvas_image_id, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.video_canvas.image = photo
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def prepare_display_frame(self, frame):
        """Prepare frame for display with proper scaling and scrolling support"""
        h, w = frame.shape[:2]
        
        if self.zoom_factor > 1.0:
            # When zoomed: extract the visible portion from the full-resolution zoomed frame
            # based on current scroll offsets
            
            # Calculate display area size
            display_w = min(self.display_width, w)
            display_h = min(self.display_height, h)
            
            # Ensure scroll offsets are within bounds
            max_scroll_x = max(0, w - display_w)
            max_scroll_y = max(0, h - display_h)
            self.scroll_offset_x = max(0, min(self.scroll_offset_x, max_scroll_x))
            self.scroll_offset_y = max(0, min(self.scroll_offset_y, max_scroll_y))
            
            # Extract the visible portion
            start_x = self.scroll_offset_x
            start_y = self.scroll_offset_y
            end_x = min(w, start_x + display_w)
            end_y = min(h, start_y + display_h)
            
            # Get the visible portion
            visible_frame = frame[start_y:end_y, start_x:end_x]
            
            # If the visible frame is smaller than display area, pad it
            if visible_frame.shape[1] < display_w or visible_frame.shape[0] < display_h:
                padded_frame = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                padded_frame[0:visible_frame.shape[0], 0:visible_frame.shape[1]] = visible_frame
                return padded_frame
            
            return visible_frame
        else:
            # When not zoomed, scale to fit display area
            return cv2.resize(frame, (self.display_width, self.display_height))
    
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
    
    def on_scroll_canvas_configure(self, event):
        """Handle scroll canvas resize - center the video canvas"""
        # Get the size of the scroll canvas
        canvas_width = event.width
        canvas_height = event.height
        
        # Get the size of the video canvas
        video_width = self.video_canvas.winfo_reqwidth()
        video_height = self.video_canvas.winfo_reqheight()
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Update the window position to center the video canvas
        self.scroll_canvas.coords(self.canvas_window, center_x, center_y)
    
    def apply_zoom(self, frame):
        """Apply zoom to frame - NEW: Returns full-resolution zoomed frame for complete scrolling"""
        h, w = frame.shape[:2]
        
        if self.zoom_factor <= 1.0:
            # No zoom - return original frame
            self.zoomed_frame_width = w
            self.zoomed_frame_height = h
            self.scroll_offset_x = 0
            self.scroll_offset_y = 0
            return frame
        
        # Calculate the size of the full zoomed frame (larger than original)
        self.zoomed_frame_width = int(w * self.zoom_factor)
        self.zoomed_frame_height = int(h * self.zoom_factor)
        
        # Use zoom target as center point for precise targeting
        if self.zoom_target_active:
            center_x = self.zoom_target_x
            center_y = self.zoom_target_y
        else:
            center_x = self.zoom_center_x
            center_y = self.zoom_center_y
        
        # Calculate the region of the original frame to extract for zooming
        zoom_w = int(w / self.zoom_factor)
        zoom_h = int(h / self.zoom_factor)
        
        # Calculate zoom window position centered on target
        start_x = max(0, center_x - zoom_w // 2)
        start_y = max(0, center_y - zoom_h // 2)
        end_x = min(w, start_x + zoom_w)
        end_y = min(h, start_y + zoom_h)
        
        # Adjust if window goes out of bounds
        if end_x - start_x < zoom_w:
            start_x = max(0, end_x - zoom_w)
        if end_y - start_y < zoom_h:
            start_y = max(0, end_y - zoom_h)
        
        # Extract the region and resize to full zoomed resolution
        zoomed_region = frame[start_y:end_y, start_x:end_x]
        
        # Scale up to full zoomed resolution - this gives us ALL the pixels!
        zoomed_full_res = cv2.resize(zoomed_region, (self.zoomed_frame_width, self.zoomed_frame_height), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Initialize scroll offset to center the target in the view when first zooming
        if not hasattr(self, '_zoom_initialized') or self.zoom_factor == 1.0:
            # Center the scroll on the target
            display_w = min(self.display_width, self.zoomed_frame_width)
            display_h = min(self.display_height, self.zoomed_frame_height)
            
            # Calculate where the target should be in the zoomed frame
            target_x_zoomed = int((center_x - start_x) / zoom_w * self.zoomed_frame_width)
            target_y_zoomed = int((center_y - start_y) / zoom_h * self.zoomed_frame_height)
            
            # Center the scroll on the target
            self.scroll_offset_x = max(0, min(target_x_zoomed - display_w // 2, 
                                            self.zoomed_frame_width - display_w))
            self.scroll_offset_y = max(0, min(target_y_zoomed - display_h // 2, 
                                            self.zoomed_frame_height - display_h))
            self._zoom_initialized = True
        
        return zoomed_full_res
    
    def on_canvas_click(self, event):
        """Handle canvas click - set zoom target or detect contour at clicked point"""
        if not self.is_connected:
            return
        
        # Convert canvas coordinates to frame coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Get current display dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        # Ensure click is within canvas bounds
        if canvas_x < 0 or canvas_x > canvas_width or canvas_y < 0 or canvas_y > canvas_height:
            return
        
        # Map to original frame coordinates with NEW scrolling system
        if self.zoom_factor > 1.0:
            # When zoomed: account for scroll offset and full-resolution mapping
            
            # Add scroll offset to get position in full zoomed frame
            zoomed_x = canvas_x + self.scroll_offset_x
            zoomed_y = canvas_y + self.scroll_offset_y
            
            # Map from zoomed frame coordinates back to original frame coordinates
            # Calculate the original zoom window
            zoom_w = int(self.original_width / self.zoom_factor)
            zoom_h = int(self.original_height / self.zoom_factor)
            
            # Get the zoom center
            if self.zoom_target_active:
                center_x = self.zoom_target_x
                center_y = self.zoom_target_y
            else:
                center_x = self.zoom_center_x
                center_y = self.zoom_center_y
            
            # Calculate zoom window position in original frame
            zoom_start_x = max(0, center_x - zoom_w // 2)
            zoom_start_y = max(0, center_y - zoom_h // 2)
            zoom_end_x = min(self.original_width, zoom_start_x + zoom_w)
            zoom_end_y = min(self.original_height, zoom_start_y + zoom_h)
            
            # Adjust if window goes out of bounds
            if zoom_end_x - zoom_start_x < zoom_w:
                zoom_start_x = max(0, zoom_end_x - zoom_w)
            if zoom_end_y - zoom_start_y < zoom_h:
                zoom_start_y = max(0, zoom_end_y - zoom_h)
            
            # Map from zoomed frame back to original coordinates
            ratio_x = zoomed_x / self.zoomed_frame_width
            ratio_y = zoomed_y / self.zoomed_frame_height
            
            frame_x = int(zoom_start_x + (ratio_x * (zoom_end_x - zoom_start_x)))
            frame_y = int(zoom_start_y + (ratio_y * (zoom_end_y - zoom_start_y)))
        else:
            # When not zoomed, map from display to original frame
            frame_x = int((canvas_x / canvas_width) * self.original_width)
            frame_y = int((canvas_y / canvas_height) * self.original_height)
        
        if self.contours_enabled:
            # Detect contour at clicked point (DO NOT reset zoom)
            self.detect_contour_at_point(frame_x, frame_y)
        else:
            # Set zoom target and activate zoom controls
            self.zoom_target_x = frame_x
            self.zoom_target_y = frame_y
            self.zoom_target_active = True
            
            # Update zoom center to target (for smooth transition)
            self.zoom_center_x = frame_x
            self.zoom_center_y = frame_y
            
            # Show zoom controls
            self.show_zoom_controls()
            
            # Update slider to current zoom level
            self.zoom_slider.set(self.zoom_factor)
            
            # Update info
            self.update_zoom_info()
    
    def on_right_click(self, event):
        """Handle right click - clear contours"""
        if self.contours_enabled:
            self.detected_contours = []
            self.contour_measurements = []
            self.status_label.config(text="🧹 Contours cleared. Click on objects to detect new contours.", fg='#3498db')
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for panning when zoomed"""
        if not self.is_connected or self.zoom_factor <= 1.0:
            return
        
        # Determine scroll direction and amount
        if event.num == 4 or event.delta > 0:
            # Scroll up - pan up
            delta_y = -20
            delta_x = 0
        elif event.num == 5 or event.delta < 0:
            # Scroll down - pan down
            delta_y = 20
            delta_x = 0
        else:
            return
        
        # Apply horizontal scrolling with Shift key
        if hasattr(event, 'state') and event.state & 0x1:  # Shift key pressed
            delta_x = delta_y
            delta_y = 0
        
        # Update scroll offsets
        self.update_scroll_offset(delta_x, delta_y)
    
    def on_key_press(self, event):
        """Handle key press for precise panning"""
        if not self.is_connected or self.zoom_factor <= 1.0:
            return
        
        delta_x = 0
        delta_y = 0
        step = 10  # Pixels per key press
        
        if event.keysym == 'Up':
            delta_y = -step
        elif event.keysym == 'Down':
            delta_y = step
        elif event.keysym == 'Left':
            delta_x = -step
        elif event.keysym == 'Right':
            delta_x = step
        else:
            return
        
        self.update_scroll_offset(delta_x, delta_y)
    
    def on_drag_start(self, event):
        """Start drag operation for smooth panning"""
        if not self.is_connected or self.zoom_factor <= 1.0:
            return
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_last_scroll_x = self.scroll_offset_x
        self.drag_last_scroll_y = self.scroll_offset_y
        self.video_canvas.config(cursor="fleur")  # Change cursor to indicate dragging
    
    def on_drag_motion(self, event):
        """Handle drag motion for smooth panning"""
        if not self.is_connected or self.zoom_factor <= 1.0:
            return
        
        # Calculate movement
        delta_x = self.drag_start_x - event.x
        delta_y = self.drag_start_y - event.y
        
        # Update scroll position
        new_scroll_x = self.drag_last_scroll_x + delta_x
        new_scroll_y = self.drag_last_scroll_y + delta_y
        
        self.scroll_offset_x = new_scroll_x
        self.scroll_offset_y = new_scroll_y
        
        # Apply bounds
        self.update_scroll_offset(0, 0)  # This will apply bounds checking
    
    def on_drag_end(self, event):
        """End drag operation"""
        self.video_canvas.config(cursor="")  # Reset cursor
    
    def update_scroll_offset(self, delta_x, delta_y):
        """Update scroll offsets with bounds checking"""
        if self.zoom_factor <= 1.0:
            return
        
        # Update offsets
        self.scroll_offset_x += delta_x
        self.scroll_offset_y += delta_y
        
        # Apply bounds checking
        display_w = min(self.display_width, self.zoomed_frame_width)
        display_h = min(self.display_height, self.zoomed_frame_height)
        
        max_scroll_x = max(0, self.zoomed_frame_width - display_w)
        max_scroll_y = max(0, self.zoomed_frame_height - display_h)
        
        self.scroll_offset_x = max(0, min(self.scroll_offset_x, max_scroll_x))
        self.scroll_offset_y = max(0, min(self.scroll_offset_y, max_scroll_y))
        
        # Update zoom info display
        self.update_zoom_info()
    
    def toggle_contours(self):
        """Toggle contour detection mode"""
        self.contours_enabled = not self.contours_enabled
        
        if self.contours_enabled:
            self.contour_btn.config(text="📐 DISABLE", bg='#e74c3c')
            # Show contour settings button
            self.contour_settings_btn.pack(side=tk.LEFT, padx=5)
            self.contour_settings_btn.config(state=tk.NORMAL)
            self.status_label.config(text="🎯 Contour mode ON! Click on objects to detect and measure contours. Zoom is preserved! Right-click to clear.", fg='#16a085')
            # Keep zoom controls visible when in contour mode (DO NOT HIDE)
        else:
            self.contour_btn.config(text="📐 CONTOURS", bg='#16a085')
            # Hide contour settings button
            self.contour_settings_btn.pack_forget()
            self.status_label.config(text="✅ Connected! Click anywhere on video to set zoom target. When zoomed: scroll with mouse wheel or arrow keys, middle-click drag to pan.", fg='#27ae60')
            # Clear existing contours
            self.detected_contours = []
            self.contour_measurements = []
    
    def open_contour_settings(self):
        """Open contour detection settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("📐 Contour Detection Settings")
        settings_window.geometry("400x350")
        settings_window.configure(bg='#34495e')
        settings_window.grab_set()
        settings_window.transient(self.root)
        
        # Center the window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (350 // 2)
        settings_window.geometry(f"400x350+{x}+{y}")
        
        tk.Label(settings_window, text="📐 Contour Detection Settings", 
                font=('Arial', 14, 'bold'), fg='white', bg='#34495e').pack(pady=15)
        
        # Settings frame
        settings_frame = tk.Frame(settings_window, bg='#34495e')
        settings_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Sensitivity setting
        tk.Label(settings_frame, text="Detection Sensitivity:", 
                fg='white', bg='#34495e', font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(10,5))
        
        sensitivity_frame = tk.Frame(settings_frame, bg='#34495e')
        sensitivity_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sensitivity_frame, text="Low", fg='white', bg='#34495e').pack(side=tk.LEFT)
        
        sensitivity_var = tk.IntVar(value=self.contour_sensitivity)
        sensitivity_scale = tk.Scale(sensitivity_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                   variable=sensitivity_var, bg='#3498db', fg='white',
                                   activebackground='#2980b9')
        sensitivity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(sensitivity_frame, text="High", fg='white', bg='#34495e').pack(side=tk.RIGHT)
        
        # Minimum area setting
        tk.Label(settings_frame, text="Minimum Object Size (pixels):", 
                fg='white', bg='#34495e', font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(15,5))
        
        area_frame = tk.Frame(settings_frame, bg='#34495e')
        area_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(area_frame, text="10", fg='white', bg='#34495e').pack(side=tk.LEFT)
        
        area_var = tk.IntVar(value=self.min_contour_area)
        area_scale = tk.Scale(area_frame, from_=10, to=5000, orient=tk.HORIZONTAL,
                            variable=area_var, bg='#e67e22', fg='white',
                            activebackground='#d35400')
        area_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(area_frame, text="5000", fg='white', bg='#34495e').pack(side=tk.RIGHT)
        
        # Info labels
        tk.Label(settings_frame, text="• Higher sensitivity detects more objects but may be noisy", 
                fg='#bdc3c7', bg='#34495e', font=('Arial', 9)).pack(anchor=tk.W, pady=(10,2))
        tk.Label(settings_frame, text="• Lower sensitivity detects only clear, distinct objects", 
                fg='#bdc3c7', bg='#34495e', font=('Arial', 9)).pack(anchor=tk.W, pady=2)
        tk.Label(settings_frame, text="• Minimum size filters out small noise and artifacts", 
                fg='#bdc3c7', bg='#34495e', font=('Arial', 9)).pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = tk.Frame(settings_window, bg='#34495e')
        button_frame.pack(side=tk.BOTTOM, pady=20)
        
        def apply_settings():
            self.contour_sensitivity = sensitivity_var.get()
            self.min_contour_area = area_var.get()
            # Clear existing contours to apply new settings
            self.detected_contours = []
            self.contour_measurements = []
            settings_window.destroy()
            self.status_label.config(text=f"⚙️ Contour settings updated! Sensitivity: {self.contour_sensitivity}, Min size: {self.min_contour_area}px", fg='#8e44ad')
        
        def reset_defaults():
            sensitivity_var.set(50)
            area_var.set(100)
        
        tk.Button(button_frame, text="✅ APPLY", command=apply_settings,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="🔄 RESET", command=reset_defaults,
                 bg='#f39c12', fg='white', font=('Arial', 11, 'bold'), width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="❌ CANCEL", command=settings_window.destroy,
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), width=12).pack(side=tk.LEFT, padx=10)
    
    def detect_contour_at_point(self, x, y):
        """Detect contour around the clicked point - optimized for zoomed object movement detection"""
        if self.frame is None:
            return
        
        try:
            # Get current frame (use the zoomed frame if we're in zoom mode for better precision)
            if self.zoom_factor > 1.0:
                # Use the zoomed frame for more precise detection
                frame = self.apply_zoom(self.frame.copy())
                
                # Recalculate coordinates for the zoomed frame
                # Map from original frame coordinates to zoomed frame coordinates
                zoom_w = int(self.original_width / self.zoom_factor)
                zoom_h = int(self.original_height / self.zoom_factor)
                zoom_start_x = max(0, self.zoom_center_x - zoom_w // 2)
                zoom_start_y = max(0, self.zoom_center_y - zoom_h // 2)
                
                # Convert original frame coordinates to zoomed frame coordinates
                x_zoomed = int(((x - zoom_start_x) / zoom_w) * self.original_width)
                y_zoomed = int(((y - zoom_start_y) / zoom_h) * self.original_height)
                
                # Use zoomed coordinates
                x, y = x_zoomed, y_zoomed
            else:
                frame = self.frame.copy()
            
            h, w = frame.shape[:2]
            
            # Validate click coordinates
            if x < 0 or x >= w or y < 0 or y >= h:
                self.status_label.config(text="❌ Click point is outside frame bounds.", fg='#e74c3c')
                return
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # For small movement detection, use smaller ROI and higher precision
            if self.zoom_factor > 1.0:
                # Smaller, more precise ROI for zoomed detection
                roi_size = max(30, min(80, min(w, h) // 8))
            else:
                # Standard ROI for normal view
                roi_size = max(50, min(150, min(w, h) // 6))
            
            x1 = max(0, x - roi_size)
            y1 = max(0, y - roi_size)
            x2 = min(w, x + roi_size)
            y2 = min(h, y + roi_size)
            
            # Extract ROI
            roi_gray = gray[y1:y2, x1:x2]
            
            if roi_gray.size == 0:
                self.status_label.config(text="❌ Invalid region selected.", fg='#e74c3c')
                return
            
            # Enhanced detection for movement tracking
            
            # Method 1: High-precision Canny for fine details
            blur_kernel = 3 if self.zoom_factor > 1.0 else max(3, 9 - (self.contour_sensitivity // 15))
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            blurred = cv2.GaussianBlur(roi_gray, (blur_kernel, blur_kernel), 0)
            
            # For movement detection, use more sensitive thresholds
            if self.zoom_factor > 1.0:
                # Higher sensitivity for zoomed objects
                sensitivity_multiplier = 1.5
            else:
                sensitivity_multiplier = 1.0
            
            # Dynamic Canny thresholds
            mean_val = np.mean(blurred)
            std_val = np.std(blurred)
            lower_thresh = max(5, int(mean_val - (std_val * (self.contour_sensitivity / 40) * sensitivity_multiplier)))
            upper_thresh = min(255, int(mean_val + (std_val * (self.contour_sensitivity / 20) * sensitivity_multiplier)))
            
            canny1 = cv2.Canny(blurred, lower_thresh, upper_thresh)
            
            # Method 2: Laplacian for fine edge detection (good for small movements)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian_edges = np.uint8(np.absolute(laplacian))
            _, laplacian_thresh = cv2.threshold(laplacian_edges, self.contour_sensitivity, 255, cv2.THRESH_BINARY)
            
            # Method 3: Sobel with higher precision
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sobel_thresh = (self.contour_sensitivity / 100) * 255 * sensitivity_multiplier
            sobel_edges = (sobel_combined > sobel_thresh).astype(np.uint8) * 255
            
            # Combine methods with emphasis on fine details
            combined_edges = cv2.bitwise_or(canny1, laplacian_thresh)
            combined_edges = cv2.bitwise_or(combined_edges, sobel_edges.astype(np.uint8))
            
            # Morphological operations - less aggressive for small objects
            if self.zoom_factor > 1.0:
                kernel_size = 2  # Smaller kernel for zoomed objects
            else:
                kernel_size = max(2, 5 - (self.contour_sensitivity // 25))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Gentle morphological operations to preserve small details
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Adaptive minimum area based on zoom level
            if self.zoom_factor > 1.0:
                # Smaller minimum area for zoomed detection
                effective_min_area = max(10, self.min_contour_area // (self.zoom_factor * 2))
            else:
                effective_min_area = self.min_contour_area
            
            # Filter contours by area and quality
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= effective_min_area:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 10:  # Minimum perimeter
                        # Additional quality check - circularity for better object detection
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        if circularity > 0.1:  # Filter out very irregular shapes
                            valid_contours.append(contour)
            
            if not valid_contours:
                self.status_label.config(text="❌ No valid objects found. Try: 1) Click closer to object edges 2) Adjust sensitivity 3) Zoom in for better detection", fg='#e74c3c')
                return
            
            # Find the best contour with improved scoring for movement detection
            click_point_roi = (x - x1, y - y1)
            best_contour = None
            best_score = float('inf')
            
            for contour in valid_contours:
                # Distance to contour
                point_distance = cv2.pointPolygonTest(contour, click_point_roi, True)
                
                # Distance to centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid_distance = np.sqrt((click_point_roi[0] - cx) ** 2 + (click_point_roi[1] - cy) ** 2)
                else:
                    centroid_distance = float('inf')
                
                # For movement detection, prioritize contours close to click point
                if point_distance >= 0:  # Point is inside contour
                    score = abs(point_distance) * 0.05  # Very low score for inside points
                else:
                    score = abs(point_distance) + (centroid_distance * 0.2)
                
                if score < best_score:
                    best_score = score
                    best_contour = contour
            
            if best_contour is not None:
                # Convert contour coordinates back to original frame coordinates
                if self.zoom_factor > 1.0:
                    # Convert from zoomed frame back to original frame coordinates
                    zoom_w = int(self.original_width / self.zoom_factor)
                    zoom_h = int(self.original_height / self.zoom_factor)
                    zoom_start_x = max(0, self.zoom_center_x - zoom_w // 2)
                    zoom_start_y = max(0, self.zoom_center_y - zoom_h // 2)
                    
                    # Adjust contour coordinates
                    best_contour_global = best_contour.copy()
                    best_contour_global[:, 0, 0] = (best_contour[:, 0, 0] / self.original_width) * zoom_w + zoom_start_x
                    best_contour_global[:, 0, 1] = (best_contour[:, 0, 1] / self.original_height) * zoom_h + zoom_start_y
                    best_contour_global = best_contour_global.astype(np.int32)
                else:
                    best_contour_global = best_contour + np.array([x1, y1])
                
                # Calculate measurements
                area = cv2.contourArea(best_contour_global)
                perimeter = cv2.arcLength(best_contour_global, True)
                
                # Get bounding rectangle
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(best_contour_global)
                
                # Calculate measurements
                aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
                extent = float(area) / (w_rect * h_rect) if w_rect * h_rect > 0 else 0
                
                # Calculate hull and solidity
                hull = cv2.convexHull(best_contour_global)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Store contour and measurements
                measurement = {
                    'contour': best_contour_global,
                    'area': area,
                    'perimeter': perimeter,
                    'width': w_rect,
                    'height': h_rect,
                    'center_x': x_rect + w_rect // 2,
                    'center_y': y_rect + h_rect // 2,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'solidity': solidity,
                    'zoom_level': self.zoom_factor  # Track zoom level for movement analysis
                }
                
                # Limit to 3 contours for movement tracking (less clutter)
                if len(self.detected_contours) >= 3:
                    self.detected_contours.pop(0)
                    self.contour_measurements.pop(0)
                
                self.detected_contours.append(best_contour_global)
                self.contour_measurements.append(measurement)
                
                # Update status with movement-focused info
                zoom_info = f" | Zoom: {self.zoom_factor:.1f}x" if self.zoom_factor > 1.0 else ""
                self.status_label.config(
                    text=f"🎯 Object #{len(self.detected_contours)} tracked! Area: {area:.0f}px² | Size: {w_rect}×{h_rect}px | Center: ({measurement['center_x']}, {measurement['center_y']}){zoom_info}", 
                    fg='#16a085'
                )
                
                print(f"Object tracked at zoom {self.zoom_factor:.1f}x: Area={area:.0f}, Size={w_rect}x{h_rect}, Center=({measurement['center_x']}, {measurement['center_y']})")
            else:
                self.status_label.config(text="❌ No object found near click point. For movement tracking: 1) Zoom in closer 2) Click on object edges 3) Adjust sensitivity", fg='#e74c3c')
            
        except Exception as e:
            print(f"Contour detection error: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"❌ Detection error: {str(e)}", fg='#e74c3c')
    
    def draw_contours_on_frame(self, frame):
        """Draw detected contours and measurements - optimized for movement tracking"""
        if not self.detected_contours:
            return frame
        
        # Create a copy to draw on
        result_frame = frame.copy()
        
        # Draw each detected contour with movement tracking focus
        for i, (contour, measurement) in enumerate(zip(self.detected_contours, self.contour_measurements)):
            # Different colors for different contours - brighter for movement tracking
            colors = [
                (0, 255, 0),      # Bright Green
                (0, 255, 255),    # Cyan
                (255, 0, 255),    # Magenta
            ]
            color = colors[i % len(colors)]
            
            # Draw contour with thicker line for better visibility
            cv2.drawContours(result_frame, [contour], -1, color, 4)
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point with crosshair for precise tracking
            center_x, center_y = measurement['center_x'], measurement['center_y']
            
            # Center crosshair for movement tracking
            crosshair_size = 15
            cv2.line(result_frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), color, 3)
            cv2.line(result_frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), color, 3)
            cv2.circle(result_frame, (center_x, center_y), 5, color, -1)
            
            # Add object number with larger font
            cv2.putText(result_frame, f"#{i+1}", 
                       (x - 25, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            # Movement tracking focused text
            text_x = x
            text_y = y - 20 if y > 80 else y + h + 25
            
            # Create compact text for movement tracking
            zoom_level = measurement.get('zoom_level', 1.0)
            text_lines = [
                f"Center: ({center_x}, {center_y})",
                f"Size: {w}×{h}px | Area: {measurement['area']:.0f}px²",
                f"Zoom: {zoom_level:.1f}x | Aspect: {measurement['aspect_ratio']:.2f}"
            ]
            
            # Calculate text background size
            max_text_width = 0
            text_height = 22
            for line in text_lines:
                (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_text_width = max(max_text_width, text_w)
            
            # Draw semi-transparent background
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (text_x - 8, text_y - 8), 
                         (text_x + max_text_width + 16, text_y + len(text_lines) * text_height + 8), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, result_frame, 0.2, 0, result_frame)
            
            # Draw text lines with larger font for better readability
            for j, line in enumerate(text_lines):
                cv2.putText(result_frame, line, 
                           (text_x, text_y + j * text_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add tracking summary at the top
        if len(self.detected_contours) > 0:
            summary_text = f"🎯 Tracking {len(self.detected_contours)} object(s) | Zoom: {self.zoom_factor:.1f}x | Sensitivity: {self.contour_sensitivity}"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_frame, (10, 10), (text_w + 25, text_h + 25), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (10, 10), (text_w + 25, text_h + 25), (0, 255, 0), 3)
            
            # Summary text
            cv2.putText(result_frame, summary_text, (18, text_h + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show movement tracking instructions
            if self.zoom_factor > 1.0:
                instruction_text = "🔍 Zoomed: Perfect for small movement detection"
            else:
                instruction_text = "💡 Tip: Zoom in for precise movement tracking"
            
            cv2.putText(result_frame, instruction_text, (18, text_h + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result_frame
    
    def draw_zoom_target(self, frame):
        """Draw zoom target crosshair on the frame"""
        if not self.zoom_target_active:
            return frame
        
        result_frame = frame.copy()
        h, w = result_frame.shape[:2]
        
        # When zoomed, the target should be at the center since we zoom TO the target
        if self.zoom_factor > 1.0:
            # Target is at center when zoomed
            target_x = w // 2
            target_y = h // 2
        else:
            # When not zoomed, show target at actual position
            target_x = self.zoom_target_x
            target_y = self.zoom_target_y
        
        # Ensure target is within frame bounds
        if 0 <= target_x < w and 0 <= target_y < h:
            # Draw crosshair target
            color = (0, 255, 255)  # Bright yellow
            thickness = 3
            size = 25
            
            # Horizontal line
            cv2.line(result_frame, (target_x - size, target_y), (target_x + size, target_y), color, thickness)
            # Vertical line  
            cv2.line(result_frame, (target_x, target_y - size), (target_x, target_y + size), color, thickness)
            
            # Center circle
            cv2.circle(result_frame, (target_x, target_y), 8, color, thickness)
            cv2.circle(result_frame, (target_x, target_y), 3, color, -1)
            
            # Add target info
            zoom_text = f"🎯 Target: ({self.zoom_target_x}, {self.zoom_target_y}) | Zoom: {self.zoom_factor:.1f}x"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(zoom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_frame, (10, h - text_h - 20), (text_w + 20, h - 5), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (10, h - text_h - 20), (text_w + 20, h - 5), color, 2)
            
            # Target text
            cv2.putText(result_frame, zoom_text, (15, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def on_zoom_slider_change(self, value):
        """Handle zoom slider changes"""
        if not self.is_connected:
            return
        
        old_zoom = self.zoom_factor
        self.zoom_factor = float(value)
        
        # If zoom level changed significantly, reset the zoom initialization
        if abs(old_zoom - self.zoom_factor) > 0.1:
            self._zoom_initialized = False
        
        self.update_zoom_info()
    
    def show_zoom_controls(self):
        """Show the zoom control panel"""
        self.zoom_control_frame.pack(fill=tk.X, pady=(5, 0), before=self.video_canvas.master)
    
    def hide_zoom_controls(self):
        """Hide the zoom control panel and reset zoom"""
        self.zoom_control_frame.pack_forget()
        self.zoom_factor = 1.0
        self.zoom_slider.set(1.0)
        self.update_zoom_label()
    
    def update_zoom_info(self):
        """Update zoom information in the control panel"""
        if self.zoom_target_active:
            if self.zoom_factor > 1.0:
                # Show scroll position when zoomed
                scroll_info = f" | Scroll: ({self.scroll_offset_x}, {self.scroll_offset_y})"
                resolution_info = f" | View: {self.zoomed_frame_width}×{self.zoomed_frame_height}px"
            else:
                scroll_info = ""
                resolution_info = ""
            
            self.zoom_coords_label.config(
                text=f"🎯 Target: ({self.zoom_target_x}, {self.zoom_target_y}) | Zoom: {self.zoom_factor:.1f}x{scroll_info}{resolution_info}"
            )
        else:
            scroll_info = f" | Scroll: ({self.scroll_offset_x}, {self.scroll_offset_y})" if self.zoom_factor > 1.0 else ""
            self.zoom_coords_label.config(
                text=f"Zoom: {self.zoom_factor:.1f}x | FPS: {self.target_fps}{scroll_info}"
            )
        # Also update the main zoom label
        self.update_zoom_label()
    
    def reset_zoom(self):
        """Reset zoom to normal view"""
        self.zoom_factor = 1.0
        self.zoom_center_x = self.original_width // 2 if self.original_width > 0 else 0
        self.zoom_center_y = self.original_height // 2 if self.original_height > 0 else 0
        self.zoom_target_active = False  # Clear target
        self.zoom_target_x = 0
        self.zoom_target_y = 0
        
        # Reset scroll offsets for new zoom system
        self.scroll_offset_x = 0
        self.scroll_offset_y = 0
        self.zoomed_frame_width = 0
        self.zoomed_frame_height = 0
        self._zoom_initialized = False
        
        # Update slider if zoom controls are visible
        if self.zoom_control_frame.winfo_viewable():
            self.zoom_slider.set(1.0)
            self.update_zoom_info()
        else:
            self.update_zoom_label()
        
        # Hide zoom controls
        self.hide_zoom_controls()
    
    def save_frame(self):
        """Save current frame"""
        if self.frame is None:
            messagebox.showwarning("Save Frame", "No frame to save!")
            return
        
        try:
            # Get current display frame (with zoom applied)
            zoomed_frame = self.apply_zoom(self.frame)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_frame_{timestamp}_zoom{self.zoom_factor:.1f}x.jpg"
            
            # Save frame
            cv2.imwrite(filename, zoomed_frame)
            
            messagebox.showinfo("Save Frame", f"Frame saved as: {filename}")
            self.status_label.config(text=f"📷 Frame saved: {filename}", fg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save frame:\n{str(e)}")
    
    def update_zoom_label(self):
        """Update zoom information display"""
        if self.zoom_target_active:
            if self.zoom_factor > 1.0:
                scroll_info = f" | Scroll: ({self.scroll_offset_x}, {self.scroll_offset_y})"
            else:
                scroll_info = ""
            self.zoom_label.config(text=f"🎯 Target Zoom: {self.zoom_factor:.1f}x | Target: ({self.zoom_target_x}, {self.zoom_target_y}){scroll_info} | FPS: {self.target_fps}")
        else:
            if self.zoom_factor > 1.0:
                scroll_info = f" | Scroll: ({self.scroll_offset_x}, {self.scroll_offset_y})"
            else:
                scroll_info = ""
            self.zoom_label.config(text=f"Zoom: {self.zoom_factor:.1f}x{scroll_info} | Ready to zoom | FPS: {self.target_fps}")
    
    def set_frame_rate(self, fps):
        """Set new frame rate"""
        if fps > 0:
            self.target_fps = fps
            self.frame_delay = 1.0 / self.target_fps
            self.update_zoom_label()
            print(f"Frame rate set to {fps} FPS")
    
    def run(self):
        """Start the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    print("🚀 Starting Fast Camera Zoom Interface...")
    app = FastCameraZoom()
    app.run()
