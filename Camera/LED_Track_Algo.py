import numpy as np
import cv2
from scipy.ndimage import shift
from scipy.signal import find_peaks
import time  # Add time import for reconnection logic

# Connect to RTSP live stream with authentication
# Replace 'username' and 'password' with actual credentials
username = "fgcam"  # Change this to your camera username
password = "admin"  # Change this to your camera password
rtsp_url = f"rtsp://{username}:{password}@169.254.160.162:8554/0/unicast"

cap = cv2.VideoCapture(rtsp_url)

# Set buffer size to reduce latency for live stream
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Additional settings for better performance and stability
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)  # Reduced timeout to 1 second
cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce processing load

# Network stream settings for better stability
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Request lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#cap=cv2.VideoCapture(r"C:\Users\user\Videos\vlc-record-2024-09-23-11h59m43s-Converting rtsp___fgcam_admin@169.254.118.8_8554_0_unicast-.mp4")

# PARAMS & DEFINITIONS
img_counter = 0
fps = 15  # Reduced from 25 to 15 for better performance
calc_length = 45  # Reduced proportionally (60*15/25)
# Removed frame_skip to prevent switching between frames
processing_counter = 0

# Stream reconnection variables
reconnect_attempts = 0
max_reconnect_attempts = 5
last_successful_frame_time = 0
stream_timeout_threshold = 3.0  # seconds

def reconnect_stream(rtsp_url, max_attempts=3):
    """Attempt to reconnect to the RTSP stream"""
    for attempt in range(max_attempts):
        print(f"Reconnection attempt {attempt + 1}/{max_attempts}...")
        
        # Close existing connection
        cap.release()
        time.sleep(1)  # Wait before reconnecting
        
        # Create new connection
        new_cap = cv2.VideoCapture(rtsp_url)
        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        new_cap.set(cv2.CAP_PROP_FPS, 15)
        new_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
        
        # Test the connection
        ret, test_frame = new_cap.read()
        if ret and test_frame is not None:
            print("✅ Reconnected successfully!")
            return new_cap
        else:
            print(f"❌ Reconnection attempt {attempt + 1} failed")
            new_cap.release()
    
    print("❌ All reconnection attempts failed")
    return None
T = 0
Led_freq = 4.4
Box_size = 100  # Adjusted box size for better precision
sBox_size = 20
MAG=20
freq_counter_th = 5

freq_th = 0.25
Mode = 1
max_history_length = 30
pixel_count_idx = 0
P = []
freq = []
freq_history = []
frame_count = 0  # To track how many frames we've processed so far

# Counters
scounter = 0
tcounter = 0
m2counter = 0

# Get frame parameters
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer setup (if needed for saving)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# Calculate grid size
WI = int(frameWidth / Box_size)
HE = int(frameHeight / Box_size)
sWI = int((Box_size+2*MAG)/ sBox_size)
sHE = int((Box_size+2*MAG)/ sBox_size)

# Initialize necessary buffers and counters
pixel_count_buff = np.zeros(calc_length)
freq_counter = np.zeros((HE, WI))
fourier_buff = np.zeros((calc_length, HE, WI))
sfourier_buff = np.zeros((calc_length, sHE, sWI))




# Initialize tracker - using available tracker for OpenCV 4.10.0
try:
    # Try MIL tracker (available in OpenCV 4.10.0)
    tracker = cv2.TrackerMIL_create()
except AttributeError:
    try:
        # Try GOTURN tracker as alternative
        tracker = cv2.TrackerGOTURN_create()
    except AttributeError:
        try:
            # Try DaSiamRPN tracker as fallback
            tracker = cv2.TrackerDaSiamRPN_create()
        except AttributeError:
            print("No compatible tracker found!")
            exit(1)

# Initialize timing for reconnection logic
last_successful_frame_time = time.time()

while True:
    ret, frame = cap.read()
    current_time = time.time()
    
    if not ret or frame is None:
        # Check if we've been without frames for too long
        time_since_last_frame = current_time - last_successful_frame_time
        
        if time_since_last_frame > stream_timeout_threshold:
            print(f"Stream timeout detected ({time_since_last_frame:.1f}s). Attempting reconnection...")
            
            # Try to reconnect
            new_cap = reconnect_stream(rtsp_url, max_reconnect_attempts)
            if new_cap is not None:
                cap = new_cap
                last_successful_frame_time = time.time()
                continue
            else:
                print("Failed to reconnect. Exiting...")
                break
        else:
            # Short timeout, just continue trying
            time.sleep(0.1)
            continue
    else:
        # Successfully got a frame
        last_successful_frame_time = current_time

    # Drop frames if processing is too slow (prevent buffer buildup)
    frames_to_drop = 0
    while cap.get(cv2.CAP_PROP_BUFFERSIZE) > 1 and frames_to_drop < 3:
        ret_drop, _ = cap.read()
        if not ret_drop:
            break
        frames_to_drop += 1
    
    if frames_to_drop > 0:
        print(f"Dropped {frames_to_drop} frames to prevent buffer buildup")

    # Clear buffer periodically to prevent accumulation
    if frame_count % 30 == 0:
        # Force buffer clear by setting buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Increment the frame count
    frame_count += 1

    # Optimize processing by reducing complex calculations frequency
    processing_counter += 1
    do_heavy_processing = (processing_counter % 3 == 0)  # Heavy processing every 3rd frame

    # Convert frame to grayscale
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = (0.5 * frame[:, :, 0] + 0.5 * frame[:, :, 2]).astype("uint8")
    #_, img = cv2.threshold(img, 254, 255, cv2.THRESH_TOZERO_INV)
    #img=(img-np.min(img))/(np.max(img)-np.min(img))
    #img=(255*img).astype("uint8")
    # Apply binary threshold
    k = np.median(img) / np.std(img)
    D_Th = np.average(img) + k * np.median(img)
    D_Th=min(D_Th,245)
    D_Th = max(D_Th, 180)

    _, B2 = cv2.threshold(img, D_Th, 255, cv2.THRESH_BINARY)  # Lowered threshold for better detection
    #print(D_Th)

    # Apply dilation
    #B3 = cv2.dilate(B2, np.ones((2, 2), np.uint8), iterations=2)  # Same dilation for noise reduction

    # Store the sum of white pixels for frequency analysis
    #freq.append(np.sum(B2))

    # **STEP 3: Skip all processing until we have 60 frames**
    if frame_count < calc_length:
        # Update the buffer but skip FFT and image processing
        for h in range(HE):
            for w in range(WI):
                b2 = B2[h * Box_size:h * Box_size + Box_size, w * Box_size:w * Box_size + Box_size]
                fourier_buff[:, h, w] = np.roll(fourier_buff[:, h, w], -1)
                fourier_buff[0, h, w] = np.sum(b2)

        continue  # Skip to the next frame until we have 60 frames

    # Once we have accumulated 60 frames, continue with regular processing
    if Mode == 1:
        # Loop through grid cells
        for h in range(HE):
            for w in range(WI):
                # Get current block in the grid
                b2 = B2[h * Box_size:h * Box_size + Box_size, w * Box_size:w * Box_size + Box_size]
                #b3 = B3[h * Box_size:h * Box_size + Box_size, w * Box_size:w * Box_size + Box_size]

                # Shift and update the Fourier buffer using np.roll
                fourier_buff[:, h, w] = np.roll(fourier_buff[:, h, w], -1)
                fourier_buff[0, h, w] = np.sum(b2)

                # Subtract the mean and apply FFT
                func = fourier_buff[:, h, w] - np.mean(fourier_buff[:, h, w])
                fftx = np.abs(np.fft.rfft(func))[3:]  # Start from 3 to skip low frequencies
                f_axis = np.linspace(0, fps / 2, len(fftx) + 3)[3:]

                # Find peaks in the FFT
                peaks, _ = find_peaks(fftx, height=np.max(fftx) * 0.5)  # Reduce peak sensitivity

                # Determine the dominant frequency (if any)
                if len(peaks) > 0 and fftx[peaks[0]] > 4 * np.median(fftx):
                    T = f_axis[peaks[0]]
                else:
                    T = 0

                # Draw rectangles around each grid cell
                cv2.rectangle(frame, (w * Box_size, h * Box_size), (w * Box_size + Box_size, h * Box_size + Box_size),
                              (0, 0, 255), 2)

                # If frequency matches LED frequency
                if np.abs(T - Led_freq) < freq_th:
                    freq_counter[h, w] += 1
                    if freq_counter[h, w] > freq_counter_th:
                        # contours, _ = cv2.findContours(b3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # if contours:
                        # c = max(contours, key=cv2.contourArea)
                        # top_point = (np.min(c[:, :, 0]) + w * Box_size, np.min(c[:, :, 1]) + h * Box_size)
                        # bottom_point = (np.max(c[:, :, 0]) + w * Box_size, np.max(c[:, :, 1]) + h * Box_size)
                        # bbox = (top_point[0], top_point[1], bottom_point[0] - top_point[0], bottom_point[1] - top_point[1])

                        # Initialize the tracker
                        # tracker.init(frame, bbox)
                        top_point = (w * Box_size-MAG, h * Box_size-MAG)
                        bottom_point = ((w + 1) * Box_size+MAG, (h + 1) * Box_size+MAG)
                        Mode = 2
                        freq_counter[h, w] = 0

                        # BREAK OUT OF BOTH LOOPS ONCE TRACKING STARTS
                        break  # Break out of inner loop
            if Mode == 2:
                break  # Break out of outer loop
    elif Mode == 2:

        TW = 1300
        TH = 1300
        BW = 0
        BH = 0
        start_h = top_point[1]
        start_w = top_point[0]
        for sh in range(sHE):

            for sw in range(sWI):

                b2 = B2[start_h + sh * sBox_size:start_h + sh * sBox_size + sBox_size,
                     start_w + sw * sBox_size:start_w + sw * sBox_size + sBox_size]
                #b3 = B3[start_h + sh * sBox_size:start_h + sh * sBox_size + sBox_size,
                #     start_w + sw * sBox_size:start_w + sw * sBox_size + sBox_size]
                img2 = frame[start_h + sh * sBox_size:start_h + sh * sBox_size + sBox_size,
                       start_w + sw * sBox_size:start_w + sw * sBox_size + sBox_size]
                z = shift(sfourier_buff[:, sw, sh], 1)
                z[0] = np.sum(b2)
                sfourier_buff[:, sw, sh] = z

                # Only do heavy FFT processing every 3rd frame
                if do_heavy_processing:
                    func = sfourier_buff[:, sw, sh] - np.average(sfourier_buff[:, sw, sh])
                    fftx = np.abs(np.fft.rfft(sfourier_buff[:, sw, sh]))[1:]
                    f_axis = np.linspace(0, fps / 2, int(0.5 * len(sfourier_buff[:, sw, sh])) + 1)[1:]
                    peaks, _ = find_peaks(fftx, height=np.max(fftx))
                else:
                    # Use previous results for non-processing frames
                    if 'prev_sT' not in locals():
                        prev_sT = 0
                    sT = prev_sT
                    
                cv2.rectangle(frame, (start_w + sw * sBox_size, start_h + sh * sBox_size),
                              (start_w + sw * sBox_size + sBox_size, start_h + sh * sBox_size + sBox_size), (0, 0, 255),2)

                # Only calculate frequency when doing heavy processing
                if do_heavy_processing:
                    # if fourier_thresh<fourier_TH:
                    if len(peaks) > 0 and fftx[peaks[0]] > 4 * np.median(fftx):
                        sT = f_axis[peaks][0]
                        prev_sT = sT  # Store for next non-processing frame
                    else:
                        sT = 0
                        prev_sT = 0

                if np.abs(sT - Led_freq) < freq_th:  # fftx[int(calc_length/fps)]>fourier_thresh:
                    stop_point = (sw * sBox_size + start_w, sh * sBox_size + start_h)
                    sbottom_point = ((sw + 1) * sBox_size + start_w, (sh + 1) * sBox_size + start_h)

                    TW = min(TW, stop_point[0])
                    TH = min(TH, stop_point[1])
                    BW = max(BW, sbottom_point[0])
                    BH = max(BH, sbottom_point[1])

                    #cv2.rectangle(frame, sbottom_point, stop_point, (0, 0, 255), 2)

                    cv2.rectangle(frame, (TW, TH), (BW, BH), (0, 255, 0), 2)
                    scounter = scounter + 1

                    if (scounter > 10) and (stop_point[0] + sBox_size < frameWidth) and (
                            stop_point[1] + sBox_size < frameHeight):
                        Mode = 3
                        bbox = (stop_point[0], stop_point[1], sbottom_point[0] - stop_point[0],
                                sbottom_point[1] - stop_point[1])
                        tracker.init(frame, bbox)
                        scounter = 0
                if (Mode == 3) or (Mode == 1):
                    break

        m2counter = m2counter + 1
        cv2.putText(frame, str(m2counter), (start_w + sw * sBox_size, start_h + sh * sBox_size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if (m2counter) > 1.5 * calc_length:
            Mode = 1
            m2counter = 0

        # if (Mode==3) or (Mode==1):
        #    break


    elif Mode == 3:
        # Tracking mode
        print(bbox)
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Count non-zero pixels inside the current bounding box
            roi_binary = B2[y:y + h, x:x + w]
            non_zero_pixel_count = np.sum(roi_binary)

            # Update pixel count buffer (shift and update)
            pixel_count_buff = np.roll(pixel_count_buff, -1)
            pixel_count_buff[-1] = non_zero_pixel_count
            tcounter = tcounter + 1
            if tcounter > calc_length:

                # Perform FFT on the pixel count buffer to detect object frequency
                pixel_count_fft = np.abs(np.fft.rfft(pixel_count_buff - np.mean(pixel_count_buff)))[1:]
                f_axis = np.linspace(0, fps / 2, len(pixel_count_fft) + 1)[1:]

                # Find peaks in the FFT to detect the dominant frequency
                peaks, _ = find_peaks(pixel_count_fft, height=np.max(pixel_count_fft))

                if len(peaks) > 0 and pixel_count_fft[peaks[0]] > 1.5 * np.median(pixel_count_fft):
                    detected_freq = f_axis[peaks[0]]
                else:
                    detected_freq = 0

                # # Update frequency history
                # freq_history.append(detected_freq)
                # if len(freq_history) > max_history_length:
                #     freq_history.pop(0)

                # mean_freq = np.mean(freq_history)
                # std_freq = np.std(freq_history)

                # Check if detected frequency is outside historical range
                if abs(detected_freq - Led_freq) > freq_th:
                    #print(
                    #    f"Frequency {detected_freq:.2f} Hz deviates from mean {mean_freq:.2f} Hz with std {std_freq:.2f}. Switching back to Mode 1.")
                    Mode = 1
                    # tracker = cv2.TrackerMIL_create()  # Reinitialize the tracker if necessary
                    freq_history = []
                    tcounter = 0
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Freq: {detected_freq:.2f} Hz', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

            # Display the tracking information
                #cv2.putText(frame, f'Pixels > 0: {int(non_zero_pixel_count)}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #        (255, 255, 255), 2)

            # cv2.putText(frame, f'Mean Freq: {mean_freq:.2f} Hz', (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            Mode = 1
            # tracker = cv2.TrackerMIL_create()

    # Show the processed binary image (B3) or frame
    # Optimize display - reduce resolution for better performance
    C2 = cv2.cvtColor(B2, cv2.COLOR_GRAY2RGB)
    FRAMES = np.concatenate((frame, C2), axis=1)
    # Reduced resolution for better performance
    prv = cv2.resize(FRAMES, (960, 360))  # Smaller than original 1280x960
    cv2.imshow("preview", prv)

    # Handle keypress with reduced wait time for better responsiveness
    k = cv2.waitKey(1)  # Reduced from 30 to 1 ms
    if k == 27 or k == ord('q'):  # ESC or 'q' to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
