import numpy as np
import cv2
from scipy.ndimage import shift
from scipy.signal import find_peaks

cap = cv2.VideoCapture(r"C:\Users\thaim\Videos\LED - color change\הקלטה - 50 מטר עם צבע לבן + HSV שינוי פרמטרים.mp4")

#cap=cv2.VideoCapture(r"C:\Users\user\Videos\vlc-record-2024-09-23-11h59m43s-Converting rtsp___fgcam_admin@169.254.118.8_8554_0_unicast-.mp4")

# PARAMS & DEFINITIONS
img_counter = 0
fps = 25
calc_length = 60
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




# Initialize tracker
tracker = cv2.TrackerCSRT_create()
while True:
    ret, frame = cap.read()
    #print(Mode)
    if not ret:
        print("End of video or error reading frame.")
        break

    # Increment the frame count
    frame_count += 1

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

                func = sfourier_buff[:, sw, sh] - np.average(sfourier_buff[:, sw, sh])
                fftx = np.abs(np.fft.rfft(sfourier_buff[:, sw, sh]))[1:]
                f_axis = np.linspace(0, fps / 2, int(0.5 * len(sfourier_buff[:, sw, sh])) + 1)[1:]
                peaks, _ = find_peaks(fftx, height=np.max(fftx))
                cv2.rectangle(frame, (start_w + sw * sBox_size, start_h + sh * sBox_size),
                              (start_w + sw * sBox_size + sBox_size, start_h + sh * sBox_size + sBox_size), (0, 0, 255),2)

                # if fourier_thresh<fourier_TH:
                if len(peaks) > 0 and fftx[peaks[0]] > 4 * np.median(fftx):
                    sT = f_axis[peaks][0]
                else:
                    sT = 0

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
    C2 = cv2.cvtColor(B2, cv2.COLOR_GRAY2RGB)
    FRAMES = np.concatenate((frame, C2), axis=1)
    prv = cv2.resize(FRAMES, (1280, 960))
    cv2.imshow("preview", prv)  # Or cv2.imshow("preview", frame) if you want the full frame

    # Handle keypress
    k = cv2.waitKey(30)
    if k == 27 or k == ord('q'):  # ESC or 'q' to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
