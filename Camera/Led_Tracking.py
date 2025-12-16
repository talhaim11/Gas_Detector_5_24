import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.signal import find_peaks

# ===== הגדרות ראשוניות =====
VIDEO_PATH = 'rtsp://fgcam:admin@169.254.4.243:8554/0/unicast'
OUTPUT_PATH = "track_80m_double.mp4"
LED_FREQ = 2
FREQ_TOL = 0.1
FPS = 25
CALC_LENGTH = 60
BOX_SIZE = 200
MARGIN = 0
PIXEL_HISTORY = FPS * 2  # היסטוריה לכל פיקסל
# נתוני מצלמה
sensor_width=3296
sensor_height=2480
pixel_size=1.12e-6 # meter
focal_length=46e-3  #meter
# ===== פתיחת וידאו =====
cap = cv2.VideoCapture(VIDEO_PATH)

# הגדרת timeout וbuffer size לחיבור יציב יותר
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second timeout
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time

if not cap.isOpened():
    raise RuntimeError("לא ניתן להתחבר למצלמה")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# בדיקה שהרזולוציה תקינה
if frame_width == 0 or frame_height == 0:
    raise RuntimeError("לא ניתן לקבל רזולוציה תקינה מהמצלמה")

pixel_size_eff_x=pixel_size*sensor_width/frame_width
pixel_size_eff_y=pixel_size*sensor_height/frame_height

IFOV_x=2*np.degrees(np.arctan(pixel_size_eff_x/(2*focal_length)))
IFOV_y=2*np.degrees(np.arctan(pixel_size_eff_y/(2*focal_length)))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (frame_width, frame_height))

print("מנסה לקרוא פריים ראשון...")
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("לא ניתן לקרוא פריים ראשון מהוידאו - בדוק חיבור למצלמה")

print(f"✅ חיבור מצלמה הצליח! רזולוציה: {frame_width}x{frame_height}")

# ===== הכנות מבנים =====
WI, HE = frame_width // BOX_SIZE, frame_height // BOX_SIZE
fourier_buff = np.zeros((CALC_LENGTH, HE, WI))
freq_counter = np.zeros((HE, WI))
pixel_count_buff = np.zeros(CALC_LENGTH)

tracker = cv2.TrackerMIL_create()

# ===== משתנים גלובליים =====
mode = 1
roi_mode1 = None  # יוגדר במוד 1
flicker_buff = None
scounter = tcounter = m2counter = 0
top_point, bottom_point = None, None
pixel_buffer = None
frame_idx = 0
# משתנים למעבר למוד 3
flicker_stable_count = 0
STABLE_FRAMES_THRESHOLD = 5  # מספר פריימים שהקונטור חייב להישאר כדי לעבור למוד 3
bbox_mode3 = None  # המלבן שמועבר למוד 3



# ===== פונקציות עזר =====
def get_fft_frequency(signal, fps):
    """ מחזיר תדר דומיננטי באות נתון """
    signal = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(signal))[1:]
    f_axis = np.linspace(0, fps / 2, len(fft_vals) + 1)[1:]
    if len(fft_vals) == 0:
        return 0
    peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals))
    if len(peaks) > 0 and fft_vals[peaks[0]] >8 * np.median(fft_vals):
        return f_axis[peaks][0]
    return 0


def process_box(frame, gray, B2, B3, h, w):
    """ מעבד בלוק גדול (Mode 1) ובודק תדר """
    global mode, top_point, bottom_point

    b2 = B2[h*BOX_SIZE:(h+1)*BOX_SIZE, w*BOX_SIZE:(w+1)*BOX_SIZE]

    # עדכון באפר
    z = shift(fourier_buff[:, h, w], 1)
    z[0] = np.sum(b2)
    fourier_buff[:, h, w] = z

    # FFT
    freq = get_fft_frequency(fourier_buff[:, h, w], FPS)

    # בדיקת קרבה לתדר LED
    if abs(freq - LED_FREQ) < FREQ_TOL:
        freq_counter[h, w] += 1
        if freq_counter[h, w] > 5:  # סף הופעל
            top_point = (w*BOX_SIZE, h*BOX_SIZE)
            bottom_point = ((w+1)*BOX_SIZE, (h+1)*BOX_SIZE)
            cv2.rectangle(frame, top_point, bottom_point, (0, 255, 0), 2)
            mode = 2
            return True
    else:
        freq_counter[h, w] = 0
    return False


def process_flicker_pixels(frame, roi, flicker_buff, frame_count, buffer_len=60, thresh=100):
    global flicker_stable_count, mode, bbox_mode3

    x1, y1, x2, y2 = roi
    roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    h, w = roi_gray.shape

    if flicker_buff is None or flicker_buff.shape[1:] != (h, w):
        flicker_buff = np.zeros((buffer_len, h, w), dtype=np.uint8)
        frame_count = 0
        flicker_stable_count = 0

    flicker_buff = np.roll(flicker_buff, -1, axis=0)
    flicker_buff[-1] = roi_gray
    frame_count += 1

    mask = np.zeros((h, w), dtype=np.uint8)
    if frame_count > buffer_len:
        diff = flicker_buff.max(axis=0) - flicker_buff.min(axis=0)
        mask[diff > thresh] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            cv2.rectangle(frame, (x1+x, y1+y), (x1+x+w_box, y1+y+h_box), (0, 0, 255), 2)

            flicker_stable_count += 1

            # אם הקונטור יציב מספיק פריימים, עוברים למוד 3
            if flicker_stable_count >= STABLE_FRAMES_THRESHOLD:
                bbox_mode3 = (x1+x, y1+y, w_box, h_box)
                tracker.init(frame, bbox_mode3)
                mode = 3
        else:
            flicker_stable_count = 0  # הקונטור נעלם, מאפסים
            bbox_mode3 = None
            mode = 1  # חזרה למוד 1

    return mask, flicker_buff, frame_count, bbox_mode3


# ===== לולאת עיבוד =====
start=0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = frame[:, :, 2]
    _, B2 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    B3 = cv2.dilate(B2, np.ones((10, 10), np.uint8), iterations=100)

    if mode == 1:
        for h in range(HE):
            for w in range(WI):
                cv2.rectangle(frame, (w*BOX_SIZE, h*BOX_SIZE),
                              (w*BOX_SIZE+BOX_SIZE, h*BOX_SIZE+BOX_SIZE),
                              (255, 255, 255), 2)
                if process_box(frame, gray, B2, B3, h, w):
                    break
            if mode == 2:
                break


    elif mode == 2:
        if top_point is not None and bottom_point is not None:
            roi_mode1 = (top_point[0]-MARGIN, top_point[1]-MARGIN, bottom_point[0]+MARGIN, bottom_point[1]+MARGIN)
            cv2.rectangle(frame, (top_point[0]-MARGIN, top_point[1]-MARGIN),(bottom_point[0]+MARGIN, bottom_point[1]+MARGIN),(255, 0, 255), 2)
            mask, flicker_buff, frame_idx,bbox_mode3 = process_flicker_pixels(frame, roi_mode1, flicker_buff, frame_idx)
            if bbox_mode3:
                (x0, y0, w0, h0) = [int(v) for v in bbox_mode3]
                if start==0:
                    center_x0 = x0 + w0 // 2
                    center_y0 = y0 + h0 // 2
                    start=1
    elif mode == 3:
        if bbox_mode3 is not None:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2

                #center_text = f"Center: ({center_x}, {center_y})"

                x_alignment=center_x-center_x0
                y_alignment = center_y - center_y0

                x_alignment_deg=np.round((center_x-center_x0)*IFOV_x,2)
                y_alignment_deg = np.round((center_y - center_y0)*IFOV_y,2)

                if (np.abs(x_alignment_deg)>0.5) or (np.abs(y_alignment_deg)>0.5):
                    color=(0,0,255)
                else:
                    color=(255,0,0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                center_text2= f"({x_alignment_deg}, {y_alignment_deg})"
                center_text = f"({x_alignment}, {y_alignment})"
                center_text3= f"({center_x}, {center_y})"
                text_position = (x, y + h + 20)
                text2_position = (x, y + h + 40)
                cv2.putText(frame, center_text3, text_position, cv2.FONT_HERSHEY_SIMPLEX,0.6, color, 2)
                #cv2.putText(frame, center_text2, text2_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                roi_binary = B2[y:y+h, x:x+w]
                non_zero_pixel_count = np.sum(roi_binary)
                pixel_count_buff = np.roll(pixel_count_buff, -1)
                pixel_count_buff[-1] = non_zero_pixel_count

                tcounter += 1
                if tcounter > CALC_LENGTH:
                    detected_freq = get_fft_frequency(pixel_count_buff, FPS)
                    if abs(detected_freq - LED_FREQ) > FREQ_TOL:
                        mode = 1
                        tcounter = 0
            else:
                start=0
                mode = 1
                flicker_stable_count = 0
                bbox_mode3 = None

    cv2.putText(frame, f"Mode: {mode}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    out.write(frame)
    cv2.imshow("preview", cv2.resize(frame, (1280, 960)))
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
