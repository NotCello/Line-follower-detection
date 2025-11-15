import numpy as np
import cv2

# --- CONFIGURATION PARAMETERS (Tune everything from here) ---

# Canny Parameters
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Gaussian Blur Parameters
BLUR_KERNEL_SIZE = (5, 5)

# Hough Transform Parameters
HOUGH_RHO = 2
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 50

# Slope Filter Parameters
SLOPE_THRESHOLD_LOW = -0.3 # For the left line (negative)
SLOPE_THRESHOLD_HIGH = 0.3 # For the right line (positive)

# ROI Polygon Parameters (Values as fractions of H and W)
ROI_TOP_Y_RATIO = 0.5   # Top Y of the trapezoid (0.5 = half screen)
ROI_TOP_X_LEFT_RATIO = 0.45 # Top left X
ROI_TOP_X_RIGHT_RATIO = 0.55 # Top right X

# Control Parameters
KP_GAIN = 0.1 # Proportional (P) controller gain

# Visualization Colors
COLOR_LEFT_LANE = (0, 255, 0)
COLOR_RIGHT_LANE = (0, 255, 0)
COLOR_CENTER_LANE = (255, 0, 0) # Blue
COLOR_CAR_CENTER = (0, 0, 255) # Red
COLOR_TEXT = (255, 255, 255)

# --- END OF CONFIGURATION ---


def get_line_parameters(segments):
    """
    Finds the parameters (m, b) for the line x = m*y + b that best
    approximates all input segments.
    """
    x_coords = []
    y_coords = []
    
    if segments is None or len(segments) == 0:
        return None

    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    if not y_coords:
        return None

    try:
        params = np.polyfit(y_coords, x_coords, 1) # Degree = 1 (a line)
        return params # Will return [m, b]
    except np.linalg.LinAlgError:
        print("Polyfit Error")
        return None


def process_frame(frame):
    """
    Processes a single video frame to find lanes, 
    calculate the error, and return an image with an overlay.
    """
    
    # === STEP 1: PRE-PROCESSING ===
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
    canny = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # === STEP 2: REGION OF INTEREST (ROI) ===
    height, width = frame.shape[:2]
    
    # Define the polygon vertices using ratios
    y_top_roi = int(height * ROI_TOP_Y_RATIO)
    x_top_left_roi = int(width * ROI_TOP_X_LEFT_RATIO)
    x_top_right_roi = int(width * ROI_TOP_X_RIGHT_RATIO)
    
    polygon = np.array([
        (0, height),              # Bottom-left
        (width, height),          # Bottom-right
        (x_top_right_roi, y_top_roi), # Top-right
        (x_top_left_roi, y_top_roi)   # Top-left
    ], dtype=np.int32)
    
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(canny, mask)

    # === STEP 3: HOUGH TRANSFORM ===
    lines = cv2.HoughLinesP(masked_image, 
                            rho=HOUGH_RHO, 
                            theta=HOUGH_THETA, 
                            threshold=HOUGH_THRESHOLD, 
                            minLineLength=HOUGH_MIN_LINE_LENGTH, 
                            maxLineGap=HOUGH_MAX_LINE_GAP)

    # === STEP 4: AVERAGING AND FILTERING LOGIC ===
    left_segment = []
    right_segment = [] # Corrected from right_Segment
    
    if lines is not None:
        for line in lines: 
            x1,y1,x2,y2 = line[0]
            if (x2-x1) == 0: continue
            
            slope = (y2-y1) / (x2-x1)

            if slope < SLOPE_THRESHOLD_LOW:
                left_segment.append(line)
            elif slope > SLOPE_THRESHOLD_HIGH:
                right_segment.append(line)

    left_line_params = get_line_parameters(left_segment)
    right_line_params = get_line_parameters(right_segment)

    # === STEP 5: STEERING ERROR CALCULATION ===
    
    # REAL car center (not for debug)
    centro_auto = width // 2 
    centro_corsia = centro_auto 
    
    if left_line_params is not None and right_line_params is not None:
        y_riferimento = height 
        
        m_sinistro, b_sinistro = left_line_params
        m_destro, b_destro = right_line_params
        
        x_sinistro = int((m_sinistro * y_riferimento) + b_sinistro)
        x_destro = int((m_destro * y_riferimento) + b_destro)
        
        centro_corsia = (x_sinistro + x_destro) // 2

    errore_pixel = centro_corsia - centro_auto
    angolo_sterzata = KP_GAIN * errore_pixel

    # === STEP 6: VISUALIZATION ===
    
    overlay_image = np.zeros_like(frame)
    y_bottom = height
    # Use the same Y as the ROI polygon for drawing
    y_top_draw = y_top_roi 

    if left_line_params is not None:
        m_left, b_left = left_line_params
        x1_left = int((m_left * y_bottom) + b_left)
        x2_left = int((m_left * y_top_draw) + b_left)
        cv2.line(overlay_image, (x1_left, y_bottom), (x2_left, y_top_draw), COLOR_LEFT_LANE, 10)

    if right_line_params is not None:
        m_right, b_right = right_line_params
        x1_right = int((m_right * y_bottom) + b_right)
        x2_right = int((m_right * y_top_draw) + b_right)
        cv2.line(overlay_image, (x1_right, y_bottom), (x2_right, y_top_draw), COLOR_RIGHT_LANE, 10)

    # Draw car center line
    cv2.line(overlay_image, (centro_auto, y_bottom), (centro_auto, y_top_draw), COLOR_CAR_CENTER, 3)
    
    # Draw lane center line (if found)
    if centro_corsia != centro_auto:
        cv2.line(overlay_image, (centro_corsia, y_bottom), (centro_corsia, y_top_draw), COLOR_CENTER_LANE, 3)
    
    # Combine the overlay with the original frame
    risultato_finale = cv2.addWeighted(frame, 0.8, overlay_image, 1.0, 0)
    
    # Add text
    testo = f"Steering Angle: {angolo_sterzata:.2f} | Pixel Error: {errore_pixel}"
    cv2.putText(risultato_finale, testo, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2, cv2.LINE_AA)
    
    # --- RETURN THE RESULTS ---
    # Your main expects these 3 values
    
    # (For future debugging, you could also return 'canny' and 'masked_image')
    # return risultato_finale, angolo_sterzata, errore_pixel, canny, masked_image
    
    return risultato_finale, angolo_sterzata, errore_pixel