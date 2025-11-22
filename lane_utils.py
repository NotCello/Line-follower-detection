import numpy as np
import cv2

# Global settings
CANNY_THRESH = (50, 150)
BLUR_KERNEL = (5, 5)

# Hough Transform
HOUGH_PARAMS = {
    'rho': 2,
    'theta': np.pi/180,
    'threshold': 50,
    'minLineLength': 20,
    'maxLineGap': 50
}

# Slope filtering (tune these for different camera angles)
SLOPE_LEFT = -0.3
SLOPE_RIGHT = 0.3

# ROI - Region of Interest (fractions of screen size)
ROI_Y_TOP = 0.5
ROI_X_LEFT = 0.45
ROI_X_RIGHT = 0.55

# P-Controller gain
KP = 0.1

def get_line_params(segments):
    """ Fits a line (m, b) through a set of segments using polyfit. """
    if segments is None or len(segments) == 0:
        return None

    x_coords = []
    y_coords = []

    for line in segments:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    if not y_coords:
        return None

    try:
        # Fit x = my + b (using y as independent var for stability with vertical lines)
        return np.polyfit(y_coords, x_coords, 1)
    except np.linalg.LinAlgError:
        return None

def process_frame(frame):
    height, width = frame.shape[:2]

    # Pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_THRESH[0], CANNY_THRESH[1])

    # Create ROI mask
    mask = np.zeros_like(edges)
    
    # Define trapezoid vertices
    y_top = int(height * ROI_Y_TOP)
    poly_pts = np.array([
        (0, height),
        (width, height),
        (int(width * ROI_X_RIGHT), y_top),
        (int(width * ROI_X_LEFT), y_top)
    ], dtype=np.int32)

    cv2.fillPoly(mask, [poly_pts], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines
    lines = cv2.HoughLinesP(masked_edges, **HOUGH_PARAMS)

    # Filter lines by slope
    left_segs = []
    right_segs = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue # avoid division by zero
            
            slope = (y2 - y1) / (x2 - x1)

            if slope < SLOPE_LEFT:
                left_segs.append(line)
            elif slope > SLOPE_RIGHT:
                right_segs.append(line)

    # Calculate average lines
    left_params = get_line_params(left_segs)
    right_params = get_line_params(right_segs)

    # Calculate steering error
    car_center = width // 2
    lane_center = car_center # default if lines lost

    if left_params is not None and right_params is not None:
        y_ref = height
        
        # Calculate x positions at the bottom of the screen
        x_left = int(left_params[0] * y_ref + left_params[1])
        x_right = int(right_params[0] * y_ref + right_params[1])
        
        lane_center = (x_left + x_right) // 2

    pixel_error = lane_center - car_center
    steering_angle = KP * pixel_error

    # Visualization
    overlay = np.zeros_like(frame)
    
    # Helper to draw lines
    def draw_lane_line(params, color):
        if params is not None:
            m, b = params
            x_bottom = int(m * height + b)
            x_top = int(m * y_top + b)
            cv2.line(overlay, (x_bottom, height), (x_top, y_top), color, 10)

    draw_lane_line(left_params, (0, 255, 0))
    draw_lane_line(right_params, (0, 255, 0))

    # Draw centers
    cv2.line(overlay, (car_center, height), (car_center, y_top), (0, 0, 255), 3) # Red (Car)
    if lane_center != car_center:
        cv2.line(overlay, (lane_center, height), (lane_center, y_top), (255, 0, 0), 3) # Blue (Lane)

    # Blend
    result = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)

    # Info text
    info = f"Angle: {steering_angle:.2f} | Error: {pixel_error}"
    cv2.putText(result, info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result, steering_angle, pixel_error