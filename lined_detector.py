import cv2
import numpy as np

def get_line_parameters(segments):
    """
    Finds the parameters (m, b) for the line x = m*y + b that best
    approximates all input segments.
    """
    # Lists to hold ALL points
    x_coords = []
    y_coords = []
    
    # If the segment list is empty, do nothing
    if segments is None or len(segments) == 0:
        return None

    # Unpack all points
    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        x_coords.append(x1)
        x_coords.append(x2)
        y_coords.append(y1)
        y_coords.append(y2)

    # If we have no points (strange, but better to check)
    if not y_coords:
        return None

    # Calculate the line x = m*y + b
    # We use (y_coords, x_coords) because y is our independent variable
    try:
        params = np.polyfit(y_coords, x_coords, 1) # Degree = 1 (a line)
        return params # Will return [m, b]
    except np.linalg.LinAlgError:
        # Calculation error, probably perfectly vertical lines
        print("Polyfit Error")
        return None

# --- STEP 0: SETUP AND LOADING ---

# Load your video
video_path = 'TestVideo/test_lane_detector.mp4' # Make sure the path and name are correct
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # --- START OF PROCESSING LOGIC (to be applied to 'frame') ---

    # === STEP 1: PRE-PROCESSING (Prepare the Image) ===
    # The goal is to create a binary image (black/white) with only the useful edges.
    
    # 1. Convert 'frame' to Grayscale (cv2.cvtColor)
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY )
    
    # 2. Apply a Gaussian Filter (Blur) to remove noise (cv2.GaussianBlur)
    #    blur = cv2.GaussianBlur(gray, (5, 5), 0) # (5, 5) is the kernel size, must be odd
    blur=cv2.GaussianBlur(gray_image,(5,5),0.2 )
    
    # 3. Edge Detection with Canny (cv2.Canny)
    #    canny = cv2.Canny(blur, 50, 150) # You will need to adjust the 50 and 150 thresholds
    canny=cv2.Canny(blur,50,150)
    # 4. (Optional) Show Canny output for debugging
    cv2.imshow('Canny Output', canny)


    # === STEP 2: REGION OF INTEREST (ROI) ===
    # The goal is to "crop" only the portion of the road we are interested in.
    
    # 1. Define a polygon (trapezoid) for the region of interest
    #    (requires the frame's height and width)
    height, width = frame.shape[:2]
    #    # The vertices (x, y) must be adjusted by watching the video!
    polygon = np.array([
        (0, height),              # Bottom-left corner
        (width, height),          # Bottom-right corner
        (width*0.55, height*0.5), # Top-right point of the lane (approx)
        (width*0.45, height*0.5)  # Top-left point of the lane (approx)
    ], dtype=np.int32)
    
    # 2. Create a black mask as large as the Canny image (np.zeros_like)
    mask = np.zeros_like(canny)
    
    # 3. "Fill" the polygon on the mask with white color (cv2.fillPoly)
    cv2.fillPoly(mask, [polygon], 255)
    
    # 4. Apply the mask to the Canny image (cv2.bitwise_and)
    masked_image = cv2.bitwise_and(canny, mask)
    
    # 5. (Optional) Show the masked image for debugging
    cv2.imshow('ROI Output', masked_image)


    # === STEP 3: LINE DETECTION (HOUGH TRANSFORM) ===
    # The goal is to find all straight-line segments in the ROI image.
    
    lines = cv2.HoughLinesP(masked_image, rho=2, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=50)
    # (You will need to adjust 'threshold', 'minLineLength', and 'maxLineGap' to find the right lines)


    # === STEP 4: AVERAGING AND FILTERING LOGIC ===
    
    left_segment = []
    right_Segment = []
    
    if lines is not None:
        # Start the loop: iterate over EVERY line found
        for line in lines: 
            x1,y1,x2,y2 = line[0]

            if (x2-x1) == 0:
                continue # Skip this line if it's vertical
 
            # THIS IS NOW INSIDE THE 'for' LOOP
            slope = (y2-y1) / (x2-x1)

            # THIS IS ALSO INSIDE THE 'for' LOOP
            # AND WITH THE CORRECT SLOPE LOGIC
            if slope < -0.3: # NEGATIVE slope = LEFT Line
                left_segment.append(line)
            elif slope > 0.3: # POSITIVE slope = RIGHT Line
                right_Segment.append(line)
        
        # The 'for' loop ends here

    # LOOK AT THIS PRINT IN THE TERMINAL!
    print(f"Left segments: {len(left_segment)}, Right segments: {len(right_Segment)}")
    
    left_line_params = get_line_parameters(left_segment)
    right_line_params = get_line_parameters(right_Segment)

    # === STEP 5: STEERING ERROR CALCULATION ===
    # The goal is to calculate a number that tells us "how much to steer".
    
    # 1. Calculate the 'lane_center'
    #    (average between the x position of the left and right lines, calculated at a fixed y, e.g., at the bottom of the screen)
    #    lane_center = (x_left + x_right) / 2
    # === STEP 5: STEERING ERROR CALCULATION ===
    
    # Initialize the values. We assume the center is
    # the car's center, so if we don't find lines, the error is 0.
    centro_auto = (width // 2) - 80 # Use // for an integer
    centro_corsia = centro_auto 
    
    # We check if we found BOTH lines.
    # If one is missing, we can't calculate the center and we use the default values.
    if left_line_params is not None and right_line_params is not None:
        
        # We choose a fixed 'y' to calculate the 'x' coordinates.
        # The bottom of the screen is a good spot.
        y_riferimento = height 
        
        # Extract the m and b parameters from the LEFT line
        m_sinistro, b_sinistro = left_line_params
        
        # Extract the m and b parameters from the RIGHT line
        m_destro, b_destro = right_line_params
        
        # Calculate the x coordinates using our equation x = m*y + b
        # We convert to integers because pixels cannot be decimals
        x_sinistro = int((m_sinistro * y_riferimento) + b_sinistro)
        x_destro = int((m_destro * y_riferimento) + b_destro)
        
        # Now we can calculate the lane center
        centro_corsia = (x_sinistro + x_destro) // 2
        
        # (Debug: you can draw these points to see if they are correct)
        cv2.circle(frame, (x_sinistro, y_riferimento), 10, (0, 255, 0), -1) # Green
        cv2.circle(frame, (x_destro, y_riferimento), 10, (0, 0, 255), -1)   # Red
        cv2.circle(frame, (centro_corsia, y_riferimento), 10, (255, 0, 0), -1) # Blue

    # Now we calculate the error
    errore_pixel = centro_corsia - centro_auto
    
    # (Optional) Calculate the steering angle (P-Control)
    Kp = 0.1 # Proportional gain (to be tuned)
    angolo_sterzata = Kp * errore_pixel

    # (Debug: print the error)
    print(f"Pixel Error: {errore_pixel}, Angle: {angolo_sterzata:.2f}")
    

   # === STEP 6: VISUALIZATION ===
    
    # 1. Create an empty (black) "overlay" image
    overlay_image = np.zeros_like(frame)
    
    # 2. Draw the two lines (left and right) on the overlay
    
    # We define the upper Y for our drawing (must match your ROI's Y)
    y_top_roi = int(height * 0.4) 
    y_bottom = height

    # Draw the LEFT line (if it was found)
    if left_line_params is not None:
        m_left, b_left = left_line_params
        # Calculate the points (x1, y1) and (x2, y2)
        x1_left = int((m_left * y_bottom) + b_left)
        x2_left = int((m_left * y_top_roi) + b_left)
        # Draw a thick green line
        cv2.line(overlay_image, (x1_left, y_bottom), (x2_left, y_top_roi), (0, 255, 0), 10)

    # Draw the RIGHT line (if it was found)
    if right_line_params is not None:
        m_right, b_right = right_line_params
        # Calculate the points (x1, y1) and (x2, y2)
        x1_right = int((m_right * y_bottom) + b_right)
        x2_right = int((m_right * y_top_roi) + b_right)
        # Draw a thick green line
        cv2.line(overlay_image, (x1_right, y_bottom), (x2_right, y_top_roi), (0, 255, 0), 10)

    # 3. Draw the lane center (blue) and the car center (red)
    #    (We only draw in the visible part of the ROI)
    
    # 3. Draw the lane center (blue) and the car center (red)
    
    # Car Center (red) - is always fixed
    cv2.line(overlay_image, (centro_auto, y_bottom), (centro_auto, y_top_roi), (0, 0, 255), 3) # RED
    
    # Lane Center (blue) - CONDITIONAL
    # Draw the blue line ONLY if we found a lane (i.e., if it's different from the car center)
    if centro_corsia != centro_auto:
        cv2.line(overlay_image, (centro_corsia, y_bottom), (centro_corsia, y_top_roi), (255, 0, 0), 3) # BLUE
    
    # 5. Combine the original 'frame' image with the 'overlay'
    #    This creates the "transparency" effect
    risultato_finale = cv2.addWeighted(frame, 0.8, overlay_image, 1.0, 0)
    
    
    # 4. Write the 'steering_angle' value on the FINAL image
    #    (We do this *after* addWeighted so it isn't transparent)
    testo_angolo = f"Steering Angle: {angolo_sterzata:.2f} Pixel Error: {errore_pixel}"
    cv2.putText(risultato_finale, testo_angolo, 
                (50, 50), # Position (x, y) from the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, # Font size
                (255, 255, 255), # Color (white)
                2, # Thickness
                cv2.LINE_AA)
    
    # 6. Show the 'final_result'
    cv2.imshow('Result', risultato_finale)
    
    # --- END OF PROCESSING LOGIC ---

    # !!! IMPORTANT: DELETE OR COMMENT OUT the old 'cv2.imshow' line!!!
    # cv2.imshow('Video Originale', frame)
    
    # For now, just show the original frame (DELETE/COMMENT THIS LINE WHEN SHOWING THE RESULT)
    #cv2.imshow('Video Originale', masked_image) 


    # --- END OF PROCESSING LOGIC ---

    # Press 'q' to exit (waits 1 millisecond)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- FINAL STEP: RELEASE AND CLOSE ---
cap.release()
cv2.destroyAllWindows()