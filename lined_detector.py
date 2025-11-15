import cv2
import lane_utils  # Import your new utility file

# --- CONFIGURATION ---
# To use the webcam, change 'video_path' to 0
# To use a video, set the file path
video_path = 'TestVideo/test_lane_detector.mp4'
# video_path = 0 # Example for webcam

# --- STEP 0: SETUP AND LOADING ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video source: {video_path}")
    exit()

print("Processing started... Press 'q' to exit.")

while cap.isOpened():
    # 'ret' is a boolean (True if frame was read correctly)
    # 'frame' is the image
    ret, frame = cap.read()

    if not ret:
        if video_path != 0:
            print("End of video.")
            break
        else:
            print("Error reading webcam, exiting.")
            break

    # --- PROCESSING ---
    # All the complexity is hidden in this single function!
    try:
        # We pass the frame to our utility and receive the results
        img_risultato, angolo, errore = lane_utils.process_frame(frame)
        
        # Print data to the terminal
        print(f"Pixel Error: {errore}, Angle: {angolo:.2f}")

        # Show only the final image
        cv2.imshow('Result', img_risultato)

    except Exception as e:
        # Handles any processing errors (e.g., corrupted frames)
        print(f"Error during frame processing: {e}")
        # Show the original frame in case of error to avoid crashing
        cv2.imshow('Result', frame) 

    # --- EXIT CONTROL ---
    # Press 'q' to exit (waits 1 millisecond)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- FINAL STEP: RELEASE AND CLOSE ---
print("Closing...")
cap.release()
cv2.destroyAllWindows()