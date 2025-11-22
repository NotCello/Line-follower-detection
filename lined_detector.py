import cv2
import lane_utils

# Set to 0 for webcam, or path string for video file
VIDEO_SOURCE = 'TestVideo/test_lane_detector.mp4'
# VIDEO_SOURCE = 0 

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Failed to open video source: {VIDEO_SOURCE}")
    exit()

print("Running... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Stream ended or error reading frame.")
        break

    # Process frame
    try:
        result_img, angle, error = lane_utils.process_frame(frame)
        
        print(f"Err: {error}, Angle: {angle:.2f}")
        cv2.imshow('Lane Assist', result_img)
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        cv2.imshow('Lane Assist', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()