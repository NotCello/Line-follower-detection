# Lane Detection & Steering Angle Estimation

## 🎥 Demo Video

![Alt Text for your GIF](/video_test.gif)

Watch the lane detection algorithm in action:

[![Watch the video](https://img.youtube.com/vi/4gZvpw_GFn4/maxresdefault.jpg)](https://youtu.be/4gZvpw_GFn4)

*(Click the image above to watch the video on YouTube)*

This project is a Python and OpenCV implementation of a lane detection algorithm for autonomous navigation. The primary goal is to process a dashcam video feed, identify lane lines, calculate the center of the lane, and estimate the steering angle required to keep the vehicle centered.

This is a fundamental algorithm for developing Advanced Driver-Assistance Systems (ADAS) and Autonomous Guided Vehicles (AGV), applicable to road, agricultural, and industrial environments.

## 🎯 Key Features

* **Real-Time Lane Detection:** Identifies road edges and lane lines from the video feed.
* **Lane Center Estimation:** Applies a robust averaging method (`np.polyfit`) to find the most likely position of both lines, even if they are dashed.
* **Error Calculation:** Calculates the error (in pixels) between the vehicle's center (red line) and the lane's center (blue line).
* **Steering Angle Estimation:** Implements a simple **Proportional (P) Controller** to convert the pixel error into a steering angle.
* **Debug Visualization:** Provides a clear visual overlay showing the detected lines (green), vehicle center, lane center, and real-time error/steering values.

## 🛠️ How It Works: The Processing Pipeline

The algorithm processes every single frame from the video through the following computer vision pipeline:

1.  **Pre-processing:** The frame is converted to grayscale, and a **Gaussian Blur** filter is applied to reduce noise and prepare the image for edge detection.
2.  **Edge Detection:** A **Canny** edge detector is used to identify gradients (edges) in the image.
3.  **Region of Interest (ROI):** A trapezoidal mask is applied to isolate only the road region, ignoring trees, the sky, and other irrelevant objects.
4.  **Line Detection:** The **Probabilistic Hough Line Transform** (`cv2.HoughLinesP`) is run on the masked image to find all straight-line segments.
5.  **Filtering & Fitting:**
    * Segments are divided into "left line" and "right line" based on their slope (negative for the left, positive for the right).
    * `np.polyfit` is used to find the best-fit line (1st-degree polynomial) for all left segments and all right segments. This provides a robust and stable average of the lines.
6.  **Calculation & Control:**
    * The `x` position of both lines is calculated at a fixed `y` coordinate (the bottom of the frame).
    * The **lane center** is calculated as the midpoint between the two lines.
    * The **error** is calculated as the difference between the lane center and the vehicle center.
    * The **steering angle** is calculated by multiplying the error by a proportional gain (Kp).
7.  **Visualization:** All calculated information (lines, centers, angle) is drawn onto an overlay image, which is then blended with the original frame.

## 🚀 Technologies Used

* **Python 3**
* **OpenCV-Python** (For all Computer Vision operations)
* **NumPy** (For matrix calculations and `polyfit`)

## ⚙️ How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/NotCello/Line-follower-detection.git](https://github.com/NotCello/Line-follower-detection.git)
    cd Line-follower-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate (Linux/macOS)
    source .venv/bin/activate
    
    # Activate (Windows)
    .\.venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    (It is recommended to create a `requirements.txt` file with `pip freeze > requirements.txt` containing `opencv-python` and `numpy`)
    ```bash
    pip install opencv-python numpy
    ```

4.  **Run the script:**
    (Ensure you have a video in the `TestVideo/` folder or modify the `video_path` in the script)
    ```bash
    python line_detector.py
    ```

## 📈 Future Improvements

* **Handle Complex Curves:** Replace the 1st-degree `np.polyfit` with a **2nd-degree polynomial (parabola)** to calculate the road's curvature and better handle sharp turns.
* **Temporal Smoothing:** Implement a **Kalman Filter** or a simple moving average on the line parameters to reduce "jitter" between frames.
* **Increase Robustness:** Add HSV color filtering to better isolate yellow and white lines in difficult lighting conditions (shadows, rain).
* **ROS/Gazebo Integration:** Publish the calculated steering angle to a simulator (like Gazebo or CoppeliaSim) to control a virtual robotic vehicle.
