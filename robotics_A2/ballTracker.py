
import cv2
import numpy as np
import time
def preprocess_cam_hsv(frame):
    # Preprocess the frames and convert them to HSV color spaces
    # The lower and higher thresholds are all adjusted through debugging
    hsv_lower = np.array([0, 96, 215])  # Default lower HSV value
    hsv_higher = np.array([18, 255, 255])  # Default upper HSV value
    # 1. Resize the frame
    current_frame = cv2.resize(frame, (640, 480))
    # 2. Color Extraction
    # Convert to HSV and apply mask
    color_hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    # 3. Blurring and edge detection and obtain the mask
    blurred = cv2.GaussianBlur(color_hsv_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, hsv_lower, hsv_higher)
    return current_frame, mask


def trackBall_ApproxPolyDP(frame):
    #time_pre_approx = time.perf_counter()
    # 1. resize the frame
    current_frame = cv2.resize(frame, (640, 480))
    # 2. obtain the gray formatting frame
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # 3. blur the frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 4. get the edges through canny
    edges = cv2.Canny(blurred, 50, 150)
    #time_pre_approx_end = time.perf_counter()
    #print(f"time approx_preprocessing:{(time_pre_approx_end - time_pre_approx) * 1000}")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    # 5. filter the contours using polygon approximating and circularity calculation.
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if 6 < len(approx) < 11:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            # Calculate the circularity and determine if it's a circle.
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if area > 100 and circularity > 0.4:
                circles.append(contour)
                cv2.drawContours(current_frame, [approx], -1, (0, 255, 0), 2)
    #time_approx_end = time.perf_counter()
    #print(f"time approx_approximating:{(time_approx_end - time_pre_approx_end) * 1000}")
    #print(f"time approx_processing:{(time_approx_end - time_pre_approx) * 1000}")
    return circles, current_frame, edges


def trackBall_Color_extraction(frame):
    #time_pre_start = time.perf_counter()
    # S1. Obtain the mask through HSV preprocessing
    current_frame, mask = preprocess_cam_hsv(frame)
    #time_pre_end = time.perf_counter()
    #print(f"time color_preprocessing:{(time_pre_end - time_pre_start) * 1000}")

    # Morphological opening, closing
    kernel = np.ones((5, 5), np.uint8)
    # S2. let the frame pass through the closing morphology and find contours
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    # Calculate circularity and get the maximum areas as the ball.
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > 100 and circularity > 0.4:
            circles.append(contour)
            #cv2.drawContours(current_frame, [contour], -1, (0, 255, 0), 2)
    #time_color_end = time.perf_counter()
    #print(f"time color_extraction:{(time_color_end - time_pre_end) * 1000}")
    #print(f"time color_processing:{(time_color_end - time_pre_start) * 1000}")
    return circles, current_frame, closing

def analyze_result(circles, current_frame, method):
    if circles:
        # Calculate the positions and draw the rectangle of the circle.
        max_contour = max(circles, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        points_left = (x - 5, y - 5)
        points_right = (x + w + 5, y + h + 5)
        # draw the rectangle
        if method == "ApproxPolyDP":
            cv2.rectangle(current_frame, points_left, points_right, (0, 255, 0), 2)
        center = ((points_left[0] + points_right[0]) >> 2, (points_left[1] + points_right[1]) >> 2)
        area = cv2.contourArea(max_contour)
        print(f"[BALL {method}] x: {center[0]} y: {center[1]} area: {area}")

def trackBall(frame):
    circles, current_frame, closing = trackBall_Color_extraction(frame)
    analyze_result(circles, current_frame, "ColorExtract")
    circles, current_frame, edges = trackBall_ApproxPolyDP(frame)
    analyze_result(circles, current_frame, "ApproxPolyDP")
    # Display images
    cv2.imshow('Original', current_frame)
    cv2.imshow('closing', closing)
    cv2.imshow('edges', edges)
