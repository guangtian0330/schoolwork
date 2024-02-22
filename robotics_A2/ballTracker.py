
import cv2
import numpy as np
def preprocess_cam_hsv(frame):
    hsv_lower = np.array([0, 58, 232])  # Default lower HSV value
    hsv_higher = np.array([18, 206, 255])  # Default upper HSV value
    # Resize the frame
    current_frame = cv2.resize(frame, (640, 480))
    # 1. Color Extraction
    # Convert to HSV and apply mask
    color_hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    # Blurring and edge detection
    blurred = cv2.GaussianBlur(color_hsv_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, hsv_lower, hsv_higher)
    return current_frame, mask


def trackBall_ApproxPolyDP(frame):
    current_frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if 6 < len(approx) < 11:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if area > 100 and circularity > 0.4:
                circles.append(contour)
                cv2.drawContours(current_frame, [approx], -1, (0, 255, 0), 2)
    return circles, current_frame, edges


def trackBall_Color_extraction(frame):
    current_frame, mask = preprocess_cam_hsv(frame)
    # Morphological opening, closing
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > 100 and circularity > 0.4:
            circles.append(contour)
    return circles, current_frame, closing

def analyze_result(circles, current_frame, method):
    if circles:
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
