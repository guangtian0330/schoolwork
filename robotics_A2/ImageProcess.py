import cv2
import numpy as np

INVALID_ARROW = 0
FORWARD_ARROW = 1
LEFT_ARROW = 2
RIGHT_ARROW = 3
arrow_template_images = ['marker_forward.png', 'marker_left.png', 'marker_right.png']

TRACKER_BALL = 0
TRACKER_ARROW = 1
tracker_type = TRACKER_BALL

# Preprocess the arrow template pictures.


def preprocess_arrow_template(image_path):
    gray_frame = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (640, 480))
    template_contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = None
    if template_contours:
        template_contour = max(template_contours, key=cv2.contourArea)
    return gray_frame, template_contour


gray_left, left_contour = preprocess_arrow_template('marker_left.png')
gray_right, right_contour = preprocess_arrow_template('marker_right.png')
gray_forward, forward_contour = preprocess_arrow_template('marker_forward.png')
result_type = []
invalid_score = 0


def preprocess_cam_hsv(frame):
    hsv_lower = np.array([0, 122, 118])  # Default lower HSV value
    hsv_higher = np.array([48, 219, 219])  # Default upper HSV value
    # Resize the frame
    current_frame = cv2.resize(frame, (640, 480))
    # 1. Color Extraction
    # Convert to HSV and apply mask
    color_hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    # Blurring and edge detection
    blurred = cv2.GaussianBlur(color_hsv_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, hsv_lower, hsv_higher)
    cv2.imshow("blurred", blurred)
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
    circles, current_frame, edges = trackBall_ApproxPolyDP(frame)
    analyze_result(circles, current_frame, "ColorExtract")
    circles, current_frame, closing = trackBall_ApproxPolyDP(frame)
    analyze_result(circles, current_frame, "ApproxPolyDP")
    # Display images
    cv2.imshow('Original', current_frame)
    cv2.imshow('edges', edges)


def get_canny(frame):
    # low_threshold = cv2.getTrackbarPos('Low Threshold', 'Canny Parameters')
    # high_threshold = cv2.getTrackbarPos('High Threshold', 'Canny Parameters')
    edges = cv2.Canny(frame, 50, 150)
    return edges


def trackArrows(frame):
    match_threshold = 0.4  # This threshold value is adjusted during debugging
    resized_frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
    edges = get_canny(gray_frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    arrow_type = INVALID_ARROW
    best_approx = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if area < 300:
            continue
        if len(approx) == 7 or len(approx) == 9:
            forward_match = cv2.matchShapes(forward_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            left_match = cv2.matchShapes(left_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            right_match = cv2.matchShapes(right_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            min_match = min(forward_match, left_match, right_match)
            if min_match > match_threshold:
                continue
            if max_area > area:
                continue
            else:
                max_contour = contour
                max_area = area
                best_approx = approx
            if len(approx) == 7:
                arrow_type = FORWARD_ARROW

    if max_contour is not None:
        angle = 90
        rect = cv2.minAreaRect(best_approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if arrow_type != FORWARD_ARROW:
            _, _, angle = rect
            if angle > 60:
                arrow_type = RIGHT_ARROW
            elif angle < 30:
                arrow_type = LEFT_ARROW
            else:
                arrow_type = INVALID_ARROW
        if arrow_type != INVALID_ARROW:
            cv2.drawContours(gray_frame, [box], 0, (0, 0, 255), 3)
            print(f"angle = {angle}, max_area = {max_area}")
            print(f"suspecting arrow type is : {arrow_type}, number of edges = {len(best_approx)}")
            cv2.drawContours(gray_frame, [max_contour], -1, (0, 255, 0), 3)
    return gray_frame, edges, arrow_type


def analyze_arrow_tracking_res(arrow_type):
    image = None
    global invalid_score
    if arrow_type == INVALID_ARROW:
        invalid_score += 1
    else:
        result_type.append(arrow_type)
    if invalid_score == 30:
        invalid_score = 0
        result_type.clear()
        cv2.destroyWindow("result")
    if len(result_type) == 10:
        mean_val = np.mean(result_type)
        print(f"mean value {mean_val} of arrow_type = {arrow_type}")
        invalid_score = 0
        result_type.clear()
        if 0 < mean_val <= 1.4:
            image = gray_forward
        elif 1.4 <= mean_val <= 2.4:
            image = gray_left
        elif mean_val >= 2.4:
            image = gray_right
    if image is not None:
        cv2.imshow("result", image)

def update_tracker_type(tracker):
    global tracker_type
    cv2.destroyAllWindows()
    tracker_type = tracker
def openCam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # low_threshold = cv2.getTrackbarPos('Low Threshold', 'Canny Parameters')
        if not tracker_type:
            trackBall(frame)
        else:
            gray_result, edges, arrow_type = trackArrows(frame)
            top_row = cv2.hconcat(([gray_result, edges]))
            cv2.imshow("debug", top_row)
            analyze_arrow_tracking_res(arrow_type)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    cv2.namedWindow('Ball Tracker 0 or Arrow Tracker 1')
    cv2.createTrackbar('Tracker', 'Ball Tracker 0 or Arrow Tracker 1', 0, 1, openCam)
    openCam()


if __name__ == "__main__":
    main()
