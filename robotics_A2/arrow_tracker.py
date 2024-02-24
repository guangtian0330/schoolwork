import time

import cv2
import numpy as np
import ballTracker

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
    # load all the 3 types of arrow pictures used for shape matching.
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


def get_canny(frame):
    edges = cv2.Canny(frame, 50, 150)
    return edges

# TrackArrows function used to track the arrow target and identify the diretion.
def trackArrows(frame):

    #time_start = time.perf_counter() # Record the starting time

    # STEP1: PREPROCESS IMAGES.
    # This threshold value is adjusted during debugging. This is the maximum values for shape matching
    match_threshold = 0.4
    resized_frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
    # Use gray formatting to get the edges of the camera, and then find contours.
    edges = get_canny(gray_frame)
    # Use the cv2.RETR_CCOMP to ensure the internal edges are observed as well.
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Record the contours with the max areas, the minimum match scores
    max_area = 0
    max_contour = None
    arrow_type = INVALID_ARROW
    best_approx = 0
    min_match_score = 100
    #time_preprocess = time.perf_counter()
    #print(f"time preprocessing:{(time_preprocess - time_start) * 1000}")

    # STEP2: FILTERING CONTOURS USING SHAPE MATCHING AND POLYGON APPROXIMATION.
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        # ignore the areas smaller than 300. This threshold value is adjusted through debugging
        if area < 300:
            continue
        # Only focus on the 7-edged and 9-edged polygons which are the forward arrow and turning arrows respectively.
        if len(approx) == 7 or len(approx) == 9:
            forward_match = cv2.matchShapes(forward_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            left_match = cv2.matchShapes(left_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            right_match = cv2.matchShapes(right_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            min_match = min(forward_match, left_match, right_match)
            # find the minimum match scores and compare it with the previous minimum one and the threshold.
            if min_match > match_threshold or min_match > min_match_score:
                continue
            # Meanwhile, find the maximum area. This 2 conditions combined are sometimes too strict and can cause problems.
            # So it's also feasible to skip one of these them.
            if max_area > area:
                continue
            else:
                # record all the optimal values.
                min_match_score = min_match
                max_contour = contour
                max_area = area
                best_approx = approx
            if len(approx) == 7:
                # the 7-edge polygon is technically the forward arrow.
                arrow_type = FORWARD_ARROW
    #arrow_identify_time_start = time.perf_counter()

    # STEP3: IDENTIFY THE ARROW DIRECTION BASED ON ANGLES OF MINIMUM AREA RECTANGLE
    if max_contour is not None:
        angle = 90
        rect = cv2.minAreaRect(best_approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # Calculate the minimum area rectangle and get the angle of it.
        # Decide the direction of the arrow base on the angle of the rectangle.
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
    # THe following is to record the time expenses.
    #arrow_identify_time_end = time.perf_counter()
    #print(f"time for arrow identifying:{(arrow_identify_time_end - arrow_identify_time_start) * 1000}")
    #print(f"time for arrow targeting:{(arrow_identify_time_end - time_preprocess) * 1000}")
    return gray_frame, edges, arrow_type

# The results may change quickly because sometimes it's affected by noises and mistakes.
# So a scoring system is added here to keep the results steady
def analyze_arrow_tracking_res(arrow_type):
    image = None
    global invalid_score
    if arrow_type == INVALID_ARROW:
        invalid_score += 1
    else:
        result_type.append(arrow_type)
    if invalid_score == 30:
        # If the invalid results hit 30 which means currently
        # there's no valid target detected, so clear all the previous records
        invalid_score = 0
        result_type.clear()
        cv2.destroyWindow("result")
    if len(result_type) == 10:
        # Calculate the average score of 10 results and keep the most frequent result.
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

def openCam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Start Ball tracking, this will not be commented for tracking balls.
        #ballTracker.trackBall(frame)

        # Start Arrow tracking.
        gray_result, edges, arrow_type = trackArrows(frame)
        top_row = cv2.hconcat(([gray_result, edges]))
        cv2.imshow("debug", top_row)
        analyze_arrow_tracking_res(arrow_type)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

openCam()

