import cv2
import numpy as np

MIN_POS_THRESHOLD = 5
FORWARD_ARROW = 0
LEFT_ARROW = 1
RIGHT_ARROW = 2
arrow_template_images = ['marker_forward.png', 'marker_left.png', 'marker_right.png']


def get_hsv_mean(frame, poi_start, poi_end):
    # Extract the ROI and calculate the average HSV values
    roi = frame[poi_start[1]:poi_end[1], poi_start[0]:poi_end[0]]
    mean_hsv_per_row = np.average(roi, axis=0)
    mean_hsv = np.average(mean_hsv_per_row, axis=0)
    return mean_hsv.astype(int)


class BallTracker:
    def __init__(self):
        self.ball_selected = False

        self.pos_start = (0, 0)
        self.pos_end = (0, 0)
        self.selected_diff = (0, 0)
        self.hsv_lower = np.array([0, 0, 0])  # Default lower HSV value
        self.hsv_higher = np.array([180, 255, 255])  # Default upper HSV value
        self.current_frame = None
        self.cap = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pos_start = (x, y)
            self.ball_selected = False

        elif event == cv2.EVENT_LBUTTONUP:
            self.pos_end = (x, y)
            self.selected_diff = (abs(self.pos_start[0] - self.pos_end[0]),
                                  abs(self.pos_start[1] - self.pos_end[1]))
            if (self.selected_diff[0] < MIN_POS_THRESHOLD or
                    self.selected_diff[1] < MIN_POS_THRESHOLD):
                self.ball_selected = False
            else:
                self.ball_selected = True
                # Obtain the lower and higher HSVs
                if self.current_frame is not None:
                    hsv_values = get_hsv_mean(
                        cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV),
                        self.pos_start, self.pos_end)
                    self.hsv_lower = np.maximum(hsv_values - 20, 0)
                    self.hsv_higher = np.minimum(hsv_values + 20, [180, 255, 255])

    def trackBall(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize the frame
        self.current_frame = cv2.resize(frame, (640, 480))

        if self.ball_selected:
            # 1. Color Extraction
            # Convert to HSV and apply mask
            color_hsv_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(color_hsv_frame, self.hsv_lower, self.hsv_higher)
            # Blurring and edge detection
            blurred = cv2.GaussianBlur(mask, (5, 5), 0)

            # Morphological opening, closing, erosion, and dilation
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                points_left = (x - 5, y - 5)
                points_right = (x + w + 5, y + h + 5)
                # draw the rectangle
                cv2.rectangle(self.current_frame, points_left, points_right, (0, 255, 0), 2)
                center = ((points_left[0] + points_right[0]) >> 2, (points_left[1] + points_right[1]) >> 2)
                area = cv2.contourArea(max_contour)
                print(f"[BALL estimated] x: {center[0]} y: {center[1]} area: {area}")
            # Display images
            cv2.imshow('Original', self.current_frame)
            cv2.imshow('closing', closing)
        else:
            cv2.imshow('Original', self.current_frame)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        cv2.namedWindow('Original')
        cv2.setMouseCallback('Original', self.mouse_callback)

        while True:
            self.trackBall()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


def get_canny(frame):
    # low_threshold = cv2.getTrackbarPos('Low Threshold', 'Canny Parameters')
    # high_threshold = cv2.getTrackbarPos('High Threshold', 'Canny Parameters')
    edges = cv2.Canny(frame, 50, 150)
    return edges


# Preprocess the arrow template pictures.
def preprocess_arrow_template(image_path):
    gray_frame = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (640, 480))
    template_contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = None
    if template_contours:
        template_contour = max(template_contours, key=cv2.contourArea)
    return gray_frame, template_contour


"""
def trackArrows(frame, forward_descriptor, left_descriptor, right_descriptor):
    resized_frame = cv2.resize(frame, (640, 480))
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)
    # edges = get_canny(gray_frame)
    
    low_threshold_H = cv2.getTrackbarPos('low_threshold_H', 'Binary Parameters')
    high_threshold_H = cv2.getTrackbarPos('high_threshold_H', 'Binary Parameters')
    low_threshold_S = cv2.getTrackbarPos('low_threshold_S', 'Binary Parameters')
    high_threshold_S = cv2.getTrackbarPos('high_threshold_S', 'Binary Parameters')
    low_threshold_V = cv2.getTrackbarPos('low_threshold_V', 'Binary Parameters')
    high_threshold_V = cv2.getTrackbarPos('high_threshold_V', 'Binary Parameters')
    low_white = np.array([low_threshold_H, low_threshold_S, low_threshold_V])
    high_white = np.array([high_threshold_H, high_threshold_S, high_threshold_V])
    
    low_white = np.array([95, 15, 210])
    high_white = np.array([186, 51, 255])
    white_mask = cv2.inRange(hsv_frame, low_white, high_white)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:
            continue
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if 6 <= len(approx) <= 9:
            contour_list.append(contour)
    if contour_list:
        max_approx = max(contour_list, key=cv2.contourArea)
        cv2.drawContours(resized_frame, [max_approx], 0, (0, 255, 0), 3)
        print(f"contour area = {cv2.contourArea(max_approx)}")
    #rect = cv2.minAreaRect(approx)
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    # arrow_type = None
    # print(f"forward = {len()}, left = {len()}, right = {len()}")
    cv2.imshow('original', resized_frame)
    cv2.imshow('white_mask', white_mask)

    return
    """


def trackArrows(frame, forward_contour, left_contour, right_contour):
    #match_threshold = cv2.getTrackbarPos('match_threshold', 'Match threshold')
    #match_threshold = match_threshold / 10
    match_threshold = 0.4
    print(f"threshold = {match_threshold}")
    resized_frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
    edges = get_canny(gray_frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    arrow_type = None
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
            # print(f"min_match = {min_match}")
            if max_area > area:
                continue
            else:
                max_contour = contour
                max_area = area
                best_approx = approx
            if len(approx) == 7:
                arrow_type = 'forward'
            else:
                arrow_type = 'turns'

    if max_contour is not None:
        angle = 90
        rect = cv2.minAreaRect(best_approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if arrow_type == 'turns':
            _, _, angle = rect
            if angle > 60:
                arrow_type = 'Right'
            elif angle < 30:
                arrow_type = 'Left'
            else:
                arrow_type = 'None'
        if arrow_type == 'Left' or arrow_type == 'Right' or arrow_type == 'forward':
            cv2.drawContours(gray_frame, [box], 0, (0, 0, 255), 3)
            print(f"angle = {angle}, max_area = {max_area}")
            print(f"suspecting arrow type is : {arrow_type}, number of edges = {len(best_approx)}")
            cv2.drawContours(gray_frame, [max_contour], -1, (0, 255, 0), 3)
    cv2.imshow('Frame', gray_frame)
    return


def openCam():
    gray_left, left_contour = preprocess_arrow_template('marker_left.png')
    gray_right, right_contour = preprocess_arrow_template('marker_right.png')
    gray_forward, forward_contour = preprocess_arrow_template('marker_forward.png')
    print(f"length of forward:{len(forward_contour)}, left:{len(left_contour)}, right:{len(right_contour)}")
    epsilon = 0.05 * cv2.arcLength(forward_contour, True)
    approx = cv2.approxPolyDP(forward_contour, epsilon, True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        trackArrows(frame, forward_contour, left_contour, right_contour)
        #cv2.drawContours(gray_forward, [approx], 0, (0, 0, 255), 3)
        #cv2.imshow('forward', gray_forward)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Run the webcam pipeline
    # ball_tracker = BallTracker()
    # ball_tracker.run()

    # cv2.namedWindow('Canny Parameters')
    # cv2.createTrackbar('Low Threshold', 'Canny Parameters', 0, 255, get_canny)
    # cv2.createTrackbar('High Threshold', 'Canny Parameters', 0, 255, get_canny)

    # cv2.namedWindow('Match threshold')
    # cv2.createTrackbar('match_threshold', 'Match threshold', 0, 10, trackArrows)

    openCam()


if __name__ == "__main__":
    main()