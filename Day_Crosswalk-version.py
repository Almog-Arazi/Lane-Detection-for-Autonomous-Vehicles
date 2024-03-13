import numpy as np
import cv2


previous_left_lines = []
previous_right_lines = []
buffer_size = 5  
smoothing_factor = 0.3  


previous_route_state = None
route_change_detected = False
route_change_start_time = None
route_change_duration = 1.7  

def process(image, vertices):
    global previous_left_lines, previous_right_lines

    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold_low = 50
    threshold_high = 170
    image_canny = cv2.Canny(image_g, threshold_low, threshold_high)
    cropped_image = region_of_interest(image_canny, vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_len = 35
    max_line_gap = 80
    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_image = draw_the_lines(image, lines, vertices) 
    return line_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines, vertices):
    global previous_left_lines, previous_right_lines, previous_route_state, route_change_detected, route_change_start_time

    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_lines.append((slope, y1 - slope * x1))
            else:
                right_lines.append((slope, y1 - slope * x1))

    overlay = img.copy()
    alpha = 0.4  
    lines_drawn = 0

    
    if left_lines:
        left_avg = np.average(left_lines, axis=0)
        previous_left_lines.append(left_avg)
        if len(previous_left_lines) > buffer_size:
            previous_left_lines = previous_left_lines[-buffer_size:]
        left_avg = np.mean(previous_left_lines, axis=0)

        x1, y1, x2, y2 = calculate_coordinates(left_avg, vertices)
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 15)
        lines_drawn += 1

    if right_lines:
        right_avg = np.average(right_lines, axis=0)
        previous_right_lines.append(right_avg)
        if len(previous_right_lines) > buffer_size:
            previous_right_lines = previous_right_lines[-buffer_size:]
        right_avg = np.mean(previous_right_lines, axis=0)

        x1, y1, x2, y2 = calculate_coordinates(right_avg, vertices)
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 15)
        lines_drawn += 1

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    
    if lines_drawn == 1:
        if previous_route_state is None:
            previous_route_state = "straight"
        elif previous_route_state == "straight":
            previous_route_state = "changing"
            route_change_detected = True
            route_change_start_time = cv2.getTickCount() / cv2.getTickFrequency()

    if route_change_detected:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - route_change_start_time < route_change_duration:
            font = cv2.FONT_HERSHEY_DUPLEX
            text = "Changing Route  >>>"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) / 2
            textY = (img.shape[0] + textsize[1]) / 2
            cv2.putText(img, text, (int(textX), int(textY)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            route_change_detected = False
            previous_route_state = None
            previous_left_lines.clear()
            previous_right_lines.clear()

    return img

def calculate_coordinates(line_parameters, vertices):
    slope, intercept = line_parameters
    y1 = max(vertices[0][0][1], vertices[0][1][1])
    y2 = min(vertices[0][2][1], vertices[0][3][1])
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2



cap = cv2.VideoCapture('crosswalkVideo.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('crosswalk-result.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20, size)

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    run_lanes = True
    
    
    roi_lanes_coords = np.array([[(400, frame.shape[0]), (1530, frame.shape[0]), (1150, 650), (910, 650)]],
                                    dtype=np.int32)  
    
    roi_crosswalks_coord = np.array([[(490, frame.shape[0]), (1460, frame.shape[0]), (1100, 650), (940, 650)]],
                                    dtype=np.int32)  

    
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_crosswalks_coord], (255, 255, 255))
    roi = cv2.bitwise_and(frame, mask)
    
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 10), np.uint8)
    dilated_mask = cv2.dilate(edges, kernel, iterations=10)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=15)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=15)

    cropped_image = region_of_interest(dilated_mask, roi_crosswalks_coord)

    
    contours, _ = cv2.findContours(cropped_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
   
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000 :  
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 3.9:
                cv2.rectangle(frame, (x, y ), (x + w + 100, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Crosswalk Detected !", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            run_lanes = False

    if run_lanes:
        process(frame, roi_lanes_coords)
    
    result.write(frame)
    
cap.release()
result.release()
cv2.destroyAllWindows()
