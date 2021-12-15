from motion_detection.pose_estimation import *
from functions import *

BODY_PARTS, POSE_PAIRS, net = init_pose_estimation_net()


def movement_detection(min_area, fg_mask, frame, pose_estimation_flag):
    movement = 0
    body_parts = []

    # Get the contours of the detected foreground mask
    contours, h = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the area of the contours and the corresponding bounding rectangle
    rectangles = []
    for c in contours:
        area = cv2.contourArea(c)
        rectangles.append((area, cv2.boundingRect(c)))

    # Draw the rectangles on the frame with the coordinates found
    for area, rect in rectangles:
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h

        # Check if the contour area is greater than the predefined minimum area [significant movement]
        if area > min_area:
            movement = 1

            # Draw the rectangle of the movement region
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Find body part where the movement occurred through pose estimation
            if pose_estimation_flag:
                points = pose_estimation(frame, 200, 200, BODY_PARTS, POSE_PAIRS, net)
                body_parts = detect_body_part_movement(points, (x1, y1, x2, y2))

                # write in the frame if movement occurred and the corresponding location
                cv2.putText(frame, "Motion Occurrence:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                i = 50
                for part in body_parts:
                    cv2.putText(frame, str(part), (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
                    i += 20

    return movement, body_parts, frame
