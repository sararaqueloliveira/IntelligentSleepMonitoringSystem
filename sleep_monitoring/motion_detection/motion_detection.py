from motion_detection.pose_estimation import *
from functions import *

BODY_PARTS, POSE_PAIRS, net = init_pose_estimation_net()


def movement_detection(min_area, image, fgmask, second):
    # Count all the non zero pixels within the mask
    contours, h = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get collection of (area, rect) from mask contours
    rectangles = []
    for c in contours:
        area = cv2.contourArea(c)
        rectangles.append((area, cv2.boundingRect(c)))

    # draw larger rectangles, with padding, on the output frame
    for area, rect in rectangles:
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h

        if area > min_area:
            # render the rectangle of the movement region
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # find body part where the movement occurred through pose estimation
            if second:
                points = pose_estimation(image, 200, 200, BODY_PARTS, POSE_PAIRS, net)
                body_parts = detect_body_part_movement(points, (x1, y1, x2, y2))

                # write in the frame if movement occurred and the corresponding location
                cv2.putText(image, "Motion Occurrence:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                i = 50
                for part in body_parts:
                    cv2.putText(image, str(part), (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
                    i += 20

    return image
