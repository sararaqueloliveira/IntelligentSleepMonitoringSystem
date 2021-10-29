import imutils

from functions import *


def init_pose_estimation_net():
    proto = '../models/pose_estimation/pose_deploy.prototxt'
    model = '../models/pose_estimation/pose_iter_440000.caffemodel'

    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    net = cv2.dnn.readNetFromCaffe(proto, model)

    return BODY_PARTS, POSE_PAIRS, net


def connect_pose_pairs(frame, points, BODY_PARTS, POSE_PAIRS):
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 2)
            cv2.ellipse(frame, points[idFrom], (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def pose_estimation(frame, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, net):
    # frame = imutils.resize(image, width=400)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    points = []
    i = 0
    for part in BODY_PARTS:
        # Slice heatmap of corresponding body part
        heatMap = out[0, i, :, :]

        # Find all the local maximums
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y), part) if conf > 0.1 else None)

        i += 1

    return points


# Classification of estimated poses:
# - lying on the left side
# - lying on the right side
# - lying up
# - lying down
def pose_classification(frame, pair):
    right_side = 0
    left_side = 0

    left_side += 1 if 4 in pair else 0  # LShoulder
    left_side += 1 if 6 in pair else 0  # LElbow
    left_side += 1 if 7 in pair else 0  # LWrist
    left_side += 1 if 15 in pair else 0  # LEye
    left_side += 1 if 17 in pair else 0  # LEar
    left_side += 1 if 11 in pair else 0  # LHip
    left_side += 1 if 12 in pair else 0  # LKnee
    left_side += 1 if 13 in pair else 0  # LAnkle

    right_side += 1 if 2 in pair else 0  # RShoulder
    right_side += 1 if 3 in pair else 0  # RElbow
    right_side += 1 if 4 in pair else 0  # RWrist
    right_side += 1 if 8 in pair else 0  # RHip
    right_side += 1 if 9 in pair else 0  # RKnee
    right_side += 1 if 10 in pair else 0  # RAnkle
    right_side += 1 if 14 in pair else 0  # REye
    right_side += 1 if 16 in pair else 0  # REar

    if right_side < left_side:
        text = "Lying on the right side"
    elif right_side > left_side:
        text = "Lying on the left side"
    else:
        text = "lying up"

    frame = draw_text(frame, text, pos=(10, 40))
    return frame


def detect_body_part_movement(points, rect):
    body_parts = []
    head = ["Nose", "REye", "LEye", "REar", "LEar"]
    upper_body = ["RShoulder", "LShoulder", "LHip", "RHip", "Neck"]
    arms = ["RWrist", "RElbow", "LElbow", "LWrist"]
    down_body = ["LKnee", "LAnkle", "RKnee", "RAnkle"]

    # detect if any of the body parts found is inside the motion rectangle calculated
    for point in points:
        if point is not None:
            px, py, part = point
            x1, y1, x2, y2 = rect

            if x1 < px < x2 and y1 < py < y2:
                # location of the movement
                if part in head:
                    body_parts.append("Head")
                elif part in upper_body:
                    body_parts.append("Upper Body")
                elif part in arms:
                    body_parts.append("Arms")
                elif part in down_body:
                    body_parts.append("Down Body")
                else:
                    body_parts.append("Background")

    # delete duplicates
    body_parts = list(dict.fromkeys(body_parts))
    print(body_parts)
    return body_parts
