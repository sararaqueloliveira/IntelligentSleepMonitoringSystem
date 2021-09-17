import cv2


class Movement:
    def movement_detection(min_area, image, fgmask):
        # Count all the non zero pixels within the mask
        contours, h = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get collection of (area, rect) from mask contours
        rectangles = []
        for c in contours:
            area = cv2.contourArea(c)
            rectangles.append((area, cv2.boundingRect(c)))

        # draw larger rectangles, with padding, on the output frame
        for area, rect in sorted(rectangles, reverse=True):
            if area > min_area:
                # add padding and render rectangle
                x, y, w, h = rect
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (128, 255, 128), 3)

        return image
