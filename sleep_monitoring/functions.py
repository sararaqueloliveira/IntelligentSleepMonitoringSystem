import cv2
import numpy as np


def draw_text(frame,
              text,
              font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 0, 0),
              text_color_bg=(255, 255, 255)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(frame, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(frame, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return frame


def draw_contours_blank(contours):
    blank = np.zeros((324, 614, 3), np.uint8)
    blank = cv2.drawContours(blank, contours, -1, (255, 255, 255), 1)
    return blank
