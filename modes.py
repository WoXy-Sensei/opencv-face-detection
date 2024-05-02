import cv2 as cv

"""
    frame: The frame in which the face is to be blurred
    bbox: The bounding box of the face to be blurred
"""


def blur_face(frame, bbox):
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]
    face = cv.GaussianBlur(face, (99, 99), 30)
    frame[y:y+h, x:x+w] = face
    return frame


"""
    frame: The frame in which the face is to be highlighted
    bbox: The bounding box of the face to be highlighted
"""


def rectangle_face(frame, bbox):
    x, y, w, h = bbox
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


"""
    frame: The frame in which the face is to be highlighted
    bbox: The bounding box of the face to be highlighted
"""


def cross_face(frame, bbox):
    x, y, w, h = bbox
    cv.line(frame, (x, y+round(h / 2)),(x+w, y+round(h / 2)), (255, 0, 255), 2)
    cv.line(frame, (x+round(w / 2), y),(x+round(w / 2), y+h), (255, 0, 255), 2)
    return frame


"""
    frame: The frame in which the face is to be highlighted
    bbox: The bounding box of the face to be highlighted
"""


def no_mode(frame, bbox):
    return frame
