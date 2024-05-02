import cv2 as cv
import mediapipe as mp
from modes import *
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="cross",help="The mode to use for face detection")
args = parser.parse_args()

# Camera
cameraId = 0  # 0 for built-in camera, 1 for external camera
cap = cv.VideoCapture(cameraId)

# Face detection
mpFace = mp.solutions.face_detection

# Modes
modes = {
    "blur": blur_face,
    "rectangle": rectangle_face,
    "cross": cross_face,
    "none": no_mode
}

def run(mode: str = "none"):
    with mpFace.FaceDetection(min_detection_confidence=0.5) as faceDetection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = faceDetection.process(frameRGB)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape  # Image height, image width for scaling
                    frame = modes[mode](frame, (int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)))

            cv.imshow(f"Frame {args.mode}", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break


if __name__ == "__main__":
    run(args.mode)
