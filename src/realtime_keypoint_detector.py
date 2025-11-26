# src/realtime_keypoint_detector_v2.py
import os, time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model

mp_pose = mp.solutions.pose

YOLO_MODEL = "yolov8n.pt"
CLASSIFIER_PATH = "models/fall_keypoint_model.h5"
SAVE_DIR = "detected_falls"
os.makedirs(SAVE_DIR, exist_ok=True)

yolo = YOLO(YOLO_MODEL)
model = load_model(CLASSIFIER_PATH)

REQUIRED_STABLE = 5  # temporal smoothing

def extract_keypoints(image, pose):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints += [lm.x, lm.y, lm.z, lm.visibility]
    return np.array(keypoints)

def main():
    cap = cv2.VideoCapture(0)
    last_state = "no_fall"
    stable_counter = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, verbose=False)
            person_crop = None
            bbox = None

            # Find largest person
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2-x1)*(y2-y1)
                        if bbox is None or area > (bbox[2]-bbox[0])*(bbox[3]-bbox[1]):
                            bbox = (x1, y1, x2, y2)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                person_crop = frame[y1:y2, x1:x2]
                keypoints = extract_keypoints(person_crop, pose)

                if keypoints is not None:
                    pred = model.predict(keypoints.reshape(1,-1), verbose=0)[0][0]
                    state = "fall" if pred > 0.5 else "no_fall"

                    # temporal smoothing
                    if state == last_state:
                        stable_counter += 1
                    else:
                        stable_counter = 0
                        last_state = state

                    final_state = last_state if stable_counter >= REQUIRED_STABLE else "no_fall"
                    color = (0,0,255) if final_state=="fall" else (0,255,0)
                    label = f"{final_state.upper()} ({pred:.2f})"

                    # Draw bbox and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Save frame if fall detected
                    if final_state=="fall":
                        fname = f"fall_{int(time.time())}_{pred:.2f}.jpg"
                        cv2.imwrite(os.path.join(SAVE_DIR, fname), frame)
                        print("⚠ Fall detected, saved:", fname)

                    # Optional crop window
                    cv2.imshow("Person Crop", person_crop)

                else:
                    cv2.putText(frame, "Pose not detected", (20,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            else:
                cv2.putText(frame, "No person detected", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Fall Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key==27 or key==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
