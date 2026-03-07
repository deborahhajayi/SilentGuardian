#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import joblib
import math
import time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ====================================================
# CONFIG
# ====================================================

SEQ_LEN = 8

THRESH_FALL = 0.5
THRESH_GESTURE = 0.6

CLASS_NAMES = ["like", "palm", "neutral"]

# ====================================================
# FALL MODEL
# ====================================================

print("Loading fall models...")

movenet = tf.lite.Interpreter(
    model_path="models/movenet_lightning.tflite"
)
movenet.allocate_tensors()

mn_in = movenet.get_input_details()
mn_out = movenet.get_output_details()

fall_model = load_model("models/fall_geom_lstm.keras")

fall_buf = deque(maxlen=SEQ_LEN)
height_buffer = deque(maxlen=300)

# ====================================================
# HAND MODEL
# ====================================================

print("Loading hand models...")

hand_model = load_model("models/hand_gesture_mlp.keras")
scaler = joblib.load("models/hand_scaler.pkl")

yolo_hand = YOLO("models/yolov8_hand_pose.pt")

# ====================================================
# FALL FEATURE ENGINEERING
# ====================================================

def angle(a,b):
    return np.arctan2(b[1]-a[1], b[0]-a[0])

def geom_from_kp(kp):

    NOSE = kp[0][:2]

    L_SHO, R_SHO = kp[5][:2], kp[6][:2]
    L_HIP, R_HIP = kp[11][:2], kp[12][:2]

    shoulders=(L_SHO+R_SHO)/2
    hips=(L_HIP+R_HIP)/2

    body_height=np.linalg.norm(NOSE-hips)

    if body_height>0:
        height_buffer.append(body_height)

    median_h=np.median(height_buffer) if height_buffer else 1

    rel_h=body_height/(median_h+1e-6)

    xs,ys=kp[:,0],kp[:,1]

    aspect=(ys.max()-ys.min()+1e-6)/(xs.max()-xs.min()+1e-6)

    return np.array([
        angle(L_SHO,R_SHO),
        angle(L_HIP,R_HIP),
        angle(shoulders,hips),
        body_height,
        aspect,
        shoulders[0],shoulders[1],
        hips[0],hips[1],
        NOSE[0],NOSE[1],
        np.linalg.norm(shoulders-hips),
        rel_h
    ],dtype=np.float32)

# ====================================================
# HAND FEATURE ENGINEERING (EXACT YOUR SCRIPT LOGIC)
# ====================================================

def normalize_hand(kp):
    kp = kp - kp[0]
    scale = np.linalg.norm(kp[9]) + 1e-6
    return kp / scale

def angle3(a,b,c):
    ba=a-b
    bc=c-b
    return np.arccos(
        np.clip(
            np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),
            -1,1
        )
    )

def extract_hand_features(kp_flat):

    kp=kp_flat.reshape(21,2)
    kp=normalize_hand(kp)

    features=[]

    fingers=[
        [0,1,2,3,4],
        [0,5,6,7,8],
        [0,9,10,11,12],
        [0,13,14,15,16],
        [0,17,18,19,20]
    ]

    for f in fingers:
        features.append(angle3(kp[f[0]],kp[f[1]],kp[f[2]]))
        features.append(angle3(kp[f[1]],kp[f[2]],kp[f[3]]))

    tips=[4,8,12,16,20]

    for t in tips:
        features.append(np.linalg.norm(kp[t]))

    for i in range(len(tips)):
        for j in range(i+1,len(tips)):
            features.append(
                np.linalg.norm(kp[tips[i]]-kp[tips[j]])
            )

    features.append(math.atan2(kp[9][1],kp[9][0]))

    return np.array(features,dtype=np.float32)

# ====================================================
# MOVENET FALL DETECTOR
# ====================================================

def run_movenet(frame):

    h,w,_=frame.shape

    size=192
    scale=size/max(h,w)

    nh,nw=int(h*scale),int(w*scale)

    img=cv2.resize(frame,(nw,nh))

    pad_top=(size-nh)//2
    pad_left=(size-nw)//2

    canvas=np.zeros((size,size,3),dtype=np.float32)
    canvas[pad_top:pad_top+nh,pad_left:pad_left+nw]=img

    canvas=cv2.cvtColor(canvas.astype(np.uint8),
                        cv2.COLOR_BGR2RGB)

    canvas=canvas.astype(np.float32)/255.0
    canvas=canvas[None,...]

    movenet.set_tensor(mn_in[0]["index"],canvas)
    movenet.invoke()

    kp=movenet.get_tensor(mn_out[0]["index"])[0,0]

    return kp,pad_top,pad_left,scale

# ====================================================
# SKELETON DRAWING
# ====================================================

CONNECTIONS=[
(0,1),(0,2),(1,3),(2,4),
(0,5),(0,6),(5,7),(7,9),
(6,8),(8,10),(5,6),(11,12),
(5,11),(6,12),(11,13),(13,15),
(12,14),(14,16)
]

def draw_skeleton(frame,kp,pad_top,pad_left,scale):

    h,w,_=frame.shape

    def map_point(y,x):

        px=(x*192-pad_left)/scale
        py=(y*192-pad_top)/scale

        return (
            int(np.clip(px,0,w-1)),
            int(np.clip(py,0,h-1))
        )

    for y,x,s in kp:
        if s<0.1: continue
        cv2.circle(frame,map_point(y,x),4,(0,255,0),-1)

    for a,b in CONNECTIONS:
        if kp[a][2]<0.1 or kp[b][2]<0.1:
            continue

        cv2.line(
            frame,
            map_point(kp[a][0],kp[a][1]),
            map_point(kp[b][0],kp[b][1]),
            (255,0,0),2
        )

# ====================================================
# HAND KEYPOINT DRAWING
# ====================================================

def draw_hand(frame,kp_flat):

    if kp_flat is None:
        return

    pts=kp_flat.reshape(21,2)

    for x,y in pts:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(
                frame,
                (int(x),int(y)),
                4,(0,255,0),-1
            )

# ====================================================
# CAMERA LOOP
# ====================================================

print("Starting combined realtime system")

cap=cv2.VideoCapture(0)

while True:

    ret,frame=cap.read()
    if not ret:
        break

    label="WAIT"
    color=(0,255,255)

    # ---------- FALL ----------
    kp,pt,pl,scale=run_movenet(frame)

    fall_buf.append(geom_from_kp(kp))

    draw_skeleton(frame,kp,pt,pl,scale)

    if len(fall_buf)==SEQ_LEN:

        X=np.array(fall_buf)[None,...]

        p=float(fall_model.predict(X,verbose=0)[0][0])

        if p>THRESH_FALL:
            label="FALL"
            color=(0,0,255)
        else:
            label="NO_FALL"
            color=(0,255,0)

    # ---------- HAND ----------
    results=yolo_hand(frame,verbose=False)

    if len(results)>0:

        r=results[0]

        if r.keypoints is not None:

            kp_hand=r.keypoints.xy

            if kp_hand is not None and len(kp_hand)>0:

                kp_flat=kp_hand[0].cpu().numpy()[:,:2].flatten()

                draw_hand(frame,kp_flat)

                try:
                    feat=extract_hand_features(kp_flat)

                    X=scaler.transform([feat])

                    prob=hand_model.predict(X,verbose=0)[0]

                    idx=int(np.argmax(prob))
                    conf=float(np.max(prob))

                    if conf>THRESH_GESTURE:
                        if CLASS_NAMES[idx]!="neutral":
                            label=CLASS_NAMES[idx]

                except Exception as e:
                    print(e)

    cv2.putText(frame,label,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,color,3)

    cv2.imshow("SilentGuardian Combined Realtime",frame)

    if cv2.waitKey(1)&0xFF in [27,ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()