import cv2
import numpy as np
import argparse
import os
from nn import EmotionCNN

ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="display only (train mode removed)")
mode = ap.parse_args().mode

if mode == "display":
    cnn = EmotionCNN('model.h5')

    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    face_net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        fh, fw = frame.shape[:2]
        ratio = min(fh, fw) / 480.0
        fnt = round(1.0 * ratio, 2)
        thick = max(1, int(3 * ratio))
        ox, oy = int(20 * ratio), int(60 * ratio)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - oy + 10), (x + w, y + h + 10), (255, 0, 0), thick)
            roi_gray = gray[y:y + h, x:x + w]
            cropped = cv2.resize(roi_gray, (48, 48))
            prediction = cnn.predict(cropped)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + ox, max(y - oy, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, fnt, (255, 255, 255), thick, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Only --mode display is supported without TensorFlow.")