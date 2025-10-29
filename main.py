import cv2
import mediapipe as mp
import numpy as np
import string
import os

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

chars = np.array(list(string.printable[:94]))
font_ratio = 0.45

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (120, 48))
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentation.process(rgb)

    mask_mp = result.segmentation_mask
    mask_mp = (mask_mp > 0.5).astype(np.uint8) * 255  # binary mask of person only

    # smooth mask for cleaner borders
    mask_mp = cv2.medianBlur(mask_mp, 5)

    ascii_frame = ""
    for y in range(h):
        for x in range(w):
            if mask_mp[y, x] > 0:
                brightness = gray[y, x]
                char = chars[int(brightness / 256 * len(chars))]
                ascii_frame += char
            else:
                ascii_frame += "."
        ascii_frame += "\n"

    os.system('cls' if os.name == 'nt' else 'clear')
    print(ascii_frame)

cap.release()
