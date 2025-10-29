import cv2
import mediapipe as mp
import numpy as np
import string
import os



mp_selfie_segentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segentation.SelfieSegmentation(model_selection=1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=20 , detectShadows=False)

chars = np.array(list(string.printable[:94]))
font_ratio=0.45

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    frame= cv2.resize(frame,(120,48))
    h,w,_ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame= cv2.flip(frame ,1)
    
    rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    result = segmentation.process(rgb)
    mask_mp = result.segmentation_mask
    masp_mp = (mask_mp>0.5).astype(np.uint8)

    mask_fg = fgbg.apply(frame)
    mask_fg = cv2.threshold(mask_fg, 200,1,cv2.THRESH_BINARY)[1]

    mask_fg = cv2.resize(mask_fg, (frame.shape[1], frame.shape[0]))
    mask_mp = cv2.resize(mask_mp, (frame.shape[1], frame.shape[0]))
    mask_mp = (mask_mp * 255).astype(np.uint8)
    mask_fg = (mask_fg * 255).astype(np.uint8)

    combined = cv2.bitwise_or(mask_mp, mask_fg)


    combined=cv2.medianBlur(combined,5)

    _ , combined = cv2.threshold(combined, 200,255, cv2.THRESH_BINARY)

    ascii_frame = ""
    for y in range(0,h):
        for x in range(0,w):
            if combined[y,x]>0:
                brightness = gray[y,x]
                char = chars[int(brightness/256*len(chars))]
                ascii_frame+=char
            else:
                ascii_frame+="."
        ascii_frame+="\n"
    os.system('cls' if os.name=='nt' else 'clear')
    print(ascii_frame)

cap.release()