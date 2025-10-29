import cv2
import mediapipe as mp
import numpy as np
import string
import curses

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

chars = np.array(list(string.printable[:94]))
font_ratio = 0.45

def main(stdscr):
    curses.curs_set(0)
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
        mask_mp = (mask_mp > 0.5).astype(np.uint8) * 255
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

        stdscr.clear()
        h_max, w_max = stdscr.getmaxyx()
        for i, line in enumerate(ascii_frame.splitlines()):
            if i >= h_max:
                break
            stdscr.addstr(i, 0, line[:w_max-1])
        stdscr.refresh()
        

    cap.release()

curses.wrapper(main)
