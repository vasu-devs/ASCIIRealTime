import cv2,os,shutil,time, string
import numpy as np

cap = cv2.VideoCapture(r"E:\STUDY\Projects\SMALL\AsciiRealTime\video.mp4", cv2.CAP_FFMPEG)


chars = string.printable
chars = [c for c in chars if c.strip()] 

term_width , _ = shutil.get_terminal_size()
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10, detectShadows=False)

def frame_to_ascii(frame, width=term_width):
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = fgbg.apply(frame)
    mask = cv2.threshold(mask , 200,255,cv2.THRESH_BINARY)[1]


    h,w=gray.shape
    aspect_ratio=h/w
    new_height=int(aspect_ratio*width*0.55)
    gray=cv2.resize(gray,(width,new_height))
    mask=cv2.resize(mask,(width,new_height))

    gray_norm = (gray/255)*(len(chars)-1)
    ascii_frame="\n".join(
        ''.join(
            chars[int(gray_norm[y,x])] if mask[y,x] > 0 else "." 
            for x in range(width)
            )
            for y in range(new_height)
        )
    return ascii_frame
    


while True:
    frame_duration = 1 / 30  
    ret , frame = cap.read()
    if not ret:
        break

    ascii_frame=frame_to_ascii(frame)
    os.system('cls' if os.name=='nt' else 'clear' )
    # cv2.imshow("Video", frame)
    print(ascii_frame)
    time.sleep(frame_duration)


cap.release()