import cv2
import os
import time


def capture_images(name):
    cap = cv2.VideoCapture(0)
    TIMER = int(1)
    start_time = time.time()

    while True:
        ret, img = cap.read()
        cv2.imshow('frame', img)

        if ret and TIMER >= 0:
            current_time = time.time()
            cv2.imwrite(f'ai_module/facebank/{name}_{time.thread_time_ns()}.jpg', img)
            
            if current_time-start_time >= 1:
                start_time = current_time
                TIMER = TIMER-1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# capture_images("Ganesh")