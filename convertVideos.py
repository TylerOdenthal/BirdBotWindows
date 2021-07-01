import cv2
from datetime import datetime

videoName = "HouseSparrow-9"
cap = cv2.VideoCapture(videoName + '.mp4')

now = datetime.now()
current_time = now.strftime("%M")

count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(videoName + "-" + current_time + '-frame{:d}.jpg'.format(count), frame)
        count += 15 # i.e. at 30 fps, this advances one second
        cap.set(1, count)
    else:
        cap.release()
        break