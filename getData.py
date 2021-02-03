import cv2
import os
import sys

label = sys.argv[1]

SAVE_PATH = os.path.join(os.getcwd(), "data", label)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

ct = int(sys.argv[2])
maxCt = int(sys.argv[3]) + 1
print("Hit Space to Capture Image")

cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read()
    cv2.imshow("Get Data : " + label,frame[50:350, 150:450])
    if cv2.waitKey(1) & 0xFF == ord(" "):
        cv2.imwrite(os.path.join(SAVE_PATH, "{}{}.jpg".format(label, ct)), frame[50:350, 150:450])
        print("{}{}{}.jpg Captured".format(os.path.join(SAVE_PATH, "{}{}.jpg".format(label, ct)), label, ct))
        ct += 1
    if ct >= maxCt:
        break

cap.release()
cv2.destroyAllWindows()
