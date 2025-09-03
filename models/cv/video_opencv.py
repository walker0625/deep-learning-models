import sys
import cv2

vcap = cv2.VideoCapture(0)

while True:
    
    ret, frame = vcap.read()
    if not ret:
        print('카메라 오류')
        sys.exit()
    
    # 좌우 반전    
    flipped_frame = cv2.flip(frame, 1)    
    
    cv2.imshow('webcam', flipped_frame)
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    