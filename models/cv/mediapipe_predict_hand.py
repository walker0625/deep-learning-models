import sys
import cv2
import mediapipe as mp
import joblib
import numpy as np

# 손 관절 추출

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = joblib.load("./rcp.pkl")
labels = ["rock", "scissors", "paper"]

vcap = cv2.VideoCapture(0)

while True:
    
    ret, frame = vcap.read()
    
    if not ret:
        print('카메라 오류')
        sys.exit()
    
    # 좌우 반전    
    frame = cv2.flip(frame, 1)    
    # contrast_frame = 255 - flipped_frame
    
    ####### 손 그리기 설정 ##################
    frame.flags.writeable = True
    
    results= hands.process(frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape
            
            landmarks = []
        
            for landmark in hand_landmarks.landmark:
                ## 좌표 모으기
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                ## 그리기
                point_x = int(landmark.x * w)
                point_y = int(landmark.y * h)
                
                cv2.circle(frame, (point_x, point_y), 5, (0,0,255), 2)
            
            pred = model.predict(np.array([landmarks]))
            print(pred[0])
            cv2.putText(frame, labels[pred[0]], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)    
            
    #######################################
    
    cv2.imshow('webcam', frame)
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    