import sys
import cv2
import mediapipe as mp

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
    
    v_hand = [5, 6, 7, 8, 9, 10, 11, 12]
    t_hand = [4, 3, 2, 1]
    f_hand = [9, 10 ,11, 12]
    
    if results.multi_hand_landmarks:
        
        print(len(results.multi_hand_landmarks)) # 손의 수를 출력
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 한 손의 좌표 수(21개)
            print(len(hand_landmarks.landmark)) 
            
            h, w, _ = frame.shape
            
            # x(0~1), y(0~1), z 
            # x/y로 이미지 사이즈에 맞게 변환 / z는 카메라와 거리인데 부정확
            for idx, landmark in enumerate(hand_landmarks.landmark):
                
                if idx in v_hand:
                    print(f'{idx} : {landmark.x}, {landmark.y}')

                    # frame의 크기에 맞게 비율을 곱해줌
                    point_x = int(landmark.x * w)
                    point_y = int(landmark.y * h)

                    cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

            # 자동 그리기
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )
    #######################################
    
    cv2.imshow('webcam', frame)
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    