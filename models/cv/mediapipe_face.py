import sys
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, # 0:근거리 / 1:원거리
    min_detection_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    refine_landmarks=True # 동공까지 체크할지 확인
)

vcap = cv2.VideoCapture(0)

while True:
    
    ret, frame = vcap.read()
    
    if not ret:
        print('카메라 오류')
        sys.exit()
    
    # 좌우 반전    
    frame = cv2.flip(frame, 1)    
    
    ###### 얼굴 찾기 ##########
    # frame.flags.writeable = True
    
    # detection_results = face_detection.process(frame)
    
    # if detection_results.detections:
    #     for detection in detection_results.detections:
    #         mp_drawing.draw_detection(frame, detection)
    ############################
    
    ###### Face Landmark 자동으로 그리기######
    frame.flags.writeable = True
    
    mesh_results = face_mesh.process(frame)
    
    nose_arr = [1]
    left_eye_arr = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye_arr = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    mouth_arr = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,
                185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]

    total_arr = left_eye_arr + right_eye_arr + nose_arr + mouth_arr
    
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            
            # 원하는 부분 그리기
            # landmarks = face_landmarks.landmark
            
            # h, w, _ = frame.shape
            
            # for idx in total_arr:
                
            #     point_x = int(landmarks[idx].x * w)
            #     point_y = int(landmarks[idx].y * h)
    
            #     cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)    
    
            # 자동 그리기        
            # mp_drawing.draw_landmarks(
            #     frame, 
            #     face_landmarks, 
            #     mp_face_mesh.FACEMESH_TESSELATION,
            #     mp_drawing.DrawingSpec(
            #         color=(0, 255, 10), 
            #         thickness=1, 
            #         circle_radius=1
            #     )
            # )
            h, w, _ = frame.shape
            
            landmarks = face_landmarks.landmark
            landmark1 = landmarks[13]
            landmark2 = landmarks[14]
            
            point_x1 = int(landmark1.x * w)
            point_y1 = int(landmark1.y * h)
            point_x2 = int(landmark2.x * w)
            point_y2 = int(landmark2.y * h)
            
            cv2.circle(frame, (point_x1, point_y1), 5, (0, 0, 255), 2)
            cv2.circle(frame, (point_x2, point_y2), 5, (0, 0, 255), 2)
            cv2.line(frame, (point_x1, point_y1), (point_x2, point_y2), (0, 0, 255), 3)
            
            distance = ((point_x2 - point_x1) ** 2 + (point_y2 - point_y1) ** 2) ** 0.5 # root를 씌우는 효과
            print(f'distance: {distance}')
            
            if distance > 20:
                print('be quite please')
            
    cv2.imshow('webcam', frame)
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    