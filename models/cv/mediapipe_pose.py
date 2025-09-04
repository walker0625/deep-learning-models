import sys
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
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

    #######포즈 그리기#########
    frame.flags.writeable = True
    
    results = pose.process(frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
    #########################

    cv2.imshow('webcam', frame)
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    