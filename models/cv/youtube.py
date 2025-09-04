# uv add yt_dlp

import yt_dlp # 유튜브 비디오 다운로드
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('../../temp/yolo11n.pt')

youtube_url = 'https://youtu.be/S5nsDT5oU90'

ydl_opts = {
    'format': 'bestvideo[ext=mp4][protocol=https]/best',
    'quite': True,
    'no_warnings': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=False)
    stream_url = info_dict['url']
    

vcap = cv2.VideoCapture(stream_url)

window_name = 'Youtube Video'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # 창 크기 조절 가능
cv2.resizeWindow(window_name, 960, 540)           # 원하는 창 크기

while True:
    if not vcap.isOpened():
        print('비디오를 열 수 없습니다')
        break
    
    ret, frame = vcap.read()

    if not ret:
        print('비디오 프레임을 읽어 올 수 없습니다')
        break
    
    # YOLO로 객체 탐지
    results = model(frame, conf=0.7) 
    result = results[0]
    boxes = result.boxes
        
    cnt = 0
        
    for x1, y1, x2, y2, conf, idx in boxes.data: # cls는 예약어라 idx로 씀
        
        # person만 box 그리기
        if idx > 0:
            continue
        
        cnt += 1
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # box 그리기(frame, 좌상단, 우하단, 색상, 두께)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cnt_text = f'person: {cnt}'
    
    cv2.putText(frame, cnt_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
    cv2.imshow(window_name, frame)    
    
    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    
vcap.release()
cv2.destroyAllWindows()    
    
