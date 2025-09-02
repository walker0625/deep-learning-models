# 모델 불러오기
from ultralytics import YOLO
model = YOLO('../../temp/yolo11n.pt') #yolo11n 부분은 모델명을 적어야 함

# 비디오 예측
video_link = 'https://youtu.be/bS1VQllcmI4'
results = model(video_link, stream=True, show=True)

# 진행 파악
for result in results:
    print(result)