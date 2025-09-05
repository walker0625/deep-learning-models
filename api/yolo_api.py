from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title='YOLO API')

@app.post('/yolo')
def yolo_check_image(
    image_file: UploadFile=File(...)
):
    image = Image.open(image_file.file)
    
    model = YOLO('../../temp/yolo11n.pt')
    results = model(image)
    boxes = results[0].boxes
    
    data = []
    
    for x1, y1, x2, y2, conf, cls in boxes.data: 
        temp = [x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item()]
        data.append(temp)
    
    return data