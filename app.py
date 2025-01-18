from flask import Flask, render_template, request
# from main import predict512, predict100
from watershed import detection, detection2
from PIL import Image
import numpy as np
import os.path
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])


def predict():
    imageFile = request.files['file_image']
    folderRgb = './static/uploads/rgb/'
    folderGray = './static/uploads/gray/'
    folderDetect = './static/uploads/detect/'
    
    # ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี
    if not os.path.exists(folderRgb):
        os.makedirs(folderRgb)
    if not os.path.exists(folderDetect):
        os.makedirs(folderDetect)
    if not os.path.exists(folderGray):
        os.makedirs(folderGray)
    
    # บันทึกรูปต้นฉบับในโฟลเดอร์ RGB
    pathImage = os.path.join(folderRgb, imageFile.filename)
    imageFile.save(pathImage)
    
    # แปลงรูปภาพเป็น Grayscale และบันทึกในโฟลเดอร์ gray
    grayscale_image = Image.open(pathImage).convert('L')
    pathImageGray = os.path.join(folderGray, f"gray_{imageFile.filename}")
    grayscale_image.save(pathImageGray)
    # Detect แล้วบันทึกโฟรเดอร์ detect
    detected_image, cout_detected= detection(pathImage)
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    pathImageDetected = os.path.join(folderDetect, f"detected_{imageFile.filename}")
    cv2.imwrite(pathImageDetected, detected_image_rgb)
    # ผลการวิเคราะห์ (ตัวอย่าง)
    prediction = "โรคราน้ำค้าง"
    
    # ส่งข้อมูลทั้งสองรูปกลับไปที่หน้าเว็บ
    return render_template(
        'index.html', 
        results=[
            (prediction, pathImage, pathImageGray, pathImageDetected, cout_detected)
        ]
    )




# def predictOld():
#     image_file = request.files['file_image']
#     rgb_folder = './static/uploads/rgb/'
#     detect_folder = './static/uploads/detect/'

#     image_path = os.path.join(rgb_folder, image_file.filename)
#     image_path_detect = os.path.join(detect_folder, image_file.filename)

#     image_file.save(image_path)

#     models = [(predict512, '_512'), (predict100, '_100')]
#     results = []

#     for model, suffix in models:
#         pred, image_detect = model(image_path)
#         if image_detect is not None:
#             image_detect = Image.fromarray(image_detect)
#             image_detect.save(image_path_detect + suffix + '.png')
#             results.append((pred, image_path_detect + suffix + '.png',f'Feature:{suffix}'))
#         else:
#             results.append((pred, image_path, f'Feature:{suffix}'))

#     return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
