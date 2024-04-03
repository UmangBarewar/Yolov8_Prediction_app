import io
import cv2
import cvzone
import math
import streamlit as st
from collections import Counter
import pyttsx3
from ultralytics import YOLO

# Load COCO class names
coco_classes = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',tre
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant',
    'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def detect_objects(image_path):
    detected_classes = set()
    model=YOLO('../Running Yolo/yolov8l.pt')
    input_image = cv2.imread(image_path)
    results = model(input_image)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            cvzone.cornerRect(input_image, (x1, y1, w, h), colorR=(255, 0, 255), colorC=(0, 255, 0))
            cvzone.putTextRect(input_image, f'{coco_classes[cls]} ', (max(0, x1), max(35, y1)),
                               scale=0.6, thickness=1, offset=3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            class_name = coco_classes[cls]
            detected_classes.add(class_name)

    total_objects = len(detected_classes)
    class_counts = Counter(detected_classes)
    class_info = "\n".join([f"{class_name} has occurred {count} times" for class_name, count in class_counts.items()])
    return input_image, total_objects, class_info

st.title('Object Detection with YOLOv8')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    image, total_objects, class_info = detect_objects(image_path)
    st.image(image, caption='Object Detection Result', use_column_width=True)
    
    st.write(f"Total types of objects detected: {total_objects}")
    st.write("Classes and their counts:")
    st.write(class_info)

    # For downloading image
    # with open("temp_image.jpg", "rb") as file:
    #     btn = st.download_button(
    #         label="Download image",
    #         data=file,
    #         file_name="object_detection_result.jpg",
    #         mime="image/jpeg"
    #     )

    engine = pyttsx3.init()
    engine.say("In the image, the following objects were detected:")
    engine.say(class_info)
    engine.runAndWait()
    
    img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
    btn = st.download_button(
        label="Download image",
        data=io.BytesIO(img_bytes),
        file_name="object_detection_result.jpg",
        mime="image/jpeg"
    )

