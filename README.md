**Object Detection Web App with YOLOv8**

This Streamlit web app allows you to upload an image, detect objects within it using the YOLOv8 model, and receive the following functionalities:

- **Object detection and bounding boxes:** The app identifies objects in the image and displays them with bounding boxes for visual clarity.
- **Object counting and classification:** It counts the number of instances of each detected object and provides a detailed breakdown of their categories.
- **Text-to-speech (TTS) output:** The app reads the detected objects and their counts aloud using pyttsx3, enhancing accessibility and user experience.
- **Image download:** Users can download the processed image with bounding boxes for further analysis or record-keeping.

**Features:**

- Leverages the powerful YOLOv8 model for real-time object detection.
- Provides a user-friendly interface with image uploading and download capabilities.
- Offers both visual and auditory (TTS) outputs for object information.
- **(Future Addition)** Considers extending functionality to handle video processing.

**Requirements:**

- Python 3.x
- Streamlit (`pip install streamlit`)
- OpenCV-Python (`pip install opencv-python`)
- cvzone (`pip install cvzone`)
- PyTorch (`pip install torch torchvision`) (ensure compatibility with YOLOv8)
- ultralytics (YOLOv8 library) (`pip install ultralytics`) (`git clone https://github.com/ultralytics/yolov8 && cd yolov8 && pip install -r requirements.txt`)
- pyttsx3 (`pip install pyttsx3`)

**Usage:**

1. Clone this repository or download the code files.
2. Install the required Python libraries mentioned above.
3. Make sure you have a YOLOv8 model downloaded and placed in the `../Running Yolo/` directory (adjust the path if needed).
4. Run the app using `streamlit run app.py`.
5. Upload an image in a supported format (JPG, PNG, JPEG).
6. The app will process the image and display the detected objects, their counts, and offer a download option.

**Contributors:**

- Umang Barewar
- Aishwarya Joshi (GitHub: https://github.com/AishwaryaJoshi087)
- Jayushna Mahadule (GitHub: https://github.com/JayushnaMahadule)
- Aayush Zade (GitHub: https://github.com/AayushZade)
- Rutuja Balbudhe (Github:
  https://github.com/Rutufied)

**Disclaimer:**

This code is provided for educational and demonstration purposes. The authors are not responsible for any misuse or unintended consequences of its use.

**Future Enhancements:**

- Video processing capabilities to analyze moving objects.
- More comprehensive object class detection support.
- User interface improvements for better interaction and information display.
- Cloud deployment for wider accessibility and scalability.

**Feel free to fork this repository and contribute to its ongoing development!**
