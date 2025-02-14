import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCQML7hWSstv_7vWqsh3lhO7JyC2ENDlAw"

# ✅ Initialize Gemini Model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Load YOLO Model
yolo_model = YOLO("best.pt")
names = yolo_model.names

# ✅ Open Video File
cap = cv2.VideoCapture('vid.mp4')
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# ✅ Constants for Tracking
current_date = time.strftime("%Y-%m-%d")
crop_folder = f"crop_{current_date}"
os.makedirs(crop_folder, exist_ok=True)

processed_track_ids = set()  # Track processed IDs to avoid duplicates

# ✅ Utility Functions
def encode_image_to_base64(image):
    """Convert an image to a base64 string."""
    _, img_buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(img_buffer).decode('utf-8')

def analyze_image_with_gemini(current_image, track_id):
    """Analyze image with Gemini to determine excavator status."""
    if current_image is None:
        return "No image available for analysis."

    image_data = encode_image_to_base64(current_image)

    message = HumanMessage(
        content=[
            {"type": "text", "text": f"""
            Analyze the following image and provide structured details about the excavator:
            
            1. **Working Status**: Is the excavator actively working or in standby mode?
            2. **Color**: What is the excavator's primary color?
            3. **Company Name**: Identify the company name/logo on the excavator.
            4. **Track ID**: {track_id}

            Return the result in a structured table format:

            | Track ID | Status  | Color | Company Name |
            |----------|---------|-------|--------------|
            | {track_id} | Working/Standby | Color Name | Company Name |
            """},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    )

    try:
        response = gemini_model.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        return "Error processing image."

def save_response_to_file(track_id, response):
    """Save analysis response to a text file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    response_filename = f"gemini_response_{current_date}_report.txt"

    try:
        with open(response_filename, "a", encoding="utf-8") as file:
            file.write(f"Track ID: {track_id} | Condition: {response} | Date: {timestamp}\n\n")
        print(f"Response saved to {response_filename}")
    except Exception as e:
        print(f"Error saving response to file: {e}")

def save_crop_image(crop, track_id):
    """Save cropped excavator image."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{crop_folder}/{track_id}_{timestamp}.jpg"
    try:
        cv2.imwrite(filename, crop)
        print(f"Cropped image saved: {filename}")
    except Exception as e:
        print(f"Error saving cropped image: {e}")
    return filename

def crop_and_process(frame, box, track_id):
    """Crop and process detected excavator image."""
    if track_id in processed_track_ids:
        return  # Avoid re-processing same track_id

    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    crop_filename = save_crop_image(crop, track_id)

    processed_track_ids.add(track_id)

    threading.Thread(target=process_crop_image, args=(crop, track_id, crop_filename)).start()

def process_crop_image(current_image, track_id, crop_filename):
    """Process and analyze cropped excavator image."""
    response_content = analyze_image_with_gemini(current_image, track_id)
    print(f"Gemini Response for Track ID {track_id}:\n{response_content}")

    save_response_to_file(track_id, response_content)

    response_filename = crop_filename.replace(".jpg", "_response.txt")
    try:
        with open(response_filename, "w", encoding="utf-8") as f:
            f.write(f"Track ID: {track_id}\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\nResponse: {response_content}\n")
    except Exception as e:
        print(f"Error saving response file: {e}")

def process_video_frame(frame):
    """Process video frame for object detection and tracking."""
    frame = cv2.resize(frame, (1020, 500))

    results = yolo_model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            excavator_label = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'Track ID: {track_id}', (x2, y2), 1, 1)
            cvzone.putTextRect(frame, f'{excavator_label}', (x1, y1), 1, 1)
            crop_and_process(frame, box, track_id)


           

    return frame

def main():
    """Main function to process video frames."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_video_frame(frame)
        

        cv2.imshow("Excavator Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
