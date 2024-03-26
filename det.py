# import os
# import cv2
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# def initialize_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     return cap

# def load_yolov8_model(model_path="yolov8n.pt"):
#     model = YOLO(model_path)
#     return model
# def save_frame(frame, save_folder, frame_count):
#     # Create the save folder if it doesn't exist
#     os.makedirs(save_folder, exist_ok=True)
    
#     # Save the frame to the save folder
#     save_path = os.path.join(save_folder, f"frame_{frame_count}.jpg")
#     cv2.imwrite(save_path, frame)
#     print(f"Frame saved: {save_path}")

# def process_frame(frame, model, box_annotator, region, save_folder, frame_count):
#     # Crop the frame to the specified region
#     x, y, width, height = region
#     cropped_frame = frame[y:y+height, x:x+width]
    
#     # Perform object detection on the cropped frame
#     result = model(cropped_frame, agnostic_nms=True)[0]
#     detections = sv.Detections.from_yolov8(result)
    
#     # Filter out detections corresponding to persons
#     person_detections = [
#         (x, confidence, class_id, bbox) 
#         for x, confidence, class_id, bbox in detections 
#         if model.model.names[class_id] == "person" and confidence > 0.5
#     ]
    
#     labels = [
#         f"person {confidence:0.2f}" 
#         for _, confidence, _, _ in person_detections
#     ]
    
#     # Annotate the cropped frame with the detections
#     annotated_cropped_frame = box_annotator.annotate(
#         scene=cropped_frame,
#         detections=person_detections,
#         labels=labels
#     )
    
#     # Save the frame if any person is detected within the region
#     if person_detections:
#         save_frame(frame, save_folder, frame_count)
    
#     # Replace the region in the original frame with the annotated cropped frame
#     annotated_frame = frame.copy()
#     annotated_frame[y:y+height, x:x+width] = annotated_cropped_frame
    
#     return annotated_frame

# def main():
#     video_path = '/home/hussain/Desktop/FSAM/pexels_videos_2670 (1080p).mp4'  # Adjust this with your video file path
#     save_folder = 'detected_frames'  # Folder to save the detected frames
#     cap = initialize_video(video_path)
#     model = load_yolov8_model()
#     box_annotator = sv.BoxAnnotator(
#         thickness=1,
#         text_thickness=1,
#         text_scale=1
#     )
#     region = (260, 200, 400, 200 )  # Adjust these values for your specific region
#     frame_count = 0  # Counter for the frames
    
#     # Set custom window size
#     cv2.namedWindow("yolov8", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("yolov8", 800, 600)  # Customize width and height as needed

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to capture frame")
#             break

#         annotated_frame = process_frame(frame, model, box_annotator, region, save_folder, frame_count)
#         frame_count += 1

#         cv2.imshow("yolov8", annotated_frame)

#         if cv2.waitKey(20) == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import os
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def load_yolov8_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    return model

def process_frame(frame, model, box_annotator, region, save_folder, frame_counter):
    # Extract region coordinates and dimensions
    x, y, width, height = region
    
    # Draw the region rectangle on the frame
    annotated_frame = frame.copy()
    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (250, 0, 255), 2)  # Changed color to red
    
    # Crop the frame to the specified region
    cropped_frame = frame[y:y+height, x:x+width]
    
    # Perform object detection on the cropped frame
    result = model(cropped_frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    
    # Filter out detections corresponding to persons
    person_detections = [
        (x, confidence, class_id, bbox) 
        for x, confidence, class_id, bbox in detections 
        if model.model.names[class_id] == "person" and confidence > 0.5
    ]
    
    # Annotate the cropped frame with the detections
    annotated_cropped_frame = box_annotator.annotate(
        scene=cropped_frame,
        detections=person_detections,
        labels=["person" for _ in person_detections]
    )
    
    # Replace the region in the original frame with the annotated cropped frame
    annotated_frame[y:y+height, x:x+width] = annotated_cropped_frame
    
    # Save the frame if a person is detected
    if person_detections:
        filename = os.path.join(save_folder, f"frame_{frame_counter}.jpg")
        cv2.imwrite(filename, frame)
    
    return annotated_frame


def main():
    video_path = '/home/hussain/Desktop/FSAM/pexels_videos_2670 (1080p).mp4'  # Adjust this with your video file path
    save_folder = 'detected_frames'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    cap = initialize_video(video_path)
    model = load_yolov8_model()
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )
    region = (460, 710, 240, 240 )  # Adjust these values for your specific region

    # Set custom window size
    cv2.namedWindow("PROTOTYPE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PROTOTYPE", 800, 600)  # Customize width and height as needed

    frame_counter = 0  # Initialize frame counter
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        annotated_frame = process_frame(frame, model, box_annotator, region, save_folder, frame_counter)
        frame_counter += 1  # Increment frame counter

        cv2.imshow("PROTOTYPE", annotated_frame)

        if cv2.waitKey(5) == 5:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

