import cv2

# Load the video
video_path = '/home/hussain/Desktop/FSAM/pexels_videos_2670 (1080p).mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video")
    exit()

# Get the video's frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region you want to crop out
x, y, width, height = 260, 200, 400, 200  # Example coordinates and dimensions, adjust as needed

# Create a VideoWriter object to write the cropped video
output_path = 'output_cropped_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Crop the frame to the defined region
        cropped_frame = frame[y:y+height, x:x+width]
        
        # Write the cropped frame to the output video
        out.write(cropped_frame)
        
        # Display the cropped frame (optional)
        cv2.imshow('Cropped Video', cropped_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
