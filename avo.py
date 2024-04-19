import cv2
import numpy as np

# Function to perform obstacle detection
def detect_obstacles(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize number of obstacles
    num_obstacles = 0

    # Iterate through contours
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)

        # If contour area is below a certain threshold, ignore it (noise)
        if area < 500:
            continue

        # Increment number of obstacles
        num_obstacles += 1

        # Draw contour
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    return frame, num_obstacles

# Function to perform obstacle avoidance
def avoid_obstacles(num_obstacles):
    # Dummy avoidance logic
    if num_obstacles > 0:
        print("Obstacle detected! Stop the robot and change direction.")
    else:
        print("No obstacles detected. Proceeding.")

# Main function
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Change the argument to the camera index if not using default camera

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame")
            break

        # Perform obstacle detection
        processed_frame, num_obstacles = detect_obstacles(frame)

        # Perform obstacle avoidance
        avoid_obstacles(num_obstacles)

        # Display the processed frame
        cv2.imshow('Obstacle Avoidance', processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
