import cv2
import torch



# Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Loading the video
video_path = "/home/aastha/Pictures/water2.mp4"

# Opening the video
cap = cv2.VideoCapture(video_path)

# Checking if video opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Setting a confidence threshold
confidence_threshold = 0.5

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Break the loop if there are no more frames
    if not ret:
        break
    
    # Perform inference
    results = model(frame)

    # Parse results
    results_data = results.pandas().xyxy[0]  # Get the results as a DataFrame
    
    # Check if there are any detections
    if not results_data.empty:
        # Boxes and Labels
        for index, row in results_data.iterrows():
            if row['confidence'] >= confidence_threshold:  # Only process confident detections
                xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
                label = f"{model.names[int(row['class'])]}: {row['confidence']:.2f}"

                # Drawing the box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Adding the label
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Output the coordinates
                print(f"Detected {label} at coordinates: ({int(xmin)}, {int(ymin)}) to ({int(xmax)}, {int(ymax)})")

    # Modified frame
    cv2.imshow("Detected Objects", frame)


# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
