import cv2
import torch
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Opening the webcam (0 usually refers to the default camera)
cap = cv2.VideoCapture(0)

# Checking if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set confidence threshold
confidence_threshold = 0.5

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Break the loop if the frame was not retrieved
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform inference
    results = model(frame)

    # Parse results
    results_data = results.pandas().xyxy[0]  # Get the results as a DataFrame
    
    # Check if there are any detections
    if not results_data.empty:
        # Draw boxes and labels
        for index, row in results_data.iterrows():
            if row['confidence'] >= confidence_threshold:  # Only process confident detections
                xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
                label = f"{model.names[int(row['class'])]}: {row['confidence']:.2f}"

                # Draw the bounding box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Draw the label
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Output the coordinates
                print(f"Detected {label} at coordinates: ({int(xmin)}, {int(ymin)}) to ({int(xmax)}, {int(ymax)})")

    # Show the modified frame with detections
    cv2.imshow("Detected Objects", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
