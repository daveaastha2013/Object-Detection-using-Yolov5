import cv2
import torch
import pandas as pd

pandas = pd

# Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Loading the image
image_path = "/home/aastha/Pictures/bus.jpg"


image = cv2.imread(image_path)

# Checking if the image was loaded successfully
if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Applying model
results = model(image)

# Getting results
results_df =results.pd().xyxy[0]  # Getting the results as a DataFrame
print(results_df)  # Printing the df for debugging

# Checking for detections
if results_df.empty:
    print("No objects detected.")
else:
    confidence_threshold = 0.5

    # Boxes and Labels
    for index, row in results_df.iterrows():
        if row['confidence'] >= confidence_threshold:  # Only process confident detections
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            # (xmin,ymin) represent top left corner of the box
            # (xmax,ymax) represent bottom right corner of the box
            label = f"{model.names[int(row['class'])]}: {row['confidence']:.2f}"

            # Drawing Box
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

            # Adding label
            cv2.putText(image, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Output coordinates 
            print(f"Detected {label} at coordinates: ({int(xmin)}, {int(ymin)}) to ({int(xmax)}, {int(ymax)})")

# Modified image
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
