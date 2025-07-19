# Object-Detection-using-Yolov5

### Overview
- A Yolov5 model is implemented to detect objects with their coordinates in images, videos and through web cameras.
- Used it to detect landmines from images and videos.

### To run the code
- Modify the ```image_path``` or ```video_path``` variables to store the full path of your image/video file.
- The properties of the box enclosing the detected objects can be modified under  ```Boxes and Labels```.

### Input and Output
- The input is in the form of image/video/live web cam (attached).
- Output gives a modified version of the image/video where the detecteed objects are enclosed within rectangles along with their individual confidences.
- For detection using live webcam, the code opens the webcam and performs real-time object detection.
- Apart from this, the name of the object detected along with its range of x,y coordinates and confidence is printed to the output terminal.
- The output terminal also shows the created data frame for results, thus making it easier to debug the code.
- We receive an average output confidence of >0.8 for images. For videos and webcams the confidence is >0.85 for evident objects and 0.7-0.8 for hidden objects.

### Future Work
- Modifying the model to predict hidden objects in videos and webcams with much more precision.
- Make it precise in detecting hidden landmines as well.





