Project 3: Real-time 2-D Object Recognition Team Members
Ronak Bhanushali
Ruohe Zhou


Video Submission
Video of everything running at once - https://www.youtube.com/watch?v=slFUOzBhv34
Videos recorded with video writer of opencv- 
Segmented output- https://youtu.be/xMUBubQdtx4
Features and bounding boxes - https://youtu.be/SR4mEE1rj2o
Note - Videos are different since both of us tried using different objects at our disposal


Operating System and IDE
Ronak -
OS: Linux
IDE: VSCode
Ruohe -
OS: MacOS
IDE: VSCode

Running the Code

ONNX FILE LINK - https://drive.google.com/file/d/1OmCGkUhHen8_ngN-cVLqLigUC5jRfixy/view?usp=sharing

Run the cmake and make. 
The following executables work the following way -
1. cleaned_frame
Shows the cleaned and thresholded image
2. colormap
Shows the segmented frame
3. task4
Shows the bounding box on objects with axis of least moment
4. task5
Saves features to csv. Press n to make a new entry. Then name the object from the terminal
5. task6
Shows the best match for unknown object. Press i for inference
6. task9
Saves images for reference on keypress "a". Inferes for unknown object on keypress "i". Uses DNN for inference

NOTE - Make sure to replace paths in all files before executing


If you completed any extensions, follow these instructions to test them:
1. Written two functions from scratch
2. Used 7 objects instead of 5
3. Used custom network and custom python script for DNN inference. Calling from c++ to run DNN. Just run task9 to test

Time Travel Days
Yes. Using 2 time travel days
