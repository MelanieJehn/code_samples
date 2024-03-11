# Natural HRI
Provides a framework for natural HRI consisting of 5 components. A BT controls the robot. The ASR module converts speech to text. The Classifier takes the text and classifies which action the robot should take. The BT can call the classify server to receive the last classification result. The Object Detector detects objects in front of the robot. The Context Extractor uses text and object detections to extract context information about the task. The BT can call the context server to receive this information which is needed to execute an action.


## Installation
The following steps need to be followed to successfully install the project.

### ASR
wav2vec2: git clone https://github.com/oliverguhr/wav2vec2-live  
clone into ASR_and_transformers directory  
pip install -r requirements.txt  
On ubuntu might need:  
sudo apt install portaudio19-dev   
  
For microphone usage:  
pip install pyaudio  

### YOLO Detector

>The Yolo models will be downloaded automatically from their official websites, if not present in the workspace.
>If the downloading fails, download manually.
>- Yolov5: https://github.com/ultralytics/yolov5/releases
>- YoloV7: https://github.com/WongKinYiu/yolov7/releases

>It is recommended not to clone the whole repository of the YOLO as we are only using a specific version of the model.

### BT
Make sure you have the required packages in your workspace, on which the additional_treenodes and nhr_bt packages depend. No further installations are required.


## Usage
Build: catkin build  
Resource: source devel/setup.bash  
Start a roscore
```
roscore
```

All scripts listed here need to be started.
> The Camera scripts need to be started in the following order of steps,

> 1) Start the **camera node** using the command:
>```
>roslaunch realsense2_camera rs_camera.launch filters:=pointcloud align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 
>```

>Make sure that the camera is connected to the USB 3.0 or higher USB port version. It can be checked by typing the command 'realsense-viewer' in the terminal to open the camera viewer.

> 2) **model_sub.py**:  
Start the YOLO detector.
>```
>rosrun inference module_sub.py
>```

> 3) **display_results.py**: <br />
>Start the viewer node to see the object detection and inferences in real time. A separate window will be opened.
>```
>rosrun inference display_results.py
>```

Use this script to start the Franka Emika Panda **robot hardware**:
```
roslaunch panda_launch panda_hardware.launch robot_ip:=panda
```

Use this script to start the **motion generators** of the Franka Emika Panda
```
roslaunch panda_control panda_motion_generator_manager.launch
```

**Listen.py**:  
This script listens to the microphone and continuously publishes tokens with ROS.  
```
python3 src/ASR_tests/listen.py
```

**Classify.py**:  
Starts the classify server for the BT to call.  
```
rosrun ASR_tests classify.py
```

**Context_net.py**:  
Start the context server to extract information from the speech and the detections.  
```
rosrun ASR_tests context_net.py
```

Start the **BT**:
``` 
roslaunch nhr_bt nhr_bt.launch
```


Optional: **display_coordinates.py**:
Before running this script, make sure that the steps 1 & 2 of the camera are completed.
```
rosrun inference display_coordinates.py
```
Use this script to save 
- the detections and inferences in inf_data.csv file
- the translation and rotation data of the Panda robot at a particular position or point in time.
- the frame as image in imgs folder
