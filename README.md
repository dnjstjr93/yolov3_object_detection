# yolov3_object_detection
Run YOLOv3 with Opencv on CPU.

## Preparations
1. Create Virtual environment
```
python -m venv yolov3_object_detection
cd yolov3_object_detection
.\Scripts\activate
# check activated virtual environment
# (yolov3_object_detection) ~\yolov3_object_detection> 
cd ..
```
2. Install requirements
```
pip install -r requirements.txt
```
or
```
python -m pip install -r requirements.txt
```
3. Download YOLOv3 weight in [here](https://pjreddie.com/media/files/yolov3.weights)

## Run
1. Use image
```
python yolo_object_detection.py 
```
