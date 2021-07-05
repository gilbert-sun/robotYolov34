
# PET recognized by YOLOv3/v4 with CV2 & MongoDB
This is PET Object detection at Win10

## return value
* return class: P, S, O, C, Ot, CH, T

* return % : class confident percentage

* return right/wrong camera status log at mongoDB

## Realsense Cam & Depth
#### ./Main.py
![img.png](img.png)

## MongoDB cmd 
![img_1.png](img_1.png)

## Play Test video
#### ./Run_yolo_video.py
![img_2.png](img_2.png)

## MongoDB chart of robot1logdb4/robot1db
#### http://localhost/mongodb-charts/ 
![img_3.png](img_3.png)

## MongoDB chart Docker container
#### https://segmentfault.com/a/1190000023502067
#### https://docs.atlas.mongodb.com/tutorial/manage-projects/#create-a-project
![img_4.png](img_4.png)

## 7 Classes model
* CLASSES = ["P", "S", "O", "C", "Ot","CH","T"]

# Environmental requirements
* Opencv 3.2 or laster
* MongoDB 4.0 or later

## Detail
See src/run_yolo_0507.py
