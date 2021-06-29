
# PET recognized by YOLOv3/v4 with CV2 & MongoDB
This is PET Object detection at Win10

## return value
* return class: P, S, O, C, Ot, CH, T

* return % : class confident percentage

* return right/wrong camera status log at mongoDB

![img.png](img.png)

## 7 Classes model
* CLASSES = ["P", "S", "O", "C", "Ot","CH","T"]

# Environmental requirements
* Opencv 3.2 or laster
* MongoDB 4.0 or later

## Detail
See src/run_yolo_0507.py
