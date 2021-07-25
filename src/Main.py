# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
from pymodbus.client.sync import ModbusTcpClient
# from modbus_datatype import *
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import os
import matplotlib.pyplot as plt
import argparse
import threading
# --------------------------------------------------------------Gilbert_Begin
import os, sys
from enum import IntEnum, Enum
from pymongo import MongoClient
import datetime, bson, time , threading

# Bottle type == Btype
PETtype = Enum('', ['P', 'COLOR', 'SOY', 'OIL', 'TRAY', 'CH', 'OTHER'])
# Bottle kind == Bkind
PETkind = Enum('', ["PET", "PP", "PS", "PLA", "PC", "PVC"])
# Log Err Kind == LEkind
LEkind = Enum('', ['RobotArm', 'VisionSys', 'ConveySys', 'ControSys'])
# Status type ==Stype
Stype = Enum("", ['good', 'bad'])

def switch(var, x=None):
    return {
        '.P': lambda x: 'P',
        '.COLOR': lambda x: 'COLOR',
        '.SOY': lambda x: 'SOY',
        '.OIL': lambda x: 'OIL',
        '.TRAY': lambda x: 'TRAY',
        '.CH': lambda x: 'CH',
        '.OTHER': lambda x: 'OTHER',
        '.PET': lambda x: 'PET',
        '.PP': lambda x: 'PP',
        '.PS': lambda x: 'PS',
        '.PLA': lambda x: 'PLA',
        '.PC': lambda x: 'PC',
        '.PVC': lambda x: 'PVC',
        '.RobotArm': lambda x: 'RobotArm',
        '.VisionSys': lambda x: 'VisionSys',
        '.ConveySys': lambda x: 'ConveySys',
        '.ControSys': lambda x: 'ControSys',
        '.good': lambda x: 'good',
        '.bad': lambda x: 'bad',
    }[str(var)](x)

settings = {
    "ip": 'localhost',  # ip:127.0.0.1
    "port": 27017,  # port
    "db_name": "mongoDBrobot4",  # database-name
    "set_name": "robot1logdb4"  # collection-name
}

class MongoLogDBmodel(object):
    RobotID = ""
    Content = ""
    Category = ""
    Status = ""
    Datetimetag = ""
    Timestamp = ""

    def __init__(self,v0, v1, v2, v3):
        self.Datetimetag = bson.Int64(int(datetime.datetime.utcnow().timestamp() * 1000))
        self.Timestamp = datetime.datetime.utcnow()
        self.RobotID = v0
        self.Content = v1
        self.Category = v2
        self.Status = v3

    def set(self, v0,v1, v2, v3):
        self.Datetimetag = bson.Int64(int(datetime.datetime.utcnow().timestamp() * 1000))
        self.Timestamp = datetime.datetime.utcnow()
        self.RobotID = v0
        self.Content = v1
        self.Category = v2
        self.Status = v3

    def get(self):
        return self.__dict__




class RobotLogModelServices(object):
    def __init__(self):
        try:
            self.conn = MongoClient(settings["ip"], settings["port"])
        except Exception as e:
            print(e)
        self.db = self.conn[settings["db_name"]]
        self.my_set = self.db[settings["set_name"]]


    # mongoDB c-r-u-d
    def create(self, model_dic):
        print("insert...1")
        self.my_set.insert_one(model_dic)

    def createdb(self, robotID, status, content, logKind):
        print("insert...2")
        log = MongoLogDBmodel(robotID, str(content), switch(logKind), switch(status))
        self.my_set.insert_one(log.get())

    def update(self, model_dic, newdic):
        print("update...")
        self.my_set.update(model_dic, newdic)

    def delete(self, model_dic):
        print("delete...")
        self.my_set.remove(model_dic)

    def dbread(self):
        print("find...")
        data = self.my_set.find()
        for idx in range((data.count())):
            print(idx," : ",data[idx]["Category"]," : ",data[idx] ["Status"])

    def dbreadall(self):
        print("list all...\n")
        datas = self.my_set.find()
        for idx in range(datas.count()):
                print("\n[{}]----------------------------------------------".format(idx) )
                for k,v in datas[idx].items():
                    print(k," : ",v)

def async_wrDB(signal):
    global old_timetag,mongo,lock

    while True:
        signal.wait()
        lock.acquire()
        now_timetag = int(datetime.datetime.utcnow().timestamp() * 1000) - 1000

        if (now_timetag > old_timetag):
                Msg = "------Alive, USB Cam connected!-------"
                mongo.createdb("robot000001", Stype.good, Msg, LEkind.VisionSys)
                # print("\n{}\n".format(Msg))
                old_timetag = int(datetime.datetime.utcnow().timestamp() * 1000)
        lock.release()
        signal.clear()

def async_wrDB1():
    global old_timetag,mongo

    now_timetag = int(datetime.datetime.utcnow().timestamp() * 1000) - 1000

    if (now_timetag > old_timetag):
                Msg = "------Alive, USB Cam async_wrDB1 connected!-------"
                mongo.createdb("robot000001", Stype.good, Msg, LEkind.VisionSys)
                print("\n{}\n".format(Msg))
                old_timetag = int(datetime.datetime.utcnow().timestamp() * 1000)

def check_UsbCamConnection(config, pipeline):
    global mongo, old_timetag,asyncDB_event,signal
    profile = ""
    try:
        profile = pipeline.start(config)
        Msg = "------Alive, USB Cam check_UsbCamConnection connected!-------"
        # print("\n{}\n".format(Msg))
        signal.set()
        # async_wrDB1()
    except Exception as e:
        Msg = "------Err, USB Cam doesn't connected!-------"
        mongo.createdb("robot000001", Stype.bad, Msg, LEkind.VisionSys)
        print("\n{}:{}\n".format(Msg, e))
        sys.exit()
    return profile

def check_CamStatus(pipeline):
    global mongo, old_timetag,asyncDB_event,signal
    frames = ""
    try:
        frames = pipeline.wait_for_frames()
        Msg = "------Alive, USB Cam check_CamStatus connected!-------"
        # print("\n{}\n".format(Msg))
        signal.set()
        # async_wrDB1()
    except Exception as e:
        Msg = "------Err, USB Cam is suddenly gone!-------"
        mongo.createdb("robot000001", Stype.bad, Msg, LEkind.VisionSys)
        print("\n{}:{}\n".format(Msg, e))
        sys.exit()
    return frames

def check_CamFrame():
    global mongo
    Msg = "------Err, Color / Depth Frame is gone!-------"
    mongo.createdb("robot000001", Stype.bad, Msg, LEkind.VisionSys)
    print("\n{}\n".format(Msg))

# --------------------------------------------------------------Gilbert_Final
def str2bool(v):
    '''
        Convert parsed argument string to boolean
        Input:
            v -> Parsed argument
    '''
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_camera():
    '''
        Initialize variables and camera settings
        Outputs:
            pipeline    -> Pipeline of the RS camera
            align       -> Object to align RGB and DETPH data
            depth_scale -> Conversion factor for depth data
    '''
    global dbThread,signal
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)

    pipeline = rs.pipeline()
    # --------------------------------------------------------------Gilbert_Begin
    dbThread.start()
    profile = check_UsbCamConnection(config, pipeline)
    # --------------------------------------------------------------Gilbert_Finish
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale

def load_detector():
    '''
        Initializes variables and YOLO net
        Outputs:
            net     -> CNN model
            ln      -> Layer names
            classes -> Classes ids
            colors  -> Colors ids
    '''
    global conf, nms

    conf = 0.5
    nms = 0.2

    # classes = open(r'D:\resources\models\classes.txt').read().strip().split('\n')
    # classes = open(r'D:\resources\models\clases.txt').read().strip().split('\n')
    #
    # np.random.seed(42)
    # colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    #
    # # net = cv.dnn.readNetFromDarknet(r'D:\resources\models\yolov4-tiny.cfg', r'D:\resources\models\yolov4-tiny.weights')
    # net = cv.dnn.readNetFromDarknet(r'D:\resources\models\yolov3_spp.cfg', r'D:\resources\models\yolov3_spp.weights')
    # # net = cv.dnn.readNetFromDarknet(r'D:\resources\models\yolov4-custom.cfg', r'D:\resources\models\yolov4-custom.weights')

    classes = open(r'../models/clases.txt').read().strip().split('\n')  # ubuntu
    # classes = open(r'/home/gilbert3/darknet/data/coco.names').read().strip().split('\n')
    # classes = open(r'/media/gilbert3/mx500_1/Downloads/yuching7petv4/obj.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # net = cv.dnn.readNetFromDarknet(r'models\yolov4-tiny.cfg', r'models\yolov4-tiny.weights')
    # net = cv.dnn.readNetFromDarknet(r'models\yolov3.cfg', r'models\yolov3.weights')#win10
    net = cv.dnn.readNetFromDarknet(r'../models/yolov3-org.cfg', r'../models/yolov3-org.weights')  # ubuntu
    # net = cv.dnn.readNetFromDarknet(r'models/yolov3_spp.cfg', r'models/yolov3_spp.weights')# ubuntu
    # net = cv.dnn.readNetFromDarknet(r'models/yolov4-custom.cfg', r'models/yolov4-custom_best.weights')

    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln, classes, colors

def trackbar_confidence(x):
    '''
        Dynamically set YOLO confidence factor
        Inputs:
            x -> Trackbar position
    '''
    global conf
    conf = x / 100

def trackbar_nsm(x):
    '''
        Dynamically set non-maximum suppresion threshold
        Inputs:
            x -> Trackbar position
    '''
    global nms
    nms = x / 100

def trackbar_depthsection(xx):
    '''
        Dynamically set depth section width ,length threshold
        Inputs:
            x -> Trackbar position
    '''
    global depthWL
    if xx > 0 and xx < 4:
        depthWL = xx
        # print ("---------------- depthWL-------------------\n", depthWL,"\n---------------- depthWL-------------------",)


def get_images(pipeline, align):
    '''
        Acquires rgb and depth images from camera
        Inputs:
            pipeline   -> Pipeline of the RS camera
            align      -> Object to align RGB and DETPH data
        Outputs:
            colorImg   -> RGB image
            depthImg   -> Gray image
            sensorTime -> Time when the frame was captured
    '''
    global IS_FILTERING, MAX_CLIPPING_DIST, MIN_CLIPPING_DIST
    # --------------------------------------------------------------Gilbert_Begin
    frames = check_CamStatus(pipeline)
    # --------------------------------------------------------------Gilbert_Finish
    sensorTime = frames.get_color_frame().get_timestamp()

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not aligned_color_frame:
        # --------------------------------------------------------------Gilbert_Begin
        check_CamFrame()
        # --------------------------------------------------------------Gilbert_Finish
        return None, None, sensorTime

    colorImg = np.asanyarray(aligned_color_frame.get_data())
    depthImg = np.asanyarray(aligned_depth_frame.get_data())

    return colorImg, depthImg, int(sensorTime)


_noise_limit = 70  # 5 --> 5mm
_diff = 5  # interval --> 5mm
# 5* 240 = 1200mm == 1.2 m max depth for convayer
_bin = [i * _diff for i in range(1, 240)]


# Really basic way of separating foreground and background.
def filter_background(roi, max_depth=970):
    # Anything further than 1200mm, we consider it as background
    # Anything less than 5mm is consider noise
    ret_val = np.ma.masked_greater(roi, max_depth)
    ret_val = np.ma.masked_less(ret_val, _noise_limit)
    unique, counts = np.unique(ret_val.mask, return_counts=True)
    _dict = dict(zip(unique, counts))

    # plt.hist(ret_val.mask.flatten() , bins=_bin, density=True)
    # True is mask area, False is non-mask area
    if False in _dict:
        return ret_val, _dict[False]
    else:
        return ret_val, 0


def dynamic_background(roi):
    # Anything less than 5mm is sure noise
    # roi = np.ma.masked_less(roi, 100)
    roi_1d = roi.flatten()
    hist, bins = np.histogram(roi_1d, bins=_bin, density=True)
    max_bin = hist.argmax() + 1

    # plt.hist(roi_1d, bins=_bin, density=True)
    # plt.title('dynamic_background')
    # plt.show()

    return filter_background(roi, max_bin * _diff)


def zAvg(depthWL, imgD, x, y, depth_scale, max_depth):
    zTotal = []
    zSum = 0
    zCount = 0
    loop = (depthWL - 1)
    for ii in range(-loop, loop + 1):
        for jj in range(-loop, loop + 1):
            zTotal.append(imgD[int(y + jj), int(x + ii)] * depth_scale * 1000)
    for idx in range(len(zTotal)):
        if zTotal[idx] > 0 and zTotal[idx] < max_depth:
            zSum += zTotal[idx]
            zCount += 1
    if zCount > 0:
        return int(zSum / zCount), zCount
    else:
        return int(imgD[int(y), int(x)] * depth_scale * 1000), 0


def process_images(img, imgD, net, ln, classes, colors, depth_scale):
    """
        Process a image throught the YOLO net and runs the postprocess
        Inputs:
            img         -> Clean image to be processed
            imgD        -> Depth information
            net         -> CNN model
            ln          -> Layer names
            classes     -> List of classes for the trained model
            colors      -> List of colors
            depth_scale -> Conversion factor for depth data
        Outputs:
            results     -> Results of the CNN processing
            imgB        -> Copy of the drawn picture
            imgObjs     -> Images of the detected objects
    """
    global conf, nms, zReal1, depthWL

    imgC = img.copy()
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False) # yolo4-tiny
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False) # yolov3_spp
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 352), swapRB=True, crop=False) # yolo4-custom
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False) # yolov3_spp
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (320, 320), swapRB=True, crop=False)  # yolov3_spp

    net.setInput(blob)
    outputs = np.vstack(net.forward(ln))
    height, width = img.shape[:2]

    boxes, confidences, classIDs, centers, depths = [], [], [], [], []
    zTotal = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > conf:
            if classID == 0:
                x, y, w, h = output[:4] * np.array([width, height, width, height])
                p0 = int(x - w // 2), int(y - h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centers.append([int(x), int(y)])
                # depths.append(imgD[int(y), int(x)] * depth_scale * 1000)
                if depthWL == 1:
                    zReal1 = int(imgD[int(y), int(x)] * depth_scale * 1000)
                    zCount = 1
                else:
                    zReal1, zCount = zAvg(depthWL, imgD, x, y, depth_scale, 970)
                depths.append(zReal1)
                zzz = int(imgD[int(y), int(x)] * depth_scale * 1000)
                print("\n---------------- z-one: ", zzz, " : zAvg :", zReal1, " : size:", (zCount),
                      "\n---------------- depthWL-------------------\n", )
                # z = imgD[int(y)-int(h/2):int(x)-int(w/2),int(y)+int(h/2):int(x)+int(w/2)] #Objection(Rcet(x,x+w,y,y+h),classes[classIDs[i]])
                # filtered_depth, _size = dynamic_background(z)
                # #(10.0= 1cm, 1000.0mm = 1m, _size= Total_pixel_point)
                # zReal1 = filtered_depth.sum() / _size
    cv.line(img, (MIN_LEFT_VALUE, 0), (MIN_LEFT_VALUE, 480), (255, 0, 0), 2)
    cv.line(img, (MAX_LEFT_VALUE, 0), (MAX_LEFT_VALUE, 480), (255, 0, 0), 2)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, nms)
    results, imgObjs = [], {}
    if len(indices) > 0:
        for j, i in enumerate(indices.flatten()):
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            cx, cy = centers[i][0], centers[i][1]
            cz = int(depths[i])
            cf, cs = int(100 * confidences[i]), int(classIDs[i])

            # SECTION Drawing bounding boxes
            # if w*h < 50000:
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = r"{}_{}_{}_{}_{}".format(centers[i][0], centers[i][1], int(zReal1), int(classIDs[i]),
                                            int(100 * confidences[i]))
            labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(img, (x, y - labelSize[1]), (x + labelSize[0], y + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.circle(img, tuple(centers[i]), 5, color, -1)
            cv.circle(imgC, tuple(centers[i]), 5, color, -1)

            # SECTION Only objects in a certain area are used
            if MIN_LEFT_VALUE < cx < MAX_LEFT_VALUE:
                results.append([cx, cy, cz, cs, cf])

                if x < 0: x, w = 0, w - x
                if y < 0: y, h = 0, h - y
                imgObjs['{}_{}'.format(j, text)] = imgC[y:y + h, x:x + w, :]

    imgB = img.copy()

    return results, imgB, imgObjs

def update_buffers(numObjs, img, imgObjs, timesSensor, imgsBuffer, imgsBufferData, imgsBufferName):
    '''
        Update buffers
        Inputs:
            numObjs        -> Number of detected objects
            img            -> Image with detected objects
            imgObjs        -> Images of the detected objects
            timesSensor    -> Time when the frame was captured
            imgsBuffer     -> Buffer containing the processed images
            imgBufferData  -> Buffer containing the images of the objects
            imgsBufferName -> Buffer containing the timestamp of the processed images
        Outputs:
            imgsBuffer     -> Buffer containing the processed images
            imgBufferData  -> Buffer containing the images of the objects
            imgsBufferName -> Buffer containing the timestamp of the processed images
    '''
    global BUFFER_SIZE

    if 0 <= numObjs <= 10:
        if len(imgsBufferName) == BUFFER_SIZE:
            imgsBuffer.pop()
            imgsBufferData.pop()
            imgsBufferName.pop()

        imgsBuffer.append(img)
        imgsBufferData.append(imgObjs)
        imgsBufferName.append(str(int(timesSensor)))

    return imgsBuffer, imgsBufferData, imgsBufferName

def send_message(client, numObjs, timesSensor, timesProcess, results):
    '''
        Send data via modbus
        Inputs:
            client         -> Modbus client
            numObjs        -> Number of detected objects
            timesSensor    -> Time when the frame was captured
            timesProcess   -> Time when the YOLO process is finished
            results        -> Results of the CNN processing
    '''
    if 0 <= numObjs <= 10:
        timeCaptureMsg = utype()
        timeCaptureMsg.data = timesSensor

        timeProcessMsg = utype()
        timeProcessMsg.data = timesProcess

        message = []
        message.append(1)  # 0 cmd
        message.append(1)  # 1 status
        message.append(numObjs)  # 2
        message.append(timeCaptureMsg.chunk[0])  # 3
        message.append(timeCaptureMsg.chunk[1])  # 4
        message.append(timeCaptureMsg.chunk[2])  # 5
        message.append(timeCaptureMsg.chunk[3])  # 6
        message.append(timeProcessMsg.chunk[0])  # 7
        message.append(timeProcessMsg.chunk[1])  # 8
        message.append(timeProcessMsg.chunk[2])  # 9
        message.append(timeProcessMsg.chunk[3])  # 10
        message.append(0)  # 11 empty
        message.append(0)  # 12 empty
        message.append(0)  # 13 empty
        message.append(0)  # 14 empty
        message.append(0)  # 15 empty
        message.append(0)  # 16 empty
        message.append(0)  # 17 empty
        message.append(0)  # 18 empty
        message.append(0)  # 19 empty

        for i in range(numObjs):
            message.append(results[i][0])  # x
            message.append(results[i][1])  # y
            message.append(results[i][2])  # z
            message.append(results[i][3])  # class
            message.append(results[i][4])  # conf

        client.write_registers(0, message, unit=1)

    return None

def send_modbus(client, numObjs, timesSensor, timesProcess, results, img, imgObjs, imgsBuffer, imgsBufferData,
                imgsBufferName):
    '''
        Send data via modbus and fill the buffers out
        Inputs:
            client         -> Modbus client
            numObjs        -> Number of detected objects
            timesSensor    -> Time when the frame was captured
            timesProcess   -> Time when the YOLO process is finished
            results        -> Results of the CNN processing
            img            -> Image with detected objects
            imgObjs        -> Images of the detected objects
            imgsBuffer     -> Buffer containing the processed images
            imgBufferData  -> Buffer containing the images of the objects
            imgsBufferName -> Buffer containing the timestamp of the processed images
    '''
    global BUFFER_SIZE, IS_SAVING

    if 0 <= numObjs <= 10:
        if len(imgsBufferName) == BUFFER_SIZE:
            imgsBuffer.pop()
            imgsBufferData.pop()
            imgsBufferName.pop()

        imgsBuffer.append(img)
        imgsBufferData.append(imgObjs)
        imgsBufferName.append(str(int(timesSensor)))

        timeCaptureMsg = utype()
        timeCaptureMsg.data = timesSensor

        timeProcessMsg = utype()
        timeProcessMsg.data = timesProcess

        message = []
        message.append(1)  # 0 cmd
        message.append(1)  # 1 status
        message.append(numObjs)  # 2
        message.append(timeCaptureMsg.chunk[0])  # 3
        message.append(timeCaptureMsg.chunk[1])  # 4
        message.append(timeCaptureMsg.chunk[2])  # 5
        message.append(timeCaptureMsg.chunk[3])  # 6
        message.append(timeProcessMsg.chunk[0])  # 7
        message.append(timeProcessMsg.chunk[1])  # 8
        message.append(timeProcessMsg.chunk[2])  # 9
        message.append(timeProcessMsg.chunk[3])  # 10
        message.append(0)  # 11 empty
        message.append(0)  # 12 empty
        message.append(0)  # 13 empty
        message.append(0)  # 14 empty
        message.append(0)  # 15 empty
        message.append(0)  # 16 empty
        message.append(0)  # 17 empty
        message.append(0)  # 18 empty
        message.append(0)  # 19 empty

        for i in range(numObjs):
            message.append(results[i][0])  # x
            message.append(results[i][1])  # y
            message.append(results[i][2])  # z
            message.append(results[i][3])  # class
            message.append(results[i][4])  # conf

        client.write_registers(0, message, unit=1)

    return imgsBuffer, imgsBufferData, imgsBufferName

def check_response(client, imgsBuffer, imgsBufferData, imgsBufferName):
    '''
        Save data according to the information retrieved by the motion comp
        Inputs:
            client         -> Modbus client
            imgsBuffer     -> Buffer containing the processed images
            imgBufferData  -> Buffer containing the images of the objects
            imgsBufferName -> Buffer containing the timestamp of the processed images
        Outputs:
    '''
    global IS_SAVING, IS_SIMULATING

    reccmd = client.read_holding_registers(11, 4, unit=1)

    cmdServer = utype()
    cmdServer.chunk[0] = reccmd.registers[0]  # 11
    cmdServer.chunk[1] = reccmd.registers[1]  # 12
    cmdServer.chunk[2] = reccmd.registers[2]  # 13
    cmdServer.chunk[3] = reccmd.registers[3]  # 14

    if reccmd.registers[0] != 0:
        client.write_register(11, 0)
        client.write_register(12, 0)
        client.write_register(13, 0)
        client.write_register(14, 0)

        timesReturn = str(int(cmdServer.data))

        if timesReturn in imgsBufferName:
            ind = imgsBufferName.index(timesReturn)
            buffObjsOut = imgsBufferData[ind]
            buffImgsOut = imgsBuffer[ind]
        else:
            buffObjsOut = None
            print('Time stamp not available')

        # TODO Check if this can be implemented in another thread
        if buffObjsOut and IS_SAVING and not IS_SIMULATING:
            folderPath = r'D:\resources\imgs\{}'.format(timesReturn)
            if not os.path.exists(folderPath):
                os.mkdir(folderPath)

            cv.imwrite(r'D:\resources\imgs\{}\{}.png'.format(timesReturn, timesReturn), buffImgsOut)
            for key in buffObjsOut.keys():
                cv.imwrite(r'D:\resources\imgs\{}\{}.jpg'.format(timesReturn, key), buffObjsOut[key])

    return None

def simulate_points(n):
    '''
        Simulate n number of points (only x and y coords)
        Inputs:
            n           -> number of points required
        Outputs:
            fakeResults -> simulated results
    '''
    global MIN_LEFT_VALUE, MAX_LEFT_VALUE, HEIGHT

    fakeResults = []
    for i in range(n):
        xpos = np.random.randint(MIN_LEFT_VALUE, MAX_LEFT_VALUE, 1)[0]
        ypos = np.random.randint(100, HEIGHT - 100, 1)[0]
        fakeResults.append([xpos, ypos, 850, 39, 90])

    return fakeResults

def run():
    global conf, nms, zReal1, depthWL
    global MAX_LEFT_VALUE, MIN_LEFT_VALUE, HEIGHT, WIDTH, BUFFER_SIZE
    global IS_SAVING, IS_SIMULATING, IS_COMMUNICATION_ACTIVE
    # --------------------------------------------------------------Gilbert_Begin
    global mongo, old_timetag
    global dbThread,lock,signal
    signal = threading.Event()
    lock = threading.Lock()
    dbThread = threading.Thread(target = async_wrDB , args=(signal,))
    mongo = RobotLogModelServices()
    old_timetag = int(datetime.datetime.utcnow().timestamp() * 1000)
    # --------------------------------------------------------------Gilbert_Finish
    # SECTION Constants
    IS_SAVING = True
    IS_SIMULATING = False
    IS_COMMUNICATION_ACTIVE = True

    MAX_LEFT_VALUE = 500
    MIN_LEFT_VALUE = 125
    BUFFER_SIZE = 20
    WIDTH, HEIGHT = 640, 480
    depthWL = 2

    net, ln, classes, colors = load_detector()
    pipeline, align, depth_scale = load_camera()

    cv.namedWindow('window')
    cv.moveWindow('window', 0, 0)
    cv.createTrackbar('conf', 'window', int(conf * 100), 100, trackbar_confidence)
    cv.createTrackbar('nms', 'window', int(nms * 100), 100, trackbar_nsm)
    cv.createTrackbar('depWL', 'window', int(depthWL), 3, trackbar_depthsection)

    fontFace = cv.FONT_HERSHEY_COMPLEX
    fps, contFrames, numObjs = 0, 0, 0
    numFrames = 30

    imgsBuffer, imgsBufferData, imgsBufferName = [], [], []

    if IS_COMMUNICATION_ACTIVE:
        client = ModbusTcpClient('127.0.0.1')

    if IS_SIMULATING:
        fakeResults = simulate_points(5)

    start = time.time()
    counter = 0
    zReal1 = 0
    rusult = []
    while True:
        img, imgD, timesSensor = get_images(pipeline, align)
        results, imgB, imgObjs = process_images(img, imgD, net, ln, classes, colors, depth_scale)
        timesProcess = int(time.time() * 1000)

        if IS_SIMULATING:
            results = fakeResults

        numObjs = len(results)
        # if IS_COMMUNICATION_ACTIVE:
        #     check_response(client, imgsBuffer, imgsBufferData, imgsBufferName)
        #     imgsBuffer, imgsBufferData, imgsBufferName = update_buffers(numObjs, imgB, imgObjs, timesSensor, imgsBuffer, imgsBufferData, imgsBufferName)
        #     send_message(client, numObjs, timesSensor, timesProcess, results)

        # SECTION Drawing information and merging images
        imgText = np.zeros(shape=(200, 2 * 640, 3), dtype=np.uint8)

        if numObjs == 0:
            cx, cy, cz, cc, cf = 0, 0, 0, 0, 0
        else:
            cx = results[0][0]
            cy = results[0][1]
            cz = results[0][2]
            cc = results[0][3]
            cf = results[0][4]

        counter += 1
        rusult.append(zReal1)

        # if counter > 999:
        #     x = np.linspace(1,1000,1000)
        #     y = np.asanyarray(rusult,dtype=float)
        #     plt.scatter(x,y,c='b')
        #     plt.plot(x,y,'-.r')
        #     plt.plot(x,y,'g')
        #     plt.show()
        #     time.sleep(1)
        #     break
        text = r'FPS: {} - ODet: {}'.format(fps, numObjs)
        cv.putText(imgText, text, (10, 30), fontFace, 1, (0, 255, 0))
        text = r'X: {} - Y: {} - Z: {} - CLS: {} - CF: {}'.format(cx, cy, cz, cc, cf)
        cv.putText(imgText, text, (10, 70), fontFace, 1, (0, 255, 0))
        text = r'TSen: {}'.format(timesSensor)
        cv.putText(imgText, text, (10, 110), fontFace, 1, (0, 255, 0))
        text = r'TPro: {}'.format(timesProcess)
        cv.putText(imgText, text, (10, 150), fontFace, 1, (0, 255, 0))

        imgDC = cv.applyColorMap(cv.convertScaleAbs(imgD, alpha=0.03), cv.COLORMAP_JET)
        imgs = np.hstack((img, imgDC))
        imgF = np.vstack((imgs, imgText))
        cv.imshow('window', imgF)

        contFrames += 1
        if contFrames == numFrames:
            end = time.time()
            seconds = end - start
            fps = int(numFrames / seconds)
            start = end
            contFrames = 0

        key = cv.waitKey(1)
        if key == 27:
            break

    if IS_COMMUNICATION_ACTIVE:
        client.close()

if __name__ == "__main__":
    run()
