from pymodbus.client.sync import ModbusTcpClient
# from modbus_datatype import *
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse

#--------------------------------------------------------------Gilbert_Begin
import os,sys
from enum import IntEnum,Enum
from pymongo import MongoClient
import datetime, bson , time

#Bottle type == Btype
PETtype = Enum('',['P','COLOR','SOY','OIL','TRAY','CH','OTHER'])
#Bottle kind == Bkind
PETkind = Enum('',["PET","PP","PS","PLA","PC","PVC"])
#Log Err Kind == LEkind
LEkind = Enum('',['RobotArm','VisionSys','ConveySys','ControSys'])
#Status type ==Stype
Stype = Enum("",['good','bad'])

def switch(var, x=None):
    return {
        '.P':           lambda x: 'P',
        '.COLOR':       lambda x: 'COLOR',
        '.SOY':         lambda x: 'SOY',
        '.OIL':         lambda x: 'OIL',
        '.TRAY':        lambda x: 'TRAY',
        '.CH':          lambda x: 'CH',
        '.OTHER':       lambda x: 'OTHER',
        '.PET':         lambda x: 'PET',
        '.PP':          lambda x: 'PP',
        '.PS':          lambda x: 'PS',
        '.PLA':         lambda x: 'PLA',
        '.PC':          lambda x: 'PC',
        '.PVC':         lambda x: 'PVC',
        '.RobotArm':    lambda x: 'RobotArm',
        '.VisionSys':   lambda x: 'VisionSys',
        '.ConveySys':   lambda x: 'ConveySys',
        '.ControSys':   lambda x: 'ControSys',
        '.good':        lambda x: 'good',
        '.bad':         lambda x: 'bad',
    }[str(var)](x)

settings = {
    "ip": 'localhost',  # ip:127.0.0.1
    "port": 27017,  # port
    "db_name": "mongoDBrobot4",  # database-name
    "set_name": "robot1logdb4"  # collection-name
}

class MongoLogDBmodel(object):
        Content = ""
        Category = ""
        Status = ""
        Datetimetag = ""
        Timestamp = ""

        def __init__(self,v1,v2,v3):
            self.Datetimetag = bson.Int64(int(datetime.datetime.now().timestamp()*1000))
            self.Timestamp = datetime.datetime.now()
            self.Content = v1
            self.Category = v2
            self.Status = v3

        def set(self,v1,v2,v3):
            self.Datetimetag = bson.Int64(int(datetime.datetime.now().timestamp()*1000))
            self.Timestamp = datetime.datetime.now()
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


    #mongoDB c-r-u-d
    def create(self, model_dic):
        print("insert...1")
        self.my_set.insert_one(model_dic)

    def createdb(self, status, content, logKind):
        print("insert...2")
        log = MongoLogDBmodel(str(content),switch(logKind),switch(status))
        self.my_set.insert_one(log.get())

    def update(self, model_dic, newdic):
        print("update...")
        self.my_set.update(model_dic, newdic)

    def delete(self, model_dic):
        print("delete...")
        self.my_set.remove(model_dic)

    def dbread(self, model_dic):
        print("find...")
        data = self.my_set.find(model_dic)
        for result in data:
            print(result["Category"], result["Status"])

    def dbreadall(self):
        print("list all...\n")
        datas = self.my_set.find()
        for data in datas:
                print("\n-------------------\n", data.items() )
                # for k,v in data.items():
                #     print(k," : ",v)


def check_UsbCamConnection(config, pipeline):
    global mongo,old_timetag
    profile = ""
    try:
        profile = pipeline.start(config)
        now_timetag = int(datetime.datetime.now().timestamp()*1000)-1000
        if(now_timetag>old_timetag):
            Msg = "------Alive, USB Cam connected!-------"
            mongo.createdb(Stype.good, Msg, LEkind.VisionSys)
            print("\n{}\n".format(Msg))
            old_timetag = int(datetime.datetime.now().timestamp()*1000)
    except Exception as e:
            Msg = "------Err, USB Cam doesn't connected!-------"
            mongo.createdb(Stype.bad, Msg, LEkind.VisionSys)
            print("\n{}:{}\n".format(Msg,e))
            sys.exit()
    return profile

def check_CamStatus(pipeline):
    global mongo,old_timetag
    frames = ""
    try:
        frames = pipeline.wait_for_frames()
        now_timetag = int(datetime.datetime.now().timestamp()*1000)-1000
        if(now_timetag>old_timetag):
            Msg = "------Alive, USB Cam connected!-------"
            mongo.createdb(Stype.good, Msg, LEkind.VisionSys)
            print("\n{}\n".format(Msg))
            old_timetag = int(datetime.datetime.now().timestamp()*1000)
    except Exception as e:
            Msg = "------Err, USB Cam is suddenly gone!-------"
            mongo.createdb(Stype.bad, Msg, LEkind.VisionSys)
            print("\n{}:{}\n".format(Msg,e))
            sys.exit()
    return frames

def check_CamFrame():
    global mongo
    Msg = "------Err, Color / Depth Frame is gone!-------"
    mongo.createdb(Stype.bad, Msg, LEkind.VisionSys)
    print("\n{}\n".format(Msg))

#--------------------------------------------------------------Gilbert_Final
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
    global WIDTH, HEIGHT

    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)

    pipeline = rs.pipeline()
    # --------------------------------------------------------------Gilbert_Begin
    profile = check_UsbCamConnection(config, pipeline)
    # --------------------------------------------------------------Gilbert_Finish
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale

def load_detector(classes_name, model_name):
    '''
        Initializes variables and YOLO net
        Outputs:
            net     -> CNN model
            ln      -> Layer names
            classes -> Classes ids
            colors  -> Colors ids
    '''
    classes = open(r'D:\resources\models\{}.txt'.format(classes_name)).read().strip().split('\n')
    # classes = open((r'models/clases.txt')).read().strip().split('\n')

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    net = cv.dnn.readNetFromDarknet(r'D:\resources\models\{}.cfg'.format(model_name),
                                    r'D:\resources\models\{}.weights'.format(model_name))
    # net = cv.dnn.readNetFromDarknet(r'models/yolov3.cfg', r'models/yolov3.weights')
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
    global CONF
    CONF = x/100

def trackbar_nsm(x):
    '''
        Dynamically set non-maximum suppresion threshold
        Inputs:
            x -> Trackbar position
    '''
    global NMS
    NMS = x/100

def trackbar_simulate(x):
    '''
        Dynamically activate/deactivate simulation
        Inputs:
            x -> Trackbar position
    '''
    global IS_SIMULATING
    IS_SIMULATING = bool(x)

def trackbar_maxdepth(x):
    '''
        Dynamically changes the maximum depth
    '''
    global MAX_CLIPPING_DIST
    # MAX_CLIPPING_DIST = 1000*(4/100)*x
    MAX_CLIPPING_DIST = x

def get_images(pipeline, align, depth_scale):
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

    if IS_FILTERING:
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

    colorImg = np.asanyarray(aligned_color_frame.get_data())
    depthImg = np.asanyarray(aligned_depth_frame.get_data())

    depthImg = depthImg * depth_scale * 1000
    depthImg = np.where((depthImg > MAX_CLIPPING_DIST) | (depthImg <= MIN_CLIPPING_DIST), 0, depthImg)     

    return colorImg, depthImg, int(sensorTime)


def process_images(img, imgD, net, ln, classes, colors):
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
    global CONF, NMS, WIDTH, HEIGHT, MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, SELECTED_CLASSES

    imgC = img.copy()
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False) # yolo4-tiny
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False) # yolov3_spp
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 352), swapRB=True, crop=False) # yolo4-custom
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False) # yolov3_spp
    # blob = cv.dnn.blobFromImage(img, 1 / 255.0, (320, 320), swapRB=True, crop=False) # yolov3_spp
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 320), swapRB=True, crop=False) # yolov3_spp

    net.setInput(blob)
    outputs = np.vstack(net.forward(ln))
    height, width = img.shape[:2]

    boxes, confidences, classIDs, centers, depths = [], [], [], [], []
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > CONF:
            x, y, w, h = output[:4] * np.array([width, height, width, height])
            p0 = int(x - w // 2), int(y - h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            centers.append([int(x),int(y)])
            imgDCropped = imgD[p0[1]:p0[1]+int(h), p0[0]:p0[0]+int(w)]
            suma, npix = np.sum(imgDCropped), cv.countNonZero(imgDCropped)
            if npix != 0:
                depths.append(suma/npix)
            else:
                depths.append(0)

    cv.line(img, (MIN_WIDTH, 0), (MIN_WIDTH, HEIGHT), (255,0,0), 2)
    cv.line(img, (MAX_WIDTH, 0), (MAX_WIDTH, HEIGHT), (255,0,0), 2)
    cv.line(img, (0, MIN_HEIGHT), (WIDTH, MIN_HEIGHT), (255,0,0), 2)
    cv.line(img, (0, MAX_HEIGHT), (WIDTH, MAX_HEIGHT), (255,0,0), 2)

    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF, NMS)
    results, imgObjs = [], {}
    if len(indices) > 0:
        for j, i in enumerate(indices.flatten()):
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            cx, cy = centers[i][0], centers[i][1]
            cz = int(depths[i])
            cf, cs = int(100*confidences[i]), int(classIDs[i])

            # SECTION Drawing bounding boxes
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = r"{}_{}_{}_{}_{}".format(centers[i][0], centers[i][1], int(depths[i]), int(classIDs[i]), int(100*confidences[i]))
            labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(img, (x, y - labelSize[1]), (x + labelSize[0], y + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv.circle(img, tuple(centers[i]), 5, color, -1) 
            cv.circle(imgC, tuple(centers[i]), 5, color, -1)

            # SECTION Only objects in a certain area are used
            if MIN_WIDTH < cx < MAX_WIDTH:    
                if MIN_HEIGHT < cy < MAX_HEIGHT: 
                    if cs == 0 and SELECTED_CLASSES[0] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]
                    
                    if cs == 1 and SELECTED_CLASSES[1] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]

                    if cs == 2 and SELECTED_CLASSES[2] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]
                    
                    if cs == 3 and SELECTED_CLASSES[3] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]

                    if cs == 4 and SELECTED_CLASSES[4] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]

                    if cs == 5 and SELECTED_CLASSES[5] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]

                    if cs == 6 and SELECTED_CLASSES[6] == 1:
                        results.append([cx, cy, cz, cs, cf])
                        if x<0: x, w = 0, w-x
                        if y<0: y, h = 0, h-y
                        imgObjs['{}_{}'.format(j, text)] = imgC[y:y+h,x:x+w,:]
            
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
        message.append(1) # 0 cmd
        message.append(1) # 1 status
        message.append(numObjs) # 2 
        message.append(timeCaptureMsg.chunk[0]) # 3
        message.append(timeCaptureMsg.chunk[1]) # 4
        message.append(timeCaptureMsg.chunk[2]) # 5
        message.append(timeCaptureMsg.chunk[3]) # 6
        message.append(timeProcessMsg.chunk[0]) # 7
        message.append(timeProcessMsg.chunk[1]) # 8
        message.append(timeProcessMsg.chunk[2]) # 9
        message.append(timeProcessMsg.chunk[3]) # 10
        message.append(0) # 11 empty 
        message.append(0) # 12 empty 
        message.append(0) # 13 empty 
        message.append(0) # 14 empty  
        message.append(0) # 15 empty  
        message.append(0) # 16 empty  
        message.append(0) # 17 empty  
        message.append(0) # 18 empty  
        message.append(0) # 19 empty  

        for i in range(numObjs):
            message.append(results[i][0]) # x
            message.append(results[i][1]) # y
            message.append(results[i][2]) # z
            message.append(results[i][3]) # class
            message.append(results[i][4]) # CONF

        client.write_registers(0, message, unit=1)

    return None

def send_modbus(client, numObjs, timesSensor, timesProcess, results, img, imgObjs, imgsBuffer, imgsBufferData, imgsBufferName):
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
        message.append(1) # 0 cmd
        message.append(1) # 1 status
        message.append(numObjs) # 2 
        message.append(timeCaptureMsg.chunk[0]) # 3
        message.append(timeCaptureMsg.chunk[1]) # 4
        message.append(timeCaptureMsg.chunk[2]) # 5
        message.append(timeCaptureMsg.chunk[3]) # 6
        message.append(timeProcessMsg.chunk[0]) # 7
        message.append(timeProcessMsg.chunk[1]) # 8
        message.append(timeProcessMsg.chunk[2]) # 9
        message.append(timeProcessMsg.chunk[3]) # 10
        message.append(0) # 11 empty 
        message.append(0) # 12 empty 
        message.append(0) # 13 empty 
        message.append(0) # 14 empty  
        message.append(0) # 15 empty  
        message.append(0) # 16 empty  
        message.append(0) # 17 empty  
        message.append(0) # 18 empty  
        message.append(0) # 19 empty  

        for i in range(numObjs):
            message.append(results[i][0]) # x
            message.append(results[i][1]) # y
            message.append(results[i][2]) # z
            message.append(results[i][3]) # class
            message.append(results[i][4]) # CONF

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
    cmdServer.chunk[0] = reccmd.registers[0] # 11
    cmdServer.chunk[1] = reccmd.registers[1] # 12
    cmdServer.chunk[2] = reccmd.registers[2] # 13
    cmdServer.chunk[3] = reccmd.registers[3] # 14

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
            buffImgsOut = None

        if IS_SAVING and not IS_SIMULATING:
            if buffObjsOut:
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
    global MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, WIDTH_GAP, HEIGHT_GAP

    fakeResults = []
    for i in range(n):
        xpos = np.random.randint(MIN_WIDTH + WIDTH_GAP, MAX_WIDTH - WIDTH_GAP, 1)[0]
        ypos = np.random.randint(MIN_HEIGHT + HEIGHT_GAP, MAX_HEIGHT - HEIGHT_GAP, 1)[0]
        fakeResults.append([xpos, ypos, 850, 39, 90])

    return fakeResults

def run(args):
    global CONF, NMS, IS_SAVING, IS_SIMULATING, IS_COMMUNICATION_ACTIVE, IS_FILTERING, MAX_CLIPPING_DIST, MIN_CLIPPING_DIST
    global WIDTH, HEIGHT, MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, WIDTH_GAP, HEIGHT_GAP, BUFFER_SIZE, SELECTED_CLASSES
    # --------------------------------------------------------------Gilbert_Begin
    global mongo,old_timetag
    mongo = RobotLogModelServices()
    old_timetag = int(datetime.datetime.now().timestamp()*1000)
    # --------------------------------------------------------------Gilbert_Finish
    # SECTION Modifiable constants
    CONF = args.confidence
    NMS = args.nms
    
    IS_SAVING = args.save
    IS_SIMULATING = args.simulate
    IS_COMMUNICATION_ACTIVE = args.communicate
    IS_FILTERING = args.filter
    
    WIDTH, HEIGHT = args.width, args.height
    MIN_WIDTH, MAX_WIDTH = args.min_width, args.max_width
    MIN_HEIGHT, MAX_HEIGHT = args.min_height, args.max_height
    WIDTH_GAP, HEIGHT_GAP = args.width_gap, args.height_gap
    MAX_CLIPPING_DIST = args.max_clipping_dist
    MIN_CLIPPING_DIST = args.min_clipping_dist
    SELECTED_CLASSES = [int(d) for d in args.selected_classes]

    classes_name, model_name = args.classes_name, args.model_name

    # SECTION Non-modifiable constants
    BUFFER_SIZE = 20
    
    imgsBuffer, imgsBufferData, imgsBufferName = [], [], []
    fontFace = cv.FONT_HERSHEY_COMPLEX
    fps, contFrames, numObjs = 0, 0, 0
    numFrames = 30

    # SECTION Setting up interface
    net, ln, classes, colors = load_detector(classes_name, model_name)
    pipeline, align, depth_scale = load_camera()

    cv.namedWindow('window')
    cv.moveWindow('window', 0, 0)
    cv.createTrackbar('confidence', 'window', int(CONF*100), 100, trackbar_confidence)
    cv.createTrackbar('nms', 'window', int(NMS*100), 100, trackbar_nsm)
    # cv.createTrackbar('max_depth', 'window', int(MAX_CLIPPING_DIST), 4000, trackbar_maxdepth)
    cv.createTrackbar('simulate', 'window', int(IS_SIMULATING), 1, trackbar_simulate)
    
    fakeResults = simulate_points(5)

    if IS_COMMUNICATION_ACTIVE:
        client = ModbusTcpClient('127.0.0.1')

    start = time.time()

    while True:
        img, imgD, timesSensor = get_images(pipeline, align, depth_scale)
        results, imgB, imgObjs = process_images(img, imgD, net, ln, classes, colors) 
        timesProcess = int(time.time() * 1000)

        if IS_SIMULATING:
            results = fakeResults

        numObjs = len(results)
        if IS_COMMUNICATION_ACTIVE:
            check_response(client, imgsBuffer, imgsBufferData, imgsBufferName)
            imgsBuffer, imgsBufferData, imgsBufferName = update_buffers(numObjs, imgB, imgObjs, timesSensor, imgsBuffer, imgsBufferData, imgsBufferName)
            send_message(client, numObjs, timesSensor, timesProcess, results)
        
        # SECTION Drawing information and merging images 
        imgText = np.zeros(shape=(200, 2*WIDTH, 3), dtype=np.uint8)
        
        if numObjs == 0:
            cx, cy, cz, cc, cf = 0, 0, 0, 0, 0
        else:
            cx = results[0][0]
            cy = results[0][1]
            cz = results[0][2]
            cc = results[0][3]
            cf = results[0][4]

        text = r'FPS: {} - ODet: {}'.format(fps, numObjs)
        cv.putText(imgText, text, (10,30), fontFace, 1, (0,255,0))
        text = r'X: {} - Y: {} - Z: {} - CLS: {} - CF: {}'.format(cx, cy, cz, cc, cf)
        cv.putText(imgText, text, (10,70), fontFace, 1, (0,255,0))
        text = r'TSen: {}'.format(timesSensor)
        cv.putText(imgText, text, (10,110), fontFace, 1, (0,255,0))
        text = r'TPro: {}'.format(timesProcess)
        cv.putText(imgText, text, (10,150), fontFace, 1, (0,255,0))
        
        imgDC = cv.applyColorMap(cv.convertScaleAbs(imgD), cv.COLORMAP_JET)
        imgs = np.hstack((img, imgDC))
        imgF = np.vstack((imgs, imgText))
        cv.imshow('window',  imgF)

        contFrames += 1
        if contFrames == numFrames:
            end = time.time()
            seconds = end - start 
            fps = int(numFrames/seconds)
            start = end
            contFrames = 0

        key = cv.waitKey(1) 
        if key == 27:
            break

    if IS_COMMUNICATION_ACTIVE:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run yolo')
    
    parser.add_argument('--classes_name',      nargs='?', default='classes', type=str,      help='Classes file name') 
    parser.add_argument('--model_name',        nargs='?', default='yolo',    type=str,      help='Model file name') 
    parser.add_argument('--save',              nargs='?', default=True,      type=str2bool, help='Flag to save detected images') 
    parser.add_argument('--simulate',          nargs='?', default=False,     type=str2bool, help='Flag to start object detection simulation')
    parser.add_argument('--communicate',       nargs='?', default=True,      type=str2bool, help='Flag to connect to modbus server') 
    parser.add_argument('--filter',            nargs='?', default=True,      type=str2bool, help='Flag to filter depth data') 
    parser.add_argument('--confidence',        nargs='?', default=0.9,       type=float,    help='YOLO confidence value (0-1)') 
    parser.add_argument('--nms',               nargs='?', default=0.2,       type=float,    help='YOLO nms value (0-1)') 
     
    parser.add_argument('--width',             nargs='?', default=640,       type=int,      help='Width of the image') 
    parser.add_argument('--height',            nargs='?', default=480,       type=int,      help='Height of the image') 
    parser.add_argument('--min_width',         nargs='?', default=125,       type=int,      help='Minimum width of the working space') 
    parser.add_argument('--max_width',         nargs='?', default=500,       type=int,      help='Maximum width of the working space') 
    parser.add_argument('--min_height',        nargs='?', default=0,         type=int,      help='Minimum height of the working space') 
    parser.add_argument('--max_height',        nargs='?', default=480,       type=int,      help='Maximum height of the working space') 
    parser.add_argument('--width_gap',         nargs='?', default=0,         type=int,      help='Width gap to simulate points') 
    parser.add_argument('--height_gap',        nargs='?', default=100,       type=int,      help='Height gap to simulate points') 
    parser.add_argument('--max_clipping_dist', nargs='?', default=900,       type=int,      help='Max depth distance in mm') 
    parser.add_argument('--min_clipping_dist', nargs='?', default=100,       type=int,      help='Min depth distance in mm') 
    parser.add_argument('--selected_classes',  nargs='?', default='1111111', type=str,      help='Specific class to be detected') 

    print(parser.parse_args())
    run(parser.parse_args())
