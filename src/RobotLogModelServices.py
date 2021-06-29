# !/usr/bin/env python3
# -*- coding:utf-8 -*-
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
                for k,v in data.items():
                    print(k," : ",v)


if __name__ == "__main__":
    # main()
    dic = {"Content": "Error",
           "Category": "RobotArm100999",
           "Status": "good",
           "Datetimetag": bson.Int64(int(datetime.datetime.now().timestamp()*1000)),
           "Timestamp":  datetime.datetime.now()}

    log1 = MongoLogDBmodel("No Error1025",LEkind.RobotArm,Stype.good)

    mongo = RobotLogModelServices()
    # mongo.create(dic)
    mongo.create(dic)
    mongo.createdb(Stype.good,"No Err1036",LEkind.ControSys)
    mongo.dbread({"Status":"good"})
    print("\n\n\n--------------------Demo Finish---------------------\n\n\n")