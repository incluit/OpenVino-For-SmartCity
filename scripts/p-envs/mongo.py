import pymongo
from pymongo import MongoClient
import pprint 

import pandas as pd

maxSevSelDelay = 100
client = MongoClient(serverSelectionTimeoutMS=maxSevSelDelay)

def testMongoServer():
    try:
        client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return 1
    else:
        return 0

db = client.smart_city_metadata

"""data1 = {   'Frame': 0,
            'Id': 0,
            'Vel_x': 0,
            'Vel_y': 1,
            'Vel': 2,
            'Acc_x': 3,
            'Acc_y': 4,
            'Acc': 9,
            'th_x': 5,
            'th_y': 3,
            'th': 1
        }"""

e = db.events

def events_list():
    events = e.aggregate([
        {'$match':{'Id':{'$exists':1}}},
        {'$project':{'_id':0,'Id':1,'frame':1,'event':1,'nearMiss':1, 'class': 1}},
    ])
    el = list(events)
    return el

c = db.collisions_data

def collision_list():
    collisions = c.aggregate([
        {'$match':{'Id':{'$exists':1}}},
        {'$match':{'$or':[{'ob1':{'$ne':0}},{'ob2':{'$ne':0}}]}},
        {'$project':{'_id':0,'frame':1,'ob1':1,'ob2':1}},
    ])
    cl = list(collisions)
    return cl

t = db.tracker_data

def tracker_list():
    trackers = t.aggregate([
        {'$match':{'Id':{'$exists':1}}},
        {'$project':{'_id':0,'ob1':0,'ob2':0,'nearMiss':0,'event':0}},
    ])
    tl = list(trackers)
    return tl

def drop_collections():
    e.drop()
    c.drop()
    t.drop()

#drop_collections()