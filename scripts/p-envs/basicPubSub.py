'''
/*
 * Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
 '''

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json
import pymongo
from pymongo import MongoClient
import pprint 
import pandas as pd
import random

# MongoDB set up
maxSevSelDelay = 100
client = MongoClient(serverSelectionTimeoutMS=maxSevSelDelay)

db = client.smart_city_metadata
iot_collection = db.aws_iot_events

AllowedActions = ['both', 'publish', 'subscribe']

# MongoDB Queries
def eventsWeightedSum():
    deltaDate = time.time() - 24*60*60
    events = iot_collection.aggregate([
        {'$match':{'timestamp':{'$gte':deltaDate}}},
        {'$match':{'event_id':{'$exists':1}}},
        {'$project':{'_id':0,'event_id':1}},
        {'$lookup':{'from':'eventsMappings','localField':'event_id','foreignField':'event_id','as':'em'}},
        {'$unwind': '$em'},
        {'$replaceRoot': {'newRoot':'$em'}},
        {'$group':{'_id':{},'total':{'$sum':'$weight'}}},
        {'$project':{'total':1,'_id':0}}
    ])
    magicSum = 0
    if events:
        magicSum = list(events)[0]['total']
    return magicSum

def lastEvent():
    deltaDate = time.time() - 24*60*60
    events = iot_collection.aggregate([
        {'$match':{'timestamp':{'$gte':deltaDate}}},
        {'$match':{'event_id':{'$exists':1}}},
    ])
    total = 0
    if events:
        total = list(events)[-1]
    return total

def defineEventsMappings():
    eventsMappings_list=[
        {'event_id':0, 'weight':45},
        {'event_id':1, 'weight':55},
        {'event_id':2, 'weight':5},
        {'event_id':3, 'weight':2},
        {'event_id':4, 'weight':12},
        {'event_id':5, 'weight':17},
        {'event_id':6, 'weight':20},
    ]
    a = db.eventsMappings.insert_many(eventsMappings_list)

#defineEventsMappings()
# Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                    help="Targeted client id")
parser.add_argument("-t", "--topic", action="store", dest="topic", default="events/", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")

args = parser.parse_args()
host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
port = args.port
useWebsocket = args.useWebsocket
clientId = args.clientId
topic = args.topic

if args.mode not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
    exit(2)

if args.useWebsocket and args.certificatePath and args.privateKeyPath:
    parser.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
    parser.error("Missing credentials for authentication.")
    exit(2)

# Port defaults
if args.useWebsocket and not args.port:  # When no port override for WebSocket, default to 443
    port = 443
if not args.useWebsocket and not args.port:  # When no port override for non-WebSocket, default to 8883
    port = 8883

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(1000)  # 1000 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec

# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()
"""if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
time.sleep(2)"""

# Publish to the same topic in a loop forever
loopCount = 1
while True:
    time.sleep(10)
    if args.mode == 'both' or args.mode == 'publish':
        """weightedSum = eventsWeightedSum()
        le = lastEvent()
        total = le['totalDetections']
        metric = 0
        if total != 0:
            metric = weightedSum / total"""
        message = {}
        message['location'] = "-31.406530, -64.189353"#"-31.385234, -64.229727" #le['location']
        message['intersection_id'] = 2
        message['metric'] = 7.2
        #message['timestamp'] = time.time() - 24*60*60 * 5
        messageJson = json.dumps(message)
        myAWSIoTMQTTClient.publish(topic, messageJson, 1)
        if args.mode == 'publish':
            print('Published topic %s: %s\n' % (topic, messageJson))