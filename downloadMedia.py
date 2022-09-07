import requests
import tweepy
import json
from pymongo import MongoClient
import sys
import spacy
import pandas as pd
import itertools

from itertools import islice
import collections


collection = None

def mongo():
    global collection

    client = MongoClient('127.0.0.1',27017) #is assigned local port
    dbName = "TwitterDump" # set-up a MongoDatabase
    db = client[dbName]
    collName = 'colTest' # here we create a collection
    collection = db[collName] #  This is for the Collection  put in the DB

    #collection.delete_many({})

mongo()

tweetList = list()

tweetCollection = collection.find({},{'id' : 1,'media':1})
for tweet in tweetCollection:
    tweetList.append(tweet)

labels=['id','media']
tweet_frame=pd.DataFrame(tweetList,columns=labels)

def downloadImage():
    for media in tweet_frame['media']:
        if(media!=None):
            if("pbs" in media):
                r = requests.get(media, allow_redirects=True)
                if r.status_code == 404:
                    #print("file was deleted")
                    pass
                else:
                    print(media)
                    open('imagefile.jpg', 'wb').write(r.content)
                    break

def downloadVideo():
    for media in tweet_frame['media']:
        if(media!=None):
            if("pbs" not in media):
                r = requests.get(media, allow_redirects=True)
                if r.status_code == 404:
                    #print("file was deleted")
                    pass
                else:
                    print(media)
                    open('videofile.mp4', 'wb').write(r.content)
                    break
downloadImage()
downloadVideo()