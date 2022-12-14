import tweepy
import json
from pymongo import MongoClient
from datetime import datetime
import time
import sys
import spacy
import pandas as pd
import itertools

from itertools import islice
import collections
#from NLP import NLP

import emoji
import re

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

# apikey = sokhJPKejEvqEiR9zY1LV5RSd
# apisecretkey = uN9QVQb6K5EgTAJOaKt6UyJ3F7OCmyPsL4fY1lA3hc3ZTcaJkL
# bearer token = AAAAAAAAAAAAAAAAAAAAABDXMwEAAAAABL2fXIJhodkuraLq6HzrjlHor3k%3Dsb0jtg6OebCDcyaT01vCycoKPgidSyxVcz9671hKxnydElOcGS
#  please put your credentials below - very important
consumer_key = "sokhJPKejEvqEiR9zY1LV5RSd"
consumer_secret ="uN9QVQb6K5EgTAJOaKt6UyJ3F7OCmyPsL4fY1lA3hc3ZTcaJkL"
access_token ="1352226025367887874-2wsA9770Yu31eu39cRi55n67JYRmZu"
access_token_secret ="hdQzIw7GF7vCM6GfkwlHv5m0Rg7IbxMbRvZuos8cLANPX"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret )
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
if (not api):
    print('Can\'t authenticate')
    print('failed cosumeer id ----------: ', consumer_key )
# set DB DETAILS


# this is to setup local Mongodb

#username = urllib.parse.quote_plus('myUserAdmin')
#password = urllib.parse.quote_plus('abc123')
#MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

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
def strip_emoji(text):
    #  copied from web - don't remeber the actual link
    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)
    return new_text

def cleanList(text):
    #  copied from web - don't remeber the actual link
    #remove emoji it works
    text = strip_emoji(text)
    text.encode("ascii", errors="ignore").decode()

    return text
    
spacy.load("en_core_web_sm")
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
  
def processTweets(tweet):
    #  this module is for cleaning text and also extracting relevant twitter feilds
    # initialise placeholders
    place_countrycode  =None
    place_name  =None
    place_country =None
    place_coordinates =None
    source =None
    exactcoord =None
    place=None
    
    # print(t)

    # Pull important data from the tweet to store in the database.
    try:
        created = tweet['created_at']
        tweet_id = tweet['id_str']  # The Tweet ID from Twitter in string format
        username = tweet['user']['screen_name']  # The username of the Tweet author
        # followers = t['user']['followers_count']  # The number of followers the Tweet author has
        text = tweet['text']  # The entire body of the Tweet
    except Exception as e:
        # if this happens, there is something wrong with JSON, so ignore this tweet
        print(e)
        return None

    try:
        # // deal with truncated
        if(tweet['truncated'] == True):
            text = tweet['extended_tweet']['full_text']
        elif(text.startswith('RT') == True):
            # print(' tweet starts with RT **********')
            # print(text)
            try:
                if( tweet['retweeted_status']['truncated'] == True):
                    # print("in .... tweet.retweeted_status.truncated == True ")
                    text = tweet['retweeted_status']['extended_tweet']['full_text']
                    # print(text)
                else:
                    text = tweet['retweeted_status']['full_text']

            except Exception as e:
                pass

    except Exception as e:
        print(e)
    # print(text)
    text = cleanList(text)
    # print(text)
    entities = tweet['entities']
    # print(entities)
    mentions =entities['user_mentions']
    mList = []

    for x in mentions:
        # print(x['screen_name'])
        mList.append(x['screen_name'])
    hashtags = entities['hashtags']  # Any hashtags used in the Tweet
    hList =[]
    for x in hashtags:
        # print(x['screen_name'])
        hList.append(x['text'])
        if hashtags == []:
            hashtags =''
        else:
            hashtags = str(hashtags).strip('[]')
    source = tweet['source']

    exactcoord = tweet['coordinates']
    coordinates = None
    if(exactcoord):
        coordinates = exactcoord['coordinates']
    geoenabled = tweet['user']['geo_enabled']
    location = tweet['user']['location']


    if ((geoenabled) and (text.startswith('RT') == False)):
        # print(tweet)
        # sys.exit() # (tweet['geo']):
        try:
            if(tweet['place']):
                # print(tweet['place'])
                place_name = tweet['place']['full_name']
                place_country = tweet['place']['country']
                place_countrycode   = tweet['place']['country_code']
                place_coordinates   = tweet['place']['bounding_box']['coordinates']
        except Exception as e:
            print(e)
            print('error from place details - maybe AttributeError: ... NoneType ... object has no attribute ..full_name ...')
    media = None
    try:
        media = entities["media"][0]["media_url"]
    except Exception as e:
        pass
    try:
        extended_entities=tweet['extended_entities']
        media = extended_entities["media"][0]["video_info"]["variants"][0]["url"]
    except Exception as e:
        pass
    if(text.startswith('RT')):
        retweet=True
    else:
        retweet=False

    url=entities['urls']
    symbols=entities['symbols']


    is_verified = tweet['user']['verified']
    is_quote = tweet['is_quote_status']

    
    tweet1 = {'_id' : tweet_id, 'date': created, 'username': username,  'text' : text,  'geoenabled' : geoenabled,  'coordinates' : coordinates,  'location' : location,  'place_name' : place_name, 'place_country' : place_country, 'country_code': place_countrycode,  'place_coordinates' : place_coordinates, 'hashtags' : hList, 'mentions' : mList, 'source' : source,'verified' :is_verified,'quote':is_quote,'retweet':retweet,'media':media,'url':url,'symbols':symbols}
    return tweet1

def processTweetsRest(tweet,i):
    place_countrycode  =None
    place_name  =None
    place_country =None
    place_coordinates =None
    source =None
    exactcoord =None
    place=None

    media = None
    
    tweet=tweet[i]
    tweet_id=tweet.id
    created=tweet.created_at
    username=tweet.user.screen_name
    text=tweet.full_text
    geoenabled=tweet.user.geo_enabled
    exactcoord = tweet.coordinates
    coordinates = None
    if(exactcoord):
        coordinates = exactcoord.coordinates
    location=tweet.user.location

    if ((geoenabled) and (text.startswith('RT') == False)):
        try:
            if(tweet.place):
                place_name = tweet.place.full_name
                place_country = tweet.place.country
                place_countrycode   = tweet.place.country_code
                place_coordinates   = tweet.place.bounding_box.coordinates
        except Exception as e:
            print(e)
            print('error from place details - maybe AttributeError: ... NoneType ... object has no attribute ..full_name ...')
    entities=tweet.entities
    hashtags=entities['hashtags']
    hList =[]
    for x in hashtags:
        # print(x['screen_name'])
        hList.append(x['text'])
        if hashtags == []:
            hashtags =''
        else:
            hashtags = str(hashtags).strip('[]')
    mentions =entities['user_mentions']
    mList = []
    for x in mentions:
        mList.append(x['screen_name'])
    source=tweet.source
    is_verified=tweet.user.verified
    is_quote=tweet.is_quote_status
    if(text.startswith('RT')):
        retweet=True
    else:
        retweet=False
    media = None
    try:
        media = entities["media"][0]["media_url"]
    except Exception as e:
        pass
    try:
        extended_entities=tweet['extended_entities']
        print("here")
        media = extended_entities["media"][0]["video_info"]["variants"][0]["url"]
    except Exception as e:
        pass
    url=entities['urls']
    symbols=entities['symbols']
    


    tweet1 = {'_id' : tweet_id, 'date': created, 'username': username,  'text' : text,  'geoenabled' : geoenabled,  'coordinates' : coordinates,  'location' : location,  'place_name' : place_name, 'place_country' : place_country, 'country_code': place_countrycode,  'place_coordinates' : place_coordinates, 'hashtags' : hList, 'mentions' : mList, 'source' : source,'verified' :is_verified,'quote':is_quote,'retweet':retweet,'media':media,'url':url,'symbols':symbols}
    return tweet1

class StreamListener(tweepy.StreamListener):
  #This is a class provided by tweepy to access the Twitter Streaming API.
    global geoEnabled
    global geoDisabled
    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")



    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        #This is where each tweet is collected
        # let us load the  json data
        t = json.loads(data)
        #  now let us process the wteet so that we will deal with cleaned and extracted JSON
        tweet = processTweets(t)
        #print(tweet)
        # now insert it
        #  for this to work you need to start a local mongodb server
        try:
            #f.write(data)
            print("Streaming API Inserted")
            collection.insert_one(tweet)
        except Exception as e:
            print(e)
            # this means some Mongo db insertion errort

#Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.

# WORDS = ['manhattan' , 'new york city', 'statue of liberty']
# LOCATIONS = [ -75,40,-72,42] # new york city
Loc_UK = [-10.392627, 49.681847, 1.055039, 61.122019] # UK and Ireland
Words_UK =["Boris", "Prime Minister", "Tories", "UK", "London", "England", "Manchester", "Sheffield", "York", "Southampton", \
 "Wales", "Cardiff", "Swansea" ,"Banff", "Bristol", "Oxford", "Birmingham" ,"Scotland", "Glasgow", "Edinburgh", "Dundee", "Aberdeen", "Highlands" \
"Inverness", "Perth", "St Andrews", "Dumfries", "Ayr" \
"Ireland", "Dublin", "Cork", "Limerick", "Galway", "Belfast"," Derry", "Armagh" \
"BoJo", "Labour", "Liberal Democrats", "SNP", "Conservatives", "First Minister", "Surgeon", "Chancelor" \
"Boris Johnson", "BoJo", "Keith Stramer"]
Words=["vaccine"]

#f = open("twitterData.txt","a+")


print("Tracking: " + str(Words))
#  here we ste the listener object
listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True))
streamer = tweepy.Stream(auth=auth, listener=listener)
streamer.filter(locations= Loc_UK, track = Words, languages = ['en'], is_async=True)
#  the following line is for pure 1% sample
# we can only use filter or sample - not both together
# streamer.sample(languages = ['en'])

Place =  'London'
Lat   =  '51.450798'
Long  =  '-0.137842'
geoTerm=Lat+','+Long+','+'10km'
#

last_id =  None
counter =0
sinceID = None
q=["vaccine"]

results = True

while results:

    if (counter < 180 ):
        try:
            results = api.search(q=q,geocode=geoTerm, count=100, lang="en", tweet_mode='extended', max_id=last_id) #, since_id = sinceID)
            for i in range(100):
                tweet = processTweetsRest(results,i)
                try:
                    collection.insert_one(tweet)
                    print("REST Inserted ",i)
                except Exception as e:
                    print("duplicate found (tweet not inserted)")
                    #print(e)
                
            
        except Exception as e:
            print(e)
        counter += 1
    else:
        #  the following let the crawler to sleep for 15 minutes; to meet the Tiwtter 15 minute restriction
        time.sleep(15*60)      
        counter = 0
        




