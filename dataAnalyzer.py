import collections
import itertools
import json
import logging
import sys
from itertools import islice

import pandas as pd
import sklearn
import spacy
import tweepy
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

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
    
spacy.load("en_core_web_sm")
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')

def spacy_tokenize(string):
  tokens = list()
  doc = nlp(string)
  for token in doc:
    tokens.append(token)
  return tokens

def normalize(tokens):
  normalized = list()
  for token in tokens:
    if (token.is_alpha or token.is_digit):
      lemma = token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_
      normalized.append(lemma)
  return normalized
  
def tokenize_normalize(string):
  return normalize(spacy_tokenize(string))
  
def extraction(f):
    imageCount=0
    videoCount=0
    retweetCount=0
    quoteCount=0
    verifiedCount=0
    geotaggedCount=0
    locationCount=0
    place_nameCount=0
    tweetCollection = collection.find({},{'text' : 1,'retweet':1,'quote':1,'media':1,'verified':1,'geoenabled':1,'location':1})
    tweetList = list()
    for tweet in tweetCollection:
        tweetList.append(tweet)

    labels=['text','retweet','quote','media','verified','geoenabled','location','place_name']
    tweet_frame=pd.DataFrame(tweetList,columns=labels)

    tweet_vals=list()
    
    for tweet in islice(tweet_frame.itertuples(index=True, name='Pandas'), 25000):
        tweet_vals.append(getattr(tweet,'text'))

    for media in tweet_frame['media']:
        if(media!=None):
            if("pbs" in media):
                imageCount+=1 
            else:
                videoCount+=1
    
    for retweet in tweet_frame['retweet']:
        if(retweet):retweetCount+=1
    for quote in tweet_frame['quote']:
        if(quote):quoteCount+=1
    for verified in tweet_frame['verified']:
        if(verified):verifiedCount+=1
    for geotagged in tweet_frame['geoenabled']:
        #print(geotagged)
        if(geotagged):geotaggedCount+=1
    for location in tweet_frame['location']:
        if(location!=None):locationCount+=1
    for place_name in tweet_frame['place_name']:
        if(place_name!=None):place_nameCount+=1
    print(f"image :{imageCount}")
    f.write(f"image :{imageCount}\n")
    print(f"video :{videoCount}")
    f.write(f"video :{videoCount}\n")
    print(f"retweet :{retweetCount}")
    f.write(f"retweet :{retweetCount}\n")
    print(f"quote :{quoteCount}")
    f.write(f"quote :{quoteCount}\n")
    print(f"verified :{verifiedCount}")
    f.write(f"verified :{verifiedCount}\n")
    print(f"geotagged :{geotaggedCount}")
    f.write(f"geotagged :{geotaggedCount}\n")
    print(f"location :{locationCount}")
    f.write(f"location :{locationCount}\n")
    print(f"place_name :{place_nameCount}")
    f.write(f"place_name :{place_nameCount}\n")
    posts=collection.count()
    print(f"posts: {posts}")
    f.write(f"posts: {posts}\n")
    duplicate=posts-len(collection.distinct('_id'))

    print(f"duplicate posts : {duplicate}")
    f.write(f"duplicate posts : {duplicate}")

def clustering(f):
    tweetCollection = collection.find({},{'id' : 1, 'text' : 1, 'hashtags' : 1,'url':1,'symbols':1})
    tweetList = list()
    for tweet in tweetCollection:
        tweetList.append(tweet)

    labels=['id','text', 'hashtags','url','symbols']
    tweet_frame=pd.DataFrame(tweetList,columns=labels)

    tweet_vals=list()
    tweet_keys=list()
    
    for tweet in islice(tweet_frame.itertuples(index=True, name='Pandas'), 25000):
        tweet_vals.append(getattr(tweet,'text'))
        tweet_keys.append(getattr(tweet,'id'))

    vectorizer = TfidfVectorizer(tokenizer = tokenize_normalize, ngram_range=(1, 2), sublinear_tf = True, max_features = 100000,stop_words='english')
    document_matrix = vectorizer.fit_transform(tweet_vals)

    num_clusters = 50
    kmeans = KMeans(n_clusters=num_clusters, init='random', n_init=1)
    kmeans.fit(document_matrix)
    
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(kmeans.labels_):
        clustering[label].append(idx)
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for cluster, indices in clustering.items():
        print("\nCluster:", cluster, " Num posts: ", len(indices))
        f.write('\nCluster:{0} Num posts: {1}; '.format(cluster, len(indices)))
        for ind in order_centroids[cluster, :20]:
            print(' %s' % terms[ind])
            try:
                f.write(' %s,' % terms[ind])
            except Exception:
                f.write(' (undefined), ')
        #print()
        f.write("\n")
        cur_docs = 0
        print("sample hashtags: ") 
        f.write(" hashtags: ")
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'hashtags']
            if ht:
                print(ht)
                try:
                    f.write(' %s' % ht)
                except Exception:
                    pass
                cur_docs+=1
        cur_docs=0
        print("sample url: ") 
        f.write("sample url: ")
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'url']
            if ht:
                print(ht)
                try:
                    f.write(' %s' % ht)
                except Exception:
                    pass
                cur_docs+=1
        cur_docs=0
        print("sample symbols: ") 
        f.write("sample symbols: ")
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'symbols']
            if ht:
                print(ht)
                try:
                    f.write(' %s' % ht)
                except Exception:
                    pass
                cur_docs+=1        
            
f= open("Results.txt","w")
extraction(f)
clustering(f)

