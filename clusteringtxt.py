import collections
import itertools
import json
import logging
import sys
from itertools import islice

import pandas as pd
import sklearn
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

import emoji
import re

logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

    
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

def processTweets(tweet):
    try:
        tweet_id = tweet['id_str']  # The Tweet ID from Twitter in string format
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
            try:
                if( tweet['retweeted_status']['truncated'] == True):
                    text = tweet['retweeted_status']['extended_tweet']['full_text']
                else:
                    text = tweet['retweeted_status']['full_text']

            except Exception as e:
                pass

    except Exception as e:
        print(e)
    text = cleanList(text)
    entities = tweet['entities']
    hashtags = entities['hashtags'] 
    hList =[]
    for x in hashtags:
        hList.append(x['text'])
        if hashtags == []:
            hashtags =''
        else:
            hashtags = str(hashtags).strip('[]')


    url=entities['urls']
    symbols=entities['symbols']

    
    tweet1 = {'_id' : tweet_id, 'text' : text, 'hashtags':hashtags,'url':url,'symbols':symbols}
    return tweet1

def clustering(tweetList):
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
        for ind in order_centroids[cluster, :20]:
            print(' %s' % terms[ind])
        #print()
        cur_docs = 0
        print("sample hashtags: ") 
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'hashtags']
            if ht:
                print(ht)
                cur_docs+=1
        cur_docs=0
        print("sample url: ") 
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'url']
            if ht:
                print(ht)
                cur_docs+=1
        cur_docs=0
        print("sample symbols: ") 
        for index in indices:
            if (cur_docs > 3):
                break
            ht=tweet_frame.loc[index,'symbols']
            if ht:
                print(ht)
                cur_docs+=1        
            
f= open("twitterData.txt","r")

tweetList=list()
for t in f:
    try:
        t=json.loads(t)
        t = processTweets(t)
        tweetList.append(t)
    except Exception:
        pass

clustering(tweetList)
input("\nType anything to continue\n")
