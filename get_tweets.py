import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import TweepError
from tweepy.streaming import Stream
import config
import sys
import io
import time
import pandas as pd
import numpy as np
import csv

MAX_TIMELINE_PAGES=17

auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


users = pd.read_csv('data/followers_' + config.target_user + '.csv', header=None, index_col=False,
                    names=['id', 'screen_name', 'name', 'entities', 'url', 'description', 'lang', 'location', 'created_at', 'follower_ids', 'friend_ids', 'tweet_count'],
                    encoding='utf-8', quoting=0)

for index,user in users.iterrows():
    
    try:
        
        print("Now getting tweets from user: {} {}".format(user.id, user.screen_name))
        alltweets = []	
        new_tweets = api.user_timeline(id = user.id, count=200, include_rts=True, exclude_replies=False)
        alltweets.extend(new_tweets)
        max_id = alltweets[-1].id - 1
        
        cursor = tweepy.Cursor(api.user_timeline, id=user.id, count=200, max_id=max_id).pages(MAX_TIMELINE_PAGES)
            
        i=0    
        for page in cursor:
            print('Retrieving page: {}'.format(i))
            i += 1
            for tweet in page:
                alltweets.append(tweet)
            max_id = page[-1].id - 1
        
        print("Tweets downloaded overall: {}".format((len(alltweets))))
        
        filename = 'data/{}_{}_{}.csv'.format(config.target_user, user.id, user.screen_name)
        with io.open(filename, 'w', encoding='utf-8') as tweets_file:
            writer = csv.writer(tweets_file)
            for tweet in alltweets:
                writer.writerow(
                    [tweet.id_str, tweet.user.id, tweet.created_at, tweet.lang, tweet.text, tweet.retweeted, tweet.favorite_count, tweet.entities,
                    tweet.in_reply_to_screen_name, tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id, 
                    tweet.quoted_status_id_str if hasattr(tweet, 'tweet.quoted_status_id_str') else None,
                    tweet.retweet_count])
    except TweepError as e:             
        if 'Failed to send request:' in e.reason:
                print("Time out error caught.")
                time.sleep(180)
                continue
                
