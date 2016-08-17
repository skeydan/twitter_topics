import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import TweepError
from tweepy.streaming import Stream
import config
import sys
import io
import time


auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)




    



    
    

