import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import TweepError
from tweepy.streaming import Stream
import config
import sys
import io
import time
import csv
import datetime
import os
import argparse
import access

parser = argparse.ArgumentParser()
parser.add_argument("app", help="application to use")
parser.add_argument("target", help="storage directory")
args = parser.parse_args()
print('App: {}    Target: {}  '.format(args.app, args.target))

def get_auth_tokens(application):
    tokens =  {
        "tweetgetter777": [access.consumer_key_1, access.consumer_secret_1, access.access_token_1, access.access_secret_1],
        "tweetgetter778": [access.consumer_key_2, access.consumer_secret_2, access.access_token_2, access.access_secret_2],
        "tweetgetter779": [access.consumer_key_3, access.consumer_secret_3, access.access_token_3, access.access_secret_3],
        "tweetgetter780": [access.consumer_key_4, access.consumer_secret_4, access.access_token_4, access.access_secret_4],
        "tweetgetter781": [access.consumer_key_5, access.consumer_secret_5, access.access_token_5, access.access_secret_5],
        "tweetgetter782": [access.consumer_key_6, access.consumer_secret_6, access.access_token_6, access.access_secret_6],
        "tweetgetter783": [access.consumer_key_7, access.consumer_secret_7, access.access_token_7, access.access_secret_7]
        }
    return tokens.get(application, "missing")

(consumer_key, consumer_secret, access_token, access_secret) = get_auth_tokens(args.app)
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

c = tweepy.Cursor(api.followers_ids, id = args.target)
ids = []
outdir = "{}/data/twitter/followers/{}".format(config.dir_prefix, args.target)
curdate = datetime.datetime.now().strftime("%Y-%m-%d")
daily_dir = '/'.join([outdir, curdate])
if not os.path.exists(daily_dir): os.makedirs(daily_dir)
filename = '/'.join([daily_dir, "follower_ids_" + args.target + ".csv"]) 

with io.open(filename, 'w', encoding='utf-8') as ids_file: 
    writer = csv.writer(ids_file)
    for page in c.pages():
        ids.extend(page)
    ids.sort()
    writer.writerow(ids)




    
    

