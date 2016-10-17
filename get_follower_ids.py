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


auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

target = sys.argv[1]

c = tweepy.Cursor(api.followers_ids, id = target)
ids = []
outdir = "{}/data/twitter/followers/{}".format(config.dir_prefix, target)
curdate = datetime.datetime.now().strftime("%Y-%m-%d")
daily_dir = '/'.join([outdir, curdate])
if not os.path.exists(daily_dir): os.makedirs(daily_dir)
filename = '/'.join([daily_dir, "follower_ids_" + target + ".csv"]) 

with io.open(filename, 'w', encoding='utf-8') as ids_file: 
    writer = csv.writer(ids_file)
    for page in c.pages():
        ids.extend(page)
    ids.sort()
    writer.writerow(ids)




    
    

