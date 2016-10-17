from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import pandas as pd
import config
import json
import datetime
import os
import csv

class UserIdListener(StreamListener):
    def __init__(self, target):      
        self.dir_prefix = config.dir_prefix
        self.outdir = "{}/data/twitter/tweets/{}".format(self.dir_prefix, target)
        curdate = datetime.datetime.now().strftime("%Y-%m-%d")
        user_ids_file = "{}/data/twitter/followers/{}/{}/follower_ids_{}.csv".format(self.dir_prefix, target, curdate, target)
        self.user_ids = [x for x in open(user_ids_file)][0].split(',')
    
    def on_data(self, data):
        try:
            tweet = data
            user_id = json.loads(data)['user']['id_str']            
            print(user_id)
            curdate = datetime.datetime.now().strftime("%Y-%m-%d")
            daily_dir = '/'.join([self.outdir, curdate])
            if not os.path.exists(daily_dir): os.makedirs(daily_dir)
            user_file = '/'.join([daily_dir, "user_" + user_id + "_" + curdate]) 
            with open(user_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                        [tweet.id_str, tweet.user.id, tweet.created_at, tweet.lang, tweet.text, tweet.retweeted, tweet.favorite_count, tweet.entities,
                        tweet.in_reply_to_screen_name, tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id, 
                        tweet.quoted_status_id_str if hasattr(tweet, 'tweet.quoted_status_id_str') else None,
                        tweet.retweet_count])
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            print(data)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = UserIdListener('trivadis')
    auth = OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_secret)

    stream = Stream(auth, l)
    # https://dev.twitter.com/streaming/overview/request-parameters#follow
    stream.filter(follow = l.user_ids)
    #stream.filter(track = 'machinelearning')
