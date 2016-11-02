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
import sys
import argparse
import access

class TwitterListener(StreamListener):
    def __init__(self, target):     
        print('Initializing parent class: {}, target = {}'.format(self, target))
        self.target = target
        self.dir_prefix = config.dir_prefix
        self.outdir = "{}/data/twitter/tweets/{}".format(self.dir_prefix, target)
            
    def on_data(self, data):
        try:
            tweet = json.loads(data)
            user_id = tweet['user']['id_str']
            curdate = datetime.datetime.now().strftime("%Y-%m-%d")
            daily_dir = '/'.join([self.outdir, curdate])
            if not os.path.exists(daily_dir): os.makedirs(daily_dir)
            user_file = '/'.join([daily_dir, "user_" + user_id + "_" + curdate])
            print('Writing: {}'.format(user_file))
            with open(user_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                        [tweet["id_str"], tweet["user"]["id"], tweet["created_at"], tweet["lang"], tweet["text"], tweet["retweeted"], tweet["favorite_count"], tweet["entities"],
                        tweet["in_reply_to_screen_name"], tweet["in_reply_to_status_id_str"], tweet["in_reply_to_user_id"],
                        tweet["quoted_status_id_str"] if hasattr(tweet, "quoted_status_id_str") else None, tweet["retweet_count"]])
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            print(data)
        return True

    def on_error(self, status):
        print('Got an error: {}'.format(status))
        
        
class UserIdListener(TwitterListener):
    def __init__(self, target):    
        super().__init__(target) 
        print('Initializing subclass: {}, target = {}'.format(self, target))
        curdate = datetime.datetime.now().strftime("%Y-%m-%d")
        user_ids_file = "{}/data/twitter/followers/{}/{}/follower_ids_{}.csv".format(self.dir_prefix, target, curdate, target)
        self.user_ids = [x for x in open(user_ids_file)][0].split(',')

def get_auth_tokens(application):
    tokens =  {
        "tweetgetter777": [access.consumer_key_1, access.consumer_secret_1, access.access_token_1, access.access_secret_1],
        "tweetgetter778": [access.consumer_key_2, access.consumer_secret_2, access.access_token_2, access.access_secret_2],
        "tweetgetter779": [access.consumer_key_3, access.consumer_secret_3, access.access_token_3, access.access_secret_3],
        "tweetgetter780": [access.consumer_key_4, access.consumer_secret_4, access.access_token_4, access.access_secret_4]
        }
    return tokens.get(application, "missing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("app", help="application to use")
    parser.add_argument("target", help="storage directory")
    parser.add_argument("-f", "--track_followers", action="store_true", help="track tweets by followers following this target")
    parser.add_argument("-t", "--track_words", action="store", help="words to track for this target")
    parser.add_argument("-l", "--languages", action="store", help="languages filter")
    args = parser.parse_args()
    print('App: {}    Target: {}    track followers? {}    track words: {}    languages: {} '.format(
        args.app, args.target, args.track_followers, args.track_words, args.languages))
    
    (consumer_key, consumer_secret, access_token, access_secret) = get_auth_tokens(args.app)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    if args.track_followers: 
        l = UserIdListener(args.target)
        stream = Stream(auth, l)
        stream.filter(follow = l.user_ids)
    else:
        l = TwitterListener(args.target)
        stream = Stream(auth, l)
        if args.languages != None: 
            stream.filter(track = [args.track_words], languages=['de'])
        else: 
            stream.filter(track = [args.track_words])
        
# see https://dev.twitter.com/streaming/overview/request-parameters
# twitter_stream_listener.py tweetgetter777 trivadis -f
# twitter_stream_listener.py tweetgetter778 dbiservices -f
# twitter_stream_listener.py tweetgetter779 toyota_vw -t 'toyota,volkswagen,vw' -l 'de'     
# twitter_stream_listener.py tweetgetter780 emilfrey_amag_asag -t 'emil frey,amag,asag' -l 'de'



# By this model, you can think of commas as logical ORs, while spaces are equivalent to logical ANDs (e.g. ‘the twitter’ is the AND twitter, and ‘the,twitter’ is the OR twitter).
# e.g. 
