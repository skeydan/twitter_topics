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

c = tweepy.Cursor(api.followers_ids, id = 'Trivadis')
ids = []
with io.open('follower_ids.csv', 'w', encoding='utf-8') as ids_file: 
    for page in c.pages():
        ids.extend(page)
    ids.sort()
    ids_file.write(','.join(map(unicode,ids)))
    
print "len(ids)=", len(ids)

users = []
with io.open('followers.csv', 'a', encoding='utf-8') as users_file:
    for i, user_id in enumerate(ids):
        try:
            user = api.get_user(user_id)
            users.append(user)
            users_file.write(u'{},"{}","{}","{}","{}","{}","{}","{}",{}\n'.
                             format(user.id, user.screen_name, user.name, user.lang, user.location, user.created_at, user.followers_ids(), map(lambda user: user.id, user.friends()), user.statuses_count))  
            print i, user_id, len(users)
        except TweepError as e:
            if 'Failed to send request:' in e.reason:
                print "Time out error caught."
                time.sleep(180)
                continue
    



    
    

