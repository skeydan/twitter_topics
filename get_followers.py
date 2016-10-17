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


auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

target = sys.argv[1]
c = tweepy.Cursor(api.followers_ids, id = target)
ids = []
filename = 'data/follower_ids_' + target + '.csv'
with io.open(filename, 'w', encoding='utf-8') as ids_file: 
    writer = csv.writer(ids_file)
    for page in c.pages():
        ids.extend(page)
    ids.sort()
    writer.writerow(ids)
    
users = []
filename = 'data/followers_' + target + '.csv'
with io.open(filename, 'a', encoding='utf-8') as users_file:
    writer = csv.writer(users_file)
    for i, user_id in enumerate(ids):
        try:
            user = api.get_user(user_id)
            users.append(user)
            writer.writerow([user.id, user.screen_name, user.name, user.entities, user.url, user.description, user.lang, user.location, user.created_at, user.followers_ids(), map(lambda user: user.id, user.friends()), user.statuses_count])
            print(i, user_id, len(users))
        except TweepError as e:
            if 'Failed to send request:' in e.reason:
                print('Time out error caught.')
                time.sleep(180)
                continue
    



    
    

