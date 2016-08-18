import pandas as pd
import numpy as np
from ast import literal_eval
import config

users = pd.read_csv('data/followers_' + config.target_user + '.csv', header=None, 
                    names=['id', 'screen_name', 'name', 'lang', 'location', 'created_at', 'follower_ids', 'friend_ids', 'tweet_count'],
                    encoding='utf-8', quoting=0)

users.follower_ids = users.follower_ids.apply(literal_eval)
users.friend_ids = users.friend_ids.apply(literal_eval)

