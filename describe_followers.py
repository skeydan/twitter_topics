import pandas as pd
import numpy as np

users = pd.read_csv('followers.csv', header=None, 
                    names=['id', 'screen_name', 'name', 'lang', 'location', 'created_at', 'follower_ids', 'friend_ids', 'tweet_count'],
                    encoding='utf-8',
                    sep=',')
