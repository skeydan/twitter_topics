import pandas as pd
import numpy as np

#from: http://help.sentiment140.com/for-students/

'''
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
'''

s140_train = pd.read_csv('model_data/training.1600000.processed.noemoticon.csv', encoding='latin-1',
                         names=['sentiment','id','date','query','user','text'], header = None)

s140_train = s140_train[['sentiment','text']]

s140_train = s140_train[s_140_train['sentiment'] != 2]

X_train = s140_train['text']
y_train = pd.where(s140_train == 0, s_140_train, 1)
