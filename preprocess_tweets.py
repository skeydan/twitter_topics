import pandas as pd
import numpy as np
from ast import literal_eval
import config
import glob
import fileinput
from nltk.tokenize import TweetTokenizer

alltweets_file = 'data/all_' + config.target_user + '_follower_tweets.csv'

user_files = glob.glob('data/Trivadis_*_*.csv')
with open(alltweets_file, 'w') as fout:
    f = fileinput.input(files=(user_files))
    for line in f:
        fout.write(line)
    f.close()
        

alltweets = pd.read_csv(alltweets_file, header=None, index_col=False,
                    names=['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                           'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                           'retweet_count', 'quoted_status_id_str'],
                    encoding='utf-8', quoting=0)

tokenizer = TweetTokenizer()
#alltweets['text_tokenized'] = alltweets['text'].apply(lambda t: tokenizer.tokenize(str(t.encode('utf-8').decode('utf-8'))))


alltweets['text_tokenized'] = alltweets['text'].apply(lambda t: tokenizer.tokenize(t))


#http://stackoverflow.com/questions/11339955/python-string-encode-decode
#https://dev.twitter.com/rest/public/timelines
#https://marcobonzanini.com/2015/03/17/mining-twitter-data-with-python-part-3-term-frequencies/
#http://stackoverflow.com/questions/5096776/unicode-decodeutf-8-ignore-raising-unicodeencodeerror

#http://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.sentiment_analyzer
    
