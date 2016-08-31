import pandas as pd
import numpy as np
from ast import literal_eval
import config
import glob
import fileinput
import re
from nltk.tokenize import TweetTokenizer


alltweets_file = config.data_dir + '/all_' + config.target_user + '_follower_tweets.csv'
tokenized_file = config.data_dir + '/tweets_tokenized.csv'
tokenized_and_preprocessed_file = config.data_dir + '/tweets_tokenized_and_preprocessed.csv'
english_file = 'data/tweets_english.csv'

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

                  
print(len(alltweets))                    

# tokenize
tokenizer = TweetTokenizer()
#alltweets['text_tokenized'] = alltweets['text'].apply(lambda t: tokenizer.tokenize(str(t.encode('utf-8').decode('utf-8'))))
alltweets_tokenized = alltweets
alltweets_tokenized['text_tokenized'] = alltweets_tokenized['text'].apply(lambda t: tokenizer.tokenize(t))
alltweets_tokenized.to_csv(tokenized_file,encoding='utf-8')

# preprocess
tokenized_and_preprocessed = alltweets_tokenized
# join list for better processing
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: " ".join(t))
# remove RT clause
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r'^RT @.*? :','', t))  

# …
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r'…', ' ', t))
# ...
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t:re.sub(r'\.\.\.', ' ', t))
# . one or more
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r' \.+ ', ' ',t)) 
# . at the end
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r' \.+$', ' ',t)) 

# other single non-chars
# test:
# 
#t='a :-) : a , a . a - a :) :( a ( a \ a / a " a'
#re.sub(r' [,.\-:()\/"] ', ' ',t)
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r' [,.\-:()&\'’\/"] ', ' ',t))
# second run for overlapping
tokenized_and_preprocessed['text_tokenized'] = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: re.sub(r' [,.\-:()&\'’\/"] ', ' ',t)) 



tokenized_and_preprocessed.to_csv(tokenized_and_preprocessed_file,encoding='utf-8')

# just EN
english_tweets = tokenized_and_preprocessed[tokenized_and_preprocessed['lang'] == 'en']
english_tweets.to_csv(english_file, encoding = 'utf-8')


