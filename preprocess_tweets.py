import pandas as pd
import numpy as np
from ast import literal_eval
import config
import sys
import glob
import fileinput
import re
from nltk.tokenize import TweetTokenizer

pd.set_option('display.max_colwidth', 200)

target = sys.argv[1]
target_month = sys.argv[2]
source_dir = "{}/data/twitter/tweets/{}".format(config.dir_prefix, target)

alltweets_file = '/'.join([source_dir, target_month + '_raw.csv'])
tokenized_file = '/'.join([source_dir, target_month  + '_tokenized.csv'])
tokenized_and_preprocessed_file = '/'.join([source_dir, target_month  + '_tokenized_and_preprocessed.csv'])
english_file = '/'.join([source_dir, target_month  + '_english.csv'])
french_file = '/'.join([source_dir, target_month  + '_french.csv'])
german_file = '/'.join([source_dir, target_month  + '_german.csv'])


user_files = glob.glob('/'.join([source_dir, target_month + '-*/user*']))
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
alltweets['text_tokenized'] = alltweets['text'].apply(lambda t: tokenizer.tokenize(str(t.encode('utf-8').decode('utf-8'))))
alltweets_tokenized = alltweets
alltweets_tokenized['text_tokenized'] = alltweets_tokenized['text'].apply(lambda t: tokenizer.tokenize(t))
alltweets_tokenized.to_csv(tokenized_file,encoding='utf-8')

# preprocess
#test_tweet = 'This is a cooool. #dummysmiley: :-) :-P <3 and some arrows < > -> <-- and what? why! where& (yes) [no] minus- plus+ \word /word2 ;-) :-( :-((( @zkajdan'
#remove_list = ['?', '!', '.', '\\', '-', ':', '(', ')', '&', '’', '/', '[', ']', '…', '>', '<', '->', '<--', '+']
#test_tweet_tokenized = tokenizer.tokenize(test_tweet)
#[i for i in test_tweet_tokenized if i not in remove_list]

remove_list = ['?', '!', '–', '.', '*', '...', '"', '\'', '\\', '-', ':', '(', ')', '&', '’', '/', '[', ']', '…', '>', '<', '->', '<--', '+']
remove_list.append('RT')
remove_list.append('via')
tokenized_and_preprocessed = alltweets_tokenized
tokenized_and_preprocessed['text_processed']  = tokenized_and_preprocessed['text_tokenized'].apply(lambda t: [token for token in t if token not in remove_list])


tokenized_and_preprocessed.to_csv(tokenized_and_preprocessed_file,encoding='utf-8')

# just EN/FR/D
english_tweets = tokenized_and_preprocessed[tokenized_and_preprocessed['lang'] == 'en']
french_tweets = tokenized_and_preprocessed[tokenized_and_preprocessed['lang'] == 'fr']
german_tweets = tokenized_and_preprocessed[tokenized_and_preprocessed['lang'] == 'de']
english_tweets.to_csv(english_file, encoding = 'utf-8')
french_tweets.to_csv(french_file, encoding = 'utf-8')
german_tweets.to_csv(german_file, encoding = 'utf-8')

