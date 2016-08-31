import pandas as pd
import numpy as np
import graphlab as gl
from graphlab import SFrame
import matplotlib.pyplot as plt
import config


tokenized_and_preprocessed_file = config.data_dir + '/tweets_tokenized_and_preprocessed.csv'
english_file = config.data_dir  + '/' + 'tweets_english.csv'

alltweets = pd.read_csv(tokenized_and_preprocessed_file, encoding='utf-8',
                                  usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                             'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                             'retweet_count', 'quoted_status_id_str', 'text_tokenized'])
tweets_english = pd.read_csv(english_file, encoding='utf-8',
                              usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                         'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                         'retweet_count', 'quoted_status_id_str', 'text_tokenized'])

tweets_by_language = pd.groupby(alltweets, 'lang').size().sort_values(ascending = False)
#print(tweets_by_language)

X_train = tweets_english['text_tokenized']

X_train_sarray = SFrame(X_train)['X1']

word_counts = gl.text_analytics.count_words(X_train_sarray)
word_counts = word_counts.dict_trim_by_keys(gl.text_analytics.stopwords(), exclude=True)

for i in range(1,6):
    topic_model = gl.topic_model.create(word_counts, num_topics=i, num_iterations=200)
    print([x['words'] for x in topic_model.get_topics(output_type='topic_words', num_words=30)])
    for j in range(1,i):
        plt.plot(range(100), topic_model.get_topics(topic_ids=[j], num_words=100)['score'])
        plt.xlabel('Word rank')
        plt.ylabel('Probability')
        plt.title('Probabilities of Top 100 Words in each Topic')
        plt.show()
        
'''        
[[':', '.', 'rt', '!', '-', '"', '?', 'oracle', '...', '\xe2\x80\xa6', '(', ')', '/', '&', 'great', 'data', 'good', '#oracle', "'", 'today', 'cloud', 'day', ':)', 'time', '#orclapex', ':-)', '\xe2\x80\x99', 'sql', '2', 'blog']]

[[':', 'rt', '"', '?', '.', '-', '...', '\xe2\x80\xa6', ')', '(', '/', 'oracle', 'data', "'", 'sql', 'blog', 'video', 'cloud', '@sfonplsql', 'database', 'business', 'post', '|', '+', '1', '#oracle', 'big', 'windows', 'read', 'free'],
['.', '!', '-', '&', 'rt', 'great', 'oracle', 'good', 'today', 'day', ':)', 'time', '#orclapex', ':-)', "it's", "i'm", 'week', '#oracle', '@oracle', 'nice', '2015', "don't", 'session', 'work', '2016', 'join', '\xe2\x80\x99', '2', 'year', 'morning']]


[['!', '-', ':', 'rt', '(', ')', '?', '...', 'data', 'great', 'today', '2', 'video', '1', '|', '3', 'week', 'day', '5', 'year', '\xe2\x80\xa6', 'session', 'twitter', 'event', '4', 'free', 'open', 'big', 'happy', '10'],
[':', 'rt', 'oracle', '\xe2\x80\xa6', '&', '#oracle', 'cloud', '#orclapex', '\xe2\x80\x99', '!', 'blog', '@oracle', '2015', "'", 'business', 'database', '#opn', 'pm', '+', '#cloud', '@', 'live', 'http', 'learn', 'service', 'partner', 'customer', 'area', 'apex', 'join'],
['.', '"', '?', '...', '/', '!', 'good', ':', 'time', ':)', ':-)', 'sql', "it's", "i'm", "don't", '@sfonplsql', 'nice', 'day', 'work', "'", 'morning', 'make', 'back', ';)', 'read', '#plsql', ';-)', '\xe2\x80\x9c', '$', 'pl']]


[[':', 'rt', '.', 'oracle', '\xe2\x80\xa6', '#oracle', 'cloud', '&', '\xe2\x80\x99', '@oracle', '#orclapex', 'business', 'database', 'blog', '#opn', 'join', 'bi', '+', 'live', '#cloud', '\xe2\x80\x9c', '!', '12c', '\xe2\x80\x9d', 'partner', 'apex', 'service', '#oow15', 'learn', '\xe2\x80\x93'],
[':', 'rt', '...', '(', ')', '?', '&', '2', '!', 'data', 'week', '1', '2015', '2016', '3', 'windows', '5', 'pm', 'twitter', '4', '@', '10', 'area', 'free', 'entered', '%', 'work', '@panicc', '@lepetitbouton87', '@mary871202'],
['.', '!', 'good', 'great', 'today', 'day', ':)', 'time', ':-)', "it's", 'nice', 'year', 'morning', "i'm", ';-)', 'back', ';)', 'session', 'event', 'open', 'people', 'happy', 'hope', '#orclapex', '?', 'start', 'forward', 'working', 'home', 'fun'],
[':', '-', '"', 'rt', '?', '/', '\xe2\x80\xa6', "'", 'sql', 'data', 'video', '@sfonplsql', '|', 'big', 'check', "don't", '#plsql', '...', '$', 'pl', 'customer', 'http', 'code', 'performance', '@youtube', 'things', ':/', 'experience', 'world', 'make']]


[['.', '/', 'good', 'day', ':)', 'sql', '...', "it's", ':-)', "i'm", 'time', "don't", '@sfonplsql', 'work', 'morning', 'nice', 'today', ';-)', ';)', 'back', '#plsql', 'pl', 'code', 'hope', 'start', 'working', 'home', '@rmoff', '#fb', 'make'],
['.', '?', '...', ':', 'data', "'", '\xe2\x80\x99', '2', '+', 'week', 'business', '#opn', '@oracle', '3', 'twitter', 'people', '5', '$', 'partner', 'customer', '4', 'big', '%', '#cloud', 'experience', '@oraclepartners', 'digital', 'security', 'social', 'check'],
[':', 'rt', 'oracle', '#oracle', 'cloud', '&', '2015', 'database', '2016', 'windows', 'bi', 'pm', '12c', 'learn', 'area', 'service', 'entered', 'today', '@panicc', '@lepetitbouton87', '@mary871202', '\xe2\x80\x93', 'top', 'server', 'live', '@', 'update', '10', 'free', '8'], 
[':', '-', '"', 'rt', '(', ')', '\xe2\x80\xa6', 'video', 'blog', '?', 'post', '|', '&', 'http', 'data', 'read', '@youtube', ':/', 'management', 'web', 'software', '\xe2\x80\x94', 'interesting', '*', 'full', '1', 'follow', 'article', 'agile', '#'], 
['!', ':', 'rt', 'great', '.', '\xe2\x80\xa6', '#orclapex', 'year', '&', 'join', '\xe2\x80\x9c', 'open', 'apex', '\xe2\x80\x9d', 'happy', 'session', 'event', 'team', 'conference', 'forward', 'community', '@ukoug', '@pythian', 'book', '#kscope16', 'today', 'time', '#datavault', 'congrats', 'awesome']]
 '''       
