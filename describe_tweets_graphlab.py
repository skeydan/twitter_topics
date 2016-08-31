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
        plt.figure()
        plt.plot(range(100), topic_model.get_topics(topic_ids=[j], num_words=100)['score'])
        plt.xlabel('Word rank')
        plt.ylabel('Probability')
        plt.title('Probabilities of Top 100 Words in each Topic')
        plt.show()
        
'''
[
['!', '?', 'oracle', 'data', 'great', '#oracle', 'cloud', 'good', ':)', 'time', 'today', 'sql', '2', "it's", 'day', 'database', '1', ':-)', 'business', 'blog', "i'm", 'week', '#orclapex', 'big', "don't", '#cloud', '12c', '#bigdata', 'rt', 'check']
]


[
['oracle', 'data', '?', '#oracle', 'cloud', 'day', 'database', 'business', 'big', '#cloud', '12c', '#bigdata', '10', 'learn', 'join', '#iamcp', 'windows', '2015', 'free', 'service', '\xe2\x80\x9c', 'bi', 'people', '\xe2\x80\x9d', '2016', 'analytics', '\xe2\x80\x93', 'live', 'partner', '#wpc16'], 
['!', '?', 'great', 'good', ':)', 'time', 'sql', "it's", ':-)', "i'm", '#orclapex', "don't", 'today', 'rt', 'post', '2', ';-)', '1', 'make', 'nice', 'work', 'year', 'top', ')', 'week', 'find', 'blog', 'check', 'community', 'happy']
]


[
['oracle', '#oracle', 'cloud', 'sql', 'database', 'blog', '12c', 'today', 'check', 'learn', '2015', 'service', 'bi', 'join', '2016', '\xe2\x80\x93', '?', 'live', 'partner', 'free', 'mobile', 'performance', 'support', '|', 'security', 'management', 'data', 'enterprise', 'part', 'apps'],
['!', '?', 'data', 'great', 'week', 'big', "don't", '#bigdata', 'video', 'day', '"', 'good', '\xe2\x80\x9c', '#cloud', 'business', '\xe2\x80\x9d', 'top', 'time', ')', 'happy', '@oracle', 'twitter', ':)', 'back', 'people', 'today', '>', 'stories', '5', 'partners'],
['!', '?', '2', "it's", ':-)', '1', "i'm", ':)', 'great', 'rt', '10', '#orclapex', 'good', '#iamcp', '+', 'windows', 'time', ';-)', '#wpc16', '$', 'year', 'post', 'nice', '#', 'code', '3', 'session', 'make', 'event', 'world']
]


[
['#orclapex', 'video', 'learn', 'join', '"', 'check', '\xe2\x80\x9c', '\xe2\x80\x9d', '2016', 'world', '2', 'http', 'today', 'live', 'open', 'https', '2015', 'time', '@', 'apex', 'conference', ':/', 'rt', 'session', 'talk', '+', 'watch', 'webinar', '2014', 'register'],
['oracle', 'data', '#oracle', 'cloud', 'database', 'business', '1', 'blog', 'big', '#cloud', '12c', '#bigdata', 'post', 'service', 'bi', 'analytics', '@oracle', 'security', 'management', 'enterprise', 'azure', '\xe2\x80\x93', 'performance', 'services', 'part', 'release', 'server', 'features', 'read', 'integration'],
['!', '?', 'sql', 'day', '10', 'today', 'make', 'windows', 'time', '5', '$', 'microsoft', 'app', 'rt', 'work', 'code', 'java', '3', 'win', '%', 'year', 'google', 'support', '8', 'developer', 'build', 'years', 'pl', '+', 'nice'],
['!', '?', 'great', ':)', 'good', ':-)', "i'm", 'week', "it's", '#iamcp', ';-)', "don't", 'top', ')', '#wpc16', 'event', 'happy', 'twitter', 'follow', '*', 'community', '>', 'people', ';)', 'stories', "that's", 'home', 'nice', 'forward', 'hope']
]


[
['?', 'good', ':)', '!', "it's", ':-)', 'sql', "don't", ';-)', 'work', ')', '*', 'find', ';)', 'nice', "that's", '>', '#plsql', 'interesting', '@franckpachot', 'pl', 'morning', "i'm", "you're", '=', "can't", 'read', 'people', "what's", 'start'],
['oracle', '#oracle', 'database', '1', 'blog', '12c', '2', 'post', 'service', 'performance', 'support', 'part', 'bi', '3', 'check', '[', 'update', '\xe2\x80\x93', 'release', '?', ']', 'server', 'features', '4', 'application', 'soa', 'developer', 'download', 'released', 'mobile'],
['cloud', 'business', '#cloud', '#bigdata', 'learn', '#iamcp', 'analytics', 'top', 'partner', '#wpc16', '@oracle', 'management', '|', '2015', 'customer', 'https', 'things', 'make', '2016', 'partners', 'webinar', 'technology', '.', 'experience', '#analytics', 'customers', '#iot', 'tech', '#opn', 'social'],
['data', '10', '"', 'big', 'windows', 'video', '\xe2\x80\x9c', 'time', '\xe2\x80\x9d', "i'm", '$', '%', 'microsoft', 'free', 'rt', 'app', '+', 'azure', '5', 'apps', 'win', 'home', '8', 'google', 'build', 'years', 'code', '3', 'java', '@youtube'],
['!', 'great', 'today', 'day', '#orclapex', 'week', 'time', 'session', 'join', 'year', 'twitter', 'event', 'happy', 'talk', 'conference', 'follow', 'open', 'days', 'team', 'forward', '@', 'community', 'presentation', 'rt', 'apex', 'ready', 'live', 'awesome', 'tomorrow', 'back']
]

'''
