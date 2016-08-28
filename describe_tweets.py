import pandas as pd
import numpy as np

import config
import operator
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


tokenized_file = config.data_dir  + '/' + 'tweets_tokenized.csv'
english_file = config.data_dir  + '/' + 'tweets_english.csv'

alltweets_tokenized = pd.read_csv(tokenized_file, encoding='utf-8',
                                  usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                             'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                             'retweet_count', 'quoted_status_id_str', 'text_tokenized'])
tweets_english = pd.read_csv(english_file, encoding='utf-8',
                              usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                         'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                         'retweet_count', 'quoted_status_id_str', 'text_tokenized'])

tweets_by_language = pd.groupby(alltweets_tokenized, 'lang').size().sort_values(ascending = False)
#print(tweets_by_language)

X_train = tweets_english['text_tokenized']
X_train = X_train.apply(lambda t: t.replace(',',''))
stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))


print( '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3,4,5     *
******************************************************************************/
''')
'''
for remove_stop_words in [stopwords_filtered, None]:
    print('\n\nStop words removed: {}\n*******************************'.format(remove_stop_words))
    for i in range(1,6):
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
                                    stop_words = remove_stop_words, max_features = 100000, ngram_range = (1,i))

        # sparse matrix
        words_matrix = vectorizer.fit_transform(X_train)
        vocabulary = vectorizer.get_feature_names()
        counts = words_matrix.sum(axis=0).A1
       
        word_counts_overall = pd.DataFrame({'word': vocabulary, 'count': counts})
           
        word_counts_for_max_ngram = word_counts_overall[word_counts_overall.word.apply(lambda c: len(c.split()) >= i)]
           
        word_counts_for_max_ngram_sorted = word_counts_for_max_ngram.sort_values(by='count', ascending=False)
        print('\nMost frequent ngrams for ngrams in range 1 - {}:'.format(i))
            
        print(word_counts_for_max_ngram_sorted[:100])
        
'''        
print( '''
/******************************************************************************
*    Co-Occurrence Matrix                                                     *
******************************************************************************/
''')

def get_co_occurrences(record):
    wordlist = record.split()
    for i in range(len(wordlist) - 1):
        for j in range(i+1, len(wordlist)):    
            w1, w2 = sorted([wordlist[i], wordlist[j]]) 
            if w1 != w2:
                co_occurrences[w1][w2] += 1
                co_occurrences[w2][w1] += 1
                

X_train = X_train[:2]
X_train =pd.Series(['one two three', 'one two', 'two three three three'])

# co_occurrences, by word, ordered alphabetically
# each word appears as key
co_occurrences = defaultdict(lambda : defaultdict(int))
X_train.apply(get_co_occurrences) 
print('co_occurrences:')

i=0
for word in co_occurrences:
    i = i+1
    if i > 9: break
    print (word)
    for oc in co_occurrences[word]:
        print (oc,':',co_occurrences[word][oc])
        
co_occurrences_df = pd.DataFrame.from_dict(co_occurrences)   
print(co_occurrences_df[:10])

# co_occurrences, for each key ordered by frequency
# type(co_occurrences_by_freq['one'])
co_occurrences_by_freq = {word: sorted(co_occurrences[word].items(), key = lambda k_v: v[1], reverse=True) for word in co_occurrences}
print('co_occurrences ordered by frequency for each word:')

i=0
for word in co_occurrences_by_freq:
    i = i+1
    if i > 9: break
    print (word)
    for tup in co_occurrences_by_freq[word]: print(tup[0] + ': ' + str(tup[1]))

# co_occurrences sorted by highest value per word
co_occurrences_by_highest_freq = sorted(co_occurrences_by_freq.items(), key = lambda k_v: k_v[1][0][1], reverse=True)
print('co_occurrences sorted by highest value per word:')
print(co_occurrences_by_highest_freq[:10])

topics = ['oracle', 'windows', 'cloud', 'sql', 'orclapex', ]
topics_hashtags = map(lambda s: '#'+s, topics)
topics_with_hashtags = topics + topics_hashtags
     
people = ['@sfonplsql']    

vendors = ['oracle']
vendors_at = map(lambda s: '@'+s, vendors)
vendors_with_at = vendors + vendors_at
     
sentiment_words = ['great', 'good', 'like', 'thanks']

# do whole co-occurrence matrix
 



print( '''
/******************************************************************************
*    Sentiment Analysis nltk                                                  *
******************************************************************************/
''')


print( '''
/******************************************************************************
*    Sentiment Analysis unsupervised                                          *
******************************************************************************/
''')


