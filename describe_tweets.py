import pandas as pd
import numpy as np

import config
import sys
import re
from ast import literal_eval
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

pd.set_option('display.max_colwidth', 200)

target = sys.argv[1]
target_month = sys.argv[2]
source_dir = "{}/data/twitter/tweets/{}".format(config.dir_prefix, target)

tokenized_and_preprocessed_file = '/'.join([source_dir, target_month  + '_tokenized_and_preprocessed.csv'])
english_file = '/'.join([source_dir, target_month  + '_english.csv'])
french_file = '/'.join([source_dir, target_month  + '_french.csv'])
german_file = '/'.join([source_dir, target_month  + '_german.csv'])

'''
alltweets = pd.read_csv(tokenized_and_preprocessed_file, encoding='utf-8',
                                  usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                             'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                             'retweet_count', 'quoted_status_id_str', 'text_tokenized'])
'''

tweets_english = pd.read_csv(english_file, encoding='utf-8', 
                              usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                         'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                         'retweet_count', 'quoted_status_id_str', 'text_tokenized', 'text_processed'],
                              converters={"text_tokenized": literal_eval, "text_processed": literal_eval})

#tweets_by_language = pd.groupby(alltweets, 'lang').size().sort_values(ascending = False)
#print(tweets_by_language)

def remove_hash(wordlist):
    return(list(map(lambda x: re.sub(r'^#','',x), wordlist)))

def remove_at(wordlist):
    return(list(map(lambda x: re.sub(r'^@','',x), wordlist)))
    
tweets_english['text_wo_#'] = tweets_english['text_processed'].apply(lambda x: remove_hash(x))
tweets_english['text_wo_#@'] = tweets_english['text_wo_#'].apply(lambda x: remove_at(x))

X_train = tweets_english['text_wo_#@'].apply(lambda x: ' '.join(x))

stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])

additional_stopwords = set(['us'])
stopwords_filtered = list(additional_stopwords.union(stopwords_nltk.difference(relevant_words)))


print( '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3,4,5     *
******************************************************************************/
''')

for remove_stop_words in [stopwords_filtered, None]:
    print('\n\nStop words removed: {}\n*******************************'.format(remove_stop_words))
    for i in range(1,6):
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = str.split, 
                                    stop_words = remove_stop_words, max_features = 100000, ngram_range = (1,i))

        # sparse matrix
        words_matrix = vectorizer.fit_transform(X_train)
        vocabulary = vectorizer.get_feature_names()
        counts = words_matrix.sum(axis=0).A1
       
        word_counts_overall = pd.DataFrame({'word': vocabulary, 'count': counts})
           
        word_counts_for_max_ngram = word_counts_overall[word_counts_overall.word.apply(lambda c: len(c.split()) >= i)]
           
        word_counts_for_max_ngram_sorted = word_counts_for_max_ngram.sort_values(by='count', ascending=False)
        print('\nMost frequent ngrams for ngrams in range 1 - {}:'.format(i))
            
        print(word_counts_for_max_ngram_sorted[:40])
        if remove_stop_words != None:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_stopwords_removed.csv'
        else:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_with_stops.csv'
        word_counts_for_max_ngram_sorted.to_csv(filename)






print( '''
/******************************************************************************
*                     LDA 1-gram                                              *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = str.split, 
                                    stop_words = stopwords_filtered, max_features = 100000, ngram_range = (1,1))
words_matrix = vectorizer.fit_transform(X_train)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" % ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([str(round(model.components_[topic_idx,i])) for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
n_top_words = 30    
for n_topics in range(1,6):
    
    lda = LatentDirichletAllocation(n_topics = n_topics, random_state=0)
    lda.fit(words_matrix)
    print("\nTopics in LDA model:")
    feature_names = vectorizer.get_feature_names()
    print_top_words(lda, feature_names, n_top_words)

   
print( '''
/******************************************************************************
*                     LDA 2-grams                                             *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = str.split, 
                                    stop_words = stopwords_filtered, max_features = 100000, ngram_range = (1,2))

words_matrix = vectorizer.fit_transform(X_train)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print("  % ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([str(round(model.components_[topic_idx,i])) for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
n_top_words = 30    
for n_topics in range(1,6):
    
    lda = LatentDirichletAllocation(n_topics = n_topics, random_state=0)
    lda.fit(words_matrix)
    print("\nTopics in LDA model:")
    feature_names = vectorizer.get_feature_names()
    print_top_words(lda, feature_names, n_top_words)
  

print( '''
/******************************************************************************
*                     LDA 3-grams                                             *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = str.split, 
                                    stop_words = stopwords_filtered, max_features = 100000, ngram_range = (1,3))
words_matrix = vectorizer.fit_transform(X_train)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" % ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([str(round(model.components_[topic_idx,i])) for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
n_top_words = 30    
for n_topics in range(1,6):
    
    lda = LatentDirichletAllocation(n_topics = n_topics, random_state=0)
    lda.fit(words_matrix)
    print("\nTopics in LDA model:")
    feature_names = vectorizer.get_feature_names()
    print_top_words(lda, feature_names, n_top_words)
   


print( '''
/******************************************************************************
*    Co-Occurrence Matrix                                                     *
******************************************************************************/
''')

co_occurrence_matrix = words_matrix.T * words_matrix
co_occurrence_matrix.setdiag(0) 
array = co_occurrence_matrix.toarray()
co_occurrences = pd.DataFrame(array)
co_occurrences.shape

vocab = vectorizer.get_feature_names()
co_occurrences['word'] = vocab
co_occurrences = co_occurrences.set_index('word')
co_occurrences.head()

co_occurrences.columns = vocab
co_occurrences.head()





