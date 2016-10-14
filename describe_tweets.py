import pandas as pd
import numpy as np

import config
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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

stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])

stopwords_filtered = list(stopwords_nltk.difference(relevant_words))

stopwords_twitter = set(['via'])


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
            
        print(word_counts_for_max_ngram_sorted[:40])
        if remove_stop_words != None:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_stopwords_removed.csv'
        else:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_with_stops.csv'
        word_counts_for_max_ngram_sorted.to_csv(filename)

'''




print( '''
/******************************************************************************
*                     LDA 1-gram                                              *
******************************************************************************/
''')
'''
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
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

'''    
print( '''
/******************************************************************************
*                     LDA 2-grams                                             *
******************************************************************************/
''')
'''
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
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
'''    

print( '''
/******************************************************************************
*                     LDA 3-grams                                             *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
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

def get_co_occurrences(record):
    wordlist = record.split()
    for i in range(len(wordlist) - 1):
        for j in range(i+1, len(wordlist)):    
            w1, w2 = sorted([wordlist[i], wordlist[j]]) 
            if w1 != w2:
                co_occurrences[w1][w2] += 1
                co_occurrences[w2][w1] += 1
                

# co_occurrences, by word, ordered alphabetically
# each word appears as key
co_occurrences = defaultdict(lambda : defaultdict(int))
X_train[:100].apply(get_co_occurrences) 
        
co_occurrences_df = pd.DataFrame.from_dict(co_occurrences)   
print(co_occurrences_df[:10])

# for every word, most frequent co-occurring words
co_occurrences_by_freq = {word: sorted(co_occurrences[word].items(), key = lambda k_v: k_v[1], reverse=True) for word in co_occurrences}
'''
print('co_occurrences ordered by frequency for each word:')
i=0
for word in co_occurrences_by_freq:
    i = i+1
    if i > 9: break
    print (word)
    for tup in co_occurrences_by_freq[word]: print(tup[0] + ': ' + str(tup[1]))
'''

# co_occurrences sorted by highest value per word
co_occurrences_by_highest_freq = sorted(co_occurrences_by_freq.items(), key = lambda k_v: k_v[1][0][1], reverse=True)
'''
print('co_occurrences sorted by highest value per word:')
print(co_occurrences_by_highest_freq[:10])
'''

topics = ['oracle', 'windows', 'cloud', 'sql', 'orclapex']
topics_hashtags = map(lambda s: '#'+s, topics)
topics_with_hashtags = topics + topics_hashtags
     
people = ['@sfonplsql']    

vendors = ['oracle']
vendors_at = map(lambda s: '@'+s, vendors)
vendors_with_at = vendors + vendors_at
     
sentiment_words = ['great', 'good', 'like', 'thanks']



