import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer


tokenized_file = 'data/tweets_tokenized.csv'
english_file = 'data/tweets_english.csv'

alltweets_tokenized = pd.read_csv(tokenized_file, encoding='utf-8',
                                  usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                             'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                             'retweet_count', 'quoted_status_id_str', 'text_tokenized'])
tweets_english = pd.read_csv(english_file, encoding='utf-8',
                              usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                         'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                         'retweet_count', 'quoted_status_id_str', 'text_tokenized'])

tweets_by_language = pd.groupby(alltweets_tokenized, 'lang').size().sort_values(ascending = False)
#print tweets_by_language

X_train = tweets_english['text_tokenized']
X_train = X_train.apply(lambda t: t.replace(',',''))
stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))

print '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3,4,5     *
******************************************************************************/
'''

for remove_stop_words in [stopwords_filtered]:
#for remove_stop_words in [stopwords_filtered, None]:
    print '\n\nStop words removed: {}\n*******************************'.format(remove_stop_words)    
    for i in range(1):
    #for i in range(1,6):
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
                                    stop_words = remove_stop_words, max_features = 10000, ngram_range = (1,i+1))

        # sparse matrix
        words_array = vectorizer.fit_transform(X_train).toarray()

        vocabulary = vectorizer.get_feature_names()
        #print vocabulary[0:10]
        #print vectorizer.vocabulary_.get('able')

        counts = np.sum(words_array, axis=0)
        word_counts_overall = pd.DataFrame({'word': vocabulary, 'count': counts})
           
        word_counts_for_max_ngram = word_counts_overall[word_counts_overall.word.apply(lambda c: len(c.split()) >= i+1)]
           
        word_counts_for_max_ngram_sorted = word_counts_for_max_ngram.sort_values(by='count', ascending=False)
        print '\nMost frequent ngrams for ngrams in range 1 - {}:'.format(i)
            
        print word_counts_for_max_ngram_sorted[:100]
        
        
print '''
/******************************************************************************
*    Co-Occurrence Matrix     *
******************************************************************************/
'''

topics = ['oracle', 'windows', 'cloud', 'sql', 'orclapex', ]
topics_hashtags = map(lambda s: '#'+s, topics)
topics_with_hashtags = topics + topics_hashtags
     
people = ['@sfonplsql']    

vendors = ['oracle']
vendors_at = map(lambda s: '@'+s, vendors)
vendors_with_at = vendors + vendors_at
     
sentiment_words = ['great', 'good', 'like', 'thanks']

# do whole co-occurrence matrix
 


print '''
/******************************************************************************
*    Sentiment Analysis nltk                                                  *
******************************************************************************/
'''


print '''
/******************************************************************************
*    Sentiment Analysis unsupervised                                          *
******************************************************************************/
'''


