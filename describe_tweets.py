import pandas as pd
import numpy as np

import config
import operator
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


print( '''
/******************************************************************************
*                     LDA                                                     *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, token_pattern = '\S+',
                                    stop_words = stopwords_filtered, max_features = 100000, ngram_range = (1,1))
words_matrix = vectorizer.fit_transform(X_train)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([str(round(model.components_[topic_idx,i])) for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
n_top_words = 30    
for n_topics in range(1,5):
    
    lda = LatentDirichletAllocation(n_topics = n_topics, random_state=0)
    lda.fit(words_matrix)
    print("\nTopics in LDA model:")
    feature_names = vectorizer.get_feature_names()
    print_top_words(lda, feature_names, n_top_words)
    
'''
Topics in LDA model:
Topic #0:
! ? oracle new data via not great #oracle cloud thanks but get good time see one today sql 2 like no i'm :) us day it's database use 1
69253.0 40089.0 24416.0 19631.0 14218.0 14126.0 12646.0 10567.0 9362.0 9347.0 8923.0 8824.0 7594.0 7462.0 7457.0 7311.0 7254.0 7125.0 7034.0 6414.0 6329.0 6260.0 6155.0 6151.0 6111.0 6099.0 6089.0 6016.0 5998.0 5938.0


Topics in LDA model:
Topic #0:
! ? not great thanks but good one time like no i'm :) day it's us see :-) today don't get week know #wpc16 very #iamcp people next looking thank
69251.0 24781.0 12640.0 10566.0 8922.0 8823.0 7462.0 7248.0 7125.0 6328.0 6259.0 6154.0 6151.0 6098.0 6089.0 6077.0 5921.0 5832.0 5366.0 5357.0 5142.0 4923.0 4563.0 4451.0 4351.0 4019.0 3732.0 3659.0 3610.0 3595.0
Topic #1:
oracle new ? data via #oracle cloud sql database blog business use using #cloud #bigdata big learn 12c 1 available video post windows service 10 2 partner check – |
24415.0 16514.0 15307.0 14217.0 14107.0 9361.0 9346.0 7033.0 6015.0 5863.0 5720.0 5664.0 5660.0 5321.0 5236.0 4972.0 4908.0 4672.0 4468.0 4287.0 4280.0 4111.0 3910.0 3738.0 3704.0 3661.0 3605.0 3444.0 3401.0 3306.0


Topics in LDA model:
Topic #0:
! great thanks i'm :) day :-) today next week via #iamcp us rt see #wpc16 very #orclapex join good new year “ looking thank ” nice last ;-) event
69252.0 10566.0 8922.0 6154.0 6150.0 6098.0 5831.0 5827.0 5018.0 4923.0 4785.0 4543.0 4521.0 4487.0 4476.0 4451.0 4351.0 4271.0 4140.0 4091.0 3796.0 3751.0 3686.0 3610.0 3594.0 3557.0 3475.0 3281.0 3247.0 3237.0
Topic #1:
? not but like no know one #bigdata big use need time it's " data get don't help way only good work % want via make $ people think better
40071.0 12645.0 8823.0 6310.0 6259.0 5695.0 5295.0 5231.0 4892.0 4832.0 4740.0 4342.0 4313.0 4303.0 4286.0 3693.0 3668.0 3547.0 3503.0 3395.0 3371.0 3221.0 3192.0 3175.0 3061.0 3015.0 2964.0 2943.0 2913.0 2822.0
Topic #2:
oracle new data #oracle cloud sql via database blog business #cloud using learn 12c 1 available post windows service 2 partner video – check | analytics microsoft 10 performance bi
24415.0 14524.0 9932.0 9361.0 9346.0 7033.0 6279.0 6015.0 5863.0 5720.0 5321.0 5182.0 4908.0 4671.0 4336.0 4287.0 4111.0 3909.0 3738.0 3732.0 3582.0 3534.0 3401.0 3328.0 3306.0 3110.0 3027.0 2983.0 2861.0 2794.0


Topics in LDA model:
Topic #0:
! great thanks today i'm day :-) next via week us rt #iamcp #wpc16 join very see #orclapex free year thank session :) nice last ;-) event 2015 new stories
69252.0 10566.0 8922.0 7124.0 6154.0 6098.0 5831.0 5018.0 4926.0 4923.0 4633.0 4498.0 4498.0 4451.0 4359.0 4351.0 4310.0 4271.0 4097.0 3754.0 3594.0 3536.0 3532.0 3475.0 3281.0 3253.0 3238.0 3036.0 2991.0 2960.0
Topic #1:
? not but good one like no use know it's need time " make only don't work people get way want open well really $ think right would yes still
40088.0 12645.0 8823.0 7461.0 7253.0 6328.0 6259.0 5998.0 5695.0 4860.0 4740.0 4652.0 4303.0 4223.0 4087.0 3814.0 3748.0 3732.0 3599.0 3504.0 3468.0 3154.0 3126.0 3021.0 2964.0 2913.0 2801.0 2792.0 2680.0 2648.0
Topic #2:
oracle new #oracle cloud via sql database blog business check 12c video post 5 service partner 4 using bi customer | @oracle w available services partners http pl 1 [
24415.0 10763.0 9361.0 9346.0 9200.0 7033.0 6015.0 5863.0 5720.0 5154.0 4671.0 4280.0 4111.0 3926.0 3738.0 3606.0 3216.0 2828.0 2794.0 2787.0 2681.0 2668.0 2602.0 2392.0 2350.0 2331.0 2279.0 2261.0 2247.0 2151.0
Topic #3:
data new #cloud #bigdata big learn read windows 10 “ looking ” – live 2 microsoft performance using part app + things java 1 apps 3 book love # #iot
14217.0 5878.0 5321.0 5236.0 4973.0 4908.0 4381.0 3909.0 3867.0 3687.0 3610.0 3557.0 3401.0 3316.0 3073.0 3027.0 2861.0 2832.0 2792.0 2767.0 2762.0 2702.0 2610.0 2551.0 2405.0 2320.0 2307.0 2272.0 2146.0 2141.0



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


 



print( '''
/******************************************************************************
*    Sentiment Analysis unsupervised                                          *
******************************************************************************/
''')


