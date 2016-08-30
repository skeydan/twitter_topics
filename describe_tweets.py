import pandas as pd
import numpy as np

import config
import operator
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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
# remove RT clause
X_train = X_train.apply(lambda t: re.sub(r'[RT, .* :,', , t)))
# remove separator and start/end tokens of nltk tokenizer
X_train = X_train.apply(lambda t: t.replace(',',''))
X_train = X_train.apply(lambda t: t.replace('[',''))
X_train = X_train.apply(lambda t: t.replace(']',''))
# other "useless" stuff


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
            
        print(word_counts_for_max_ngram_sorted[:40])
        if remove_stop_words != None:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_stopwords_removed.csv'
        else:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_with_stops.csv'
        word_counts_for_max_ngram_sorted.to_csv(filename)
'''

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
: . rt ! - " ? oracle ... … ( ) new / & not great via data good today #oracle cloud but ' thanks day one sql time
43147.0 34539.0 27811.0 17747.0 12238.0 9532.0 9424.0 6139.0 5371.0 5292.0 4868.0 4814.0 4327.0 3773.0 3742.0 3162.0 2935.0 2591.0 2512.0 2251.0 2212.0 2172.0 2122.0 2097.0 2059.0 2029.0 2013.0 1899.0 1869.0 1863.0


Topics in LDA model:
Topic #0:
. ! : rt ? ) ( ... - great … / not good today thanks day but sql :) see time :-) one new oracle @sfonplsql #orclapex get i'm
28150.0 17747.0 14602.0 11564.0 6026.0 4755.0 4750.0 3309.0 3214.0 2860.0 2677.0 2589.0 2335.0 2229.0 2205.0 2029.0 2012.0 1903.0 1868.0 1789.0 1768.0 1679.0 1673.0 1651.0 1533.0 1526.0 1512.0 1479.0 1417.0 1380.0
Topic #1:
: rt " - . oracle ? new … & via data cloud ... #oracle ’ @oracle blog video / business windows ' using #cloud customer bi | not service
28545.0 16247.0 9531.0 9024.0 6389.0 4613.0 3398.0 2793.0 2616.0 2597.0 2588.0 2491.0 2121.0 2062.0 1631.0 1338.0 1306.0 1224.0 1204.0 1184.0 1131.0 1097.0 1096.0 969.0 904.0 886.0 878.0 850.0 826.0 772.0


Topics in LDA model:
Topic #0:
! ( ) - : rt data :) great 2015 2016 thanks pm + thank big area entered @panicc @lepetitbouton87 @mary871202 i'm c 10 team happy . conference love year
17730.0 4868.0 4813.0 3495.0 2171.0 2097.0 1677.0 1531.0 1505.0 1406.0 1248.0 1193.0 1143.0 1128.0 987.0 978.0 918.0 860.0 847.0 846.0 846.0 811.0 761.0 760.0 752.0 669.0 667.0 626.0 613.0 609.0
Topic #1:
: rt - oracle & cloud via ’ new @oracle #oracle video … business #opn looking #cloud customer | data windows partner “ ” service #oow15 ... w live –
15408.0 8300.0 6419.0 3214.0 2205.0 2121.0 1661.0 1461.0 1459.0 1306.0 1215.0 1204.0 1148.0 1125.0 975.0 906.0 903.0 873.0 850.0 832.0 790.0 758.0 730.0 691.0 667.0 654.0 643.0 595.0 595.0 574.0
Topic #2:
. : rt " ? ... … / not oracle new - good but ' day one sql get time today no 2 @sfonplsql #orclapex like great use :-) don't
33870.0 25569.0 17414.0 9531.0 9080.0 4270.0 4144.0 3301.0 3161.0 2924.0 2689.0 2324.0 2250.0 2096.0 2054.0 2012.0 1898.0 1868.0 1856.0 1710.0 1663.0 1603.0 1545.0 1512.0 1479.0 1468.0 1429.0 1421.0 1377.0 1332.0


Topics in LDA model:
Topic #0:
. ! : rt ? new great ... data not thanks day see :) no like but #orclapex get one time it's know today don't first using year big 2
23441.0 17727.0 6704.0 5809.0 5373.0 4326.0 2934.0 2778.0 2510.0 2157.0 2028.0 2012.0 1780.0 1762.0 1603.0 1573.0 1543.0 1477.0 1416.0 1394.0 1360.0 1333.0 1328.0 1292.0 1234.0 1214.0 1200.0 1079.0 978.0 967.0
Topic #1:
" - . ? ’ :-) next video : need still looking | not read via “ code ” better yes — right forward but cool ; one #braincandy got
9531.0 4857.0 3582.0 2047.0 1744.0 1672.0 1361.0 1204.0 1193.0 1175.0 923.0 920.0 849.0 837.0 824.0 772.0 730.0 706.0 691.0 688.0 621.0 602.0 587.0 565.0 553.0 528.0 521.0 443.0 430.0 419.0
Topic #2:
: rt oracle … . - & good #oracle cloud ? @oracle 1 ' week business database #opn live #cloud ... customer bi service partner learn morning # check #oow15
26112.0 16554.0 6138.0 3906.0 3415.0 3241.0 2708.0 2250.0 2171.0 2121.0 1905.0 1306.0 1250.0 1226.0 1222.0 1131.0 1122.0 974.0 929.0 903.0 890.0 886.0 878.0 771.0 758.0 745.0 724.0 717.0 703.0 686.0
Topic #3:
: rt ( ) . / - via ... sql i'm us @sfonplsql … + windows very thank w pl 3 #plsql post free c http @ would 10 blog
9139.0 5444.0 4868.0 4813.0 4101.0 3773.0 3592.0 1818.0 1703.0 1588.0 1380.0 1359.0 1270.0 1237.0 1129.0 1096.0 1076.0 988.0 965.0 958.0 949.0 898.0 898.0 836.0 789.0 788.0 786.0 783.0 762.0 728.0


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
*    Sentiment Analysis nltk                                                  *
******************************************************************************/
''')


print( '''
/******************************************************************************
*    Sentiment Analysis unsupervised                                          *
******************************************************************************/
''')


