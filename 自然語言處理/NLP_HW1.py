'''
名稱: NLP作業一 I3B32
姓名: Dino
日期: 2023/3/11
'''
import nltk
import string
import re
import numpy as np
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords          
from nltk.stem import PorterStemmer        
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')
nltk.download('stopwords')

def char_calculator(tweet):
    l = 0
    for i  in range(len(tweet)):
        l += len(tweet[i])
    return l

def text_process(tweet):
    #sub(目標, 代替字, 文本) ''前面加r防止轉意 (Ex:\\)
    tweet = re.sub(r'^RT[\s]+', '', tweet) #替換RT(retweet)開頭
    tweet = re.sub(r'https?://.*[\r\n]*', '', tweet)  #\r\n換行
    tweet = re.sub(r'#', '', tweet) #替換hashtag

    tokenizer = TweetTokenizer()
    tweet_tokenized = tokenizer.tokenize(tweet) #將句子拆分成一個一個單字

    stopwords_english = stopwords.words('english') 
    tweet_processsed = [word for word in tweet_tokenized if word not in stopwords_english and word not in string.punctuation] #將非停用詞和非標點符號的word提取出來

    stemmer = PorterStemmer()
    tweet_after_stem = []
    for word in tweet_processsed: #從提取過後的word中
        word = stemmer.stem(word) #將單字還原(處理ing, ed等)並轉換成小寫
        tweet_after_stem.append(word)
    return tweet_after_stem

def features_extraction(word_l , freqs_dict):
    x = np.zeros((1, 3)) #建立陣列 (1*3)
    x[0,0] = 1 
    for word in word_l: #將陣列儲存成 [1, 該字在pos dict出現次數, 該字在neg dict出現次數]
        try:
            x[0,1] += freqs_dict[(word,1)] #用try防止字不在字典裡
        except:
            x[0,1] += 0
        try: 
            x[0,2] += freqs_dict[(word,0.0)]
        except:
            x[0,2] += 0

    assert(x.shape == (1, 3))
    return x
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
test_pos = positive_tweets[4000:]
train_pos = positive_tweets[:4000]
test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

print("----------")
print("| 步驟一 |")
print("----------")
#步驟1 印出train和test的長度和前10個
print("positive testing :\n", test_pos[:10], "\nAll positive testing length :", char_calculator(test_pos))
print("--------------------")
print("positive training :\n", train_pos[:10], "\nAll positive training length :", char_calculator(train_pos))
print("--------------------")
print("negative testing :\n", test_neg[:10], "\nAll negative testing length :", char_calculator(test_neg))
print("--------------------")
print("negative training :\n", train_neg[:10], "\nAll negative training length :", char_calculator(train_neg))


train_pos_after_processed = [] #儲存處理後的資料
train_neg_after_processed = []
for word in train_pos:
    train_pos_after_processed.append(text_process(word))
for word in train_neg:
    train_neg_after_processed.append(text_process(word))

print("----------")
print("| 步驟二 |")
print("----------")
print("pos_tweets 處理前後差別 :") #步驟2 印出處理前後差別
for i in range(10):
    print("No.{}".format(i+1))
    print(train_pos[i])
    print(train_pos_after_processed[i])
print("--------------------")
print("neg_tweets 處理前後差別 :")
for i in range(10):
    print("No.{}".format(i+1))
    print(train_neg[i])
    print(train_neg_after_processed[i])


freq_pos = {} #python dictionary 類似其他語言的hash map
for i in range(len(train_pos_after_processed)):
    for word in train_pos_after_processed[i]:#將出現在pos_tweets的字一一累積
        if (word, 1) not in freq_pos:
            freq_pos[(word,1)] = 1
        else:
            freq_pos[(word,1)] = freq_pos[(word,1)] + 1

freq_neg = {}
for i in range(len(train_neg_after_processed)):
    for word in train_neg_after_processed[i]: #將出現在neg_tweets的字一一累積
        if (word, 0) not in freq_neg:
            freq_neg[(word,0)] = 1
        else:
            freq_neg[(word,0)] = freq_neg[(word,0)] + 1

freqs_dict = dict(freq_pos)
freqs_dict.update(freq_neg) #將兩個dict合併

print("----------")
print("| 步驟三 |")
print("----------")
print("positive frequent dictionary :") #步驟3 印出pos和neg頻率字典前10
i = 0
for key in freq_pos:
    if(i >= 10):break
    print(key, ":", freq_pos[key]) 
    i += 1
print("--------------------")
i = 0
print("negative frequent dictionary :")
for key in freq_neg:
    if(i >= 10):break
    print(key, ":", freq_neg[key]) 
    i += 1

train_x = train_pos + train_neg
X = np.zeros((len(train_x), 3)) #建立資料長度的陣列
for i in range(len(train_x)): #步驟4 製作Features Extraction模組
    X[i, :]= features_extraction(text_process(train_x[i]), freqs_dict)


print("----------")
print("| 步驟五 |")
print("----------")
for i in range(10): #步驟5 training data列印
    print("Training date 第{}筆資料".format(i+1))
    print("原始資料 :", train_x[i])
    print("經過處理 :", text_process(train_x[i]))
    print("特徵擷取 :", X[i], "\n")