
import time
start_time = time.time()


import numpy as np
import pandas as pd
import pickle as p

from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from tqdm import tqdm
from itertools import islice


#Подготовим нужные функции для лемматизатора
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.NOUN

from nltk import WordNetLemmatizer
def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                    for word, tag in pos_tagged])




#Это для уменьшения словаря слов
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
sw0 = np.array(['.',',','!','?','`','\'','[',']','``','@','#','№','$','^','*','-','&','\\',';',':','>','<'])
sw1 = np.array(list(set(stopwords.words('english'))))
sw2 = np.append(sw1, sw0)
nltk.download('punkt')




data = pd.read_csv(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\data.csv')

with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\vecs.pkl', 'rb') as file:
    vecs = p.load(file)
with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\wv.pkl', 'rb') as file:
    wv = p.load(file)
with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\data_x.pkl', 'rb') as file:
    data_x = p.load(file)


def find_word_docs(word):
    global data
    docs = np.array([], dtype=np.int64)
    for i in data.index:
        try:
            if((word in data['Lyric_clear'][i]) or (word in data['Artist_clear'][i])):
                docs = np.append(docs, i)
        except: 
            pass
    return docs












x = pd.DataFrame()

def score(document):
    global x
    return x['rel'][document]

def retrieve(query):
    global sw2, vecs, wv, data_x, x
    query_tok = [i.lower() for i in word_tokenize(my_lemmatizer(query)) if i not in sw2]
    query_docs = np.array(list(map(find_word_docs, query_tok)))
    docs_itog = np.array(list(set(query_docs[0]).intersection(*query_docs[1:])))

    zero_lyric = sum(vecs.values()) / len(vecs)
    dim = 300
    tfidf_lyric = TfidfVectorizer()
    query_lyric_vec = tfidf_lyric.fit_transform([' '.join(query_tok)])
    vocab_lyric = np.zeros((len(tfidf_lyric.vocabulary_.keys()), dim))
    for key in tqdm(tfidf_lyric.vocabulary_.keys()):
        vocab_lyric[tfidf_lyric.vocabulary_[key]] = vecs.get(key, zero_lyric)
    query_lyric_x = pd.Series((query_lyric_vec.dot(vocab_lyric))[-1])

    dim = 100
    query_artist_vec = tfidf_lyric.fit_transform([' '.join(query_tok)])
    vocab_artist = np.zeros((len(tfidf_lyric.vocabulary_.keys()), dim))
    for key in tqdm(tfidf_lyric.vocabulary_.keys()):
        try:
            vocab_artist[tfidf_lyric.vocabulary_[key]] = wv[key]
        except:
            vocab_artist[tfidf_lyric.vocabulary_[key]] = np.zeros(dim)
    query_artist_x = pd.Series((query_artist_vec.dot(vocab_artist))[-1])

    x = pd.DataFrame(index=docs_itog)
    x['Song'] = data_x[300].iloc[docs_itog] 
    x['Song'] = x['Song'] / x['Song'].max() * 2
    x['Popularity'] = data_x[301].iloc[docs_itog]
    x['Popularity'] = x['Popularity'] / x['Popularity'].max() * 2
    x['cosdist_lyric'] = (cosine_distances(data_x.iloc[docs_itog,:300],[query_lyric_x.values]) - 2) * -1
    x['cosdist_artist'] = (cosine_distances(data_x.iloc[docs_itog,302:],[query_artist_x.values]) - 2) * -1

    k = np.array([ 1.55735732, 0.33114107, 13.83990411, 1.6114938])

    x['rel'] = x.dot(k) 
    x['rel'] = x['rel'] / x['rel'].max()
    x = x.sort_values(by=['rel'], ascending=False)
    
    return x.index[:10]



print("--- %s seconds ---" % (time.time() - start_time))




