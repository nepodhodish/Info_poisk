import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import nltk
import pickle as p




'''
#Подготовим нужные функции для лемматизатора
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import pos_tag


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
    global o
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








def do_clear(state):
    global sw2
    return ' '.join([i.lower() for i in word_tokenize(my_lemmatizer(state)) if i not in sw2])
'''
def find_word_docs(word):
    global data
    docs = np.array([])
    for i in data.index:
        if(word in data['Lyric_clear'][i]):
            docs = np.append(docs, i)
    return docs

data = pd.read_csv(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\data.csv')
    
with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\vocabulary.pkl', 'rb') as file:
    vocabulary = p.load(file)

def main():
    #global data_artists, data_lyrics, 
    global data, sw0
    global vocabulary

    

    
    '''
    №1
    pool = mp.Pool(12) 
    data_artists = pd.read_csv(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\artists-data.csv')
    data_lyrics = pd.read_csv(r'C:/Users/Anton/Desktop/проектыPython/ml/lesson9/домашка/lyrics-data.csv')

    обрабатывам 2 файла в один

    data = data_lyrics[data_lyrics['Idiom'] == 'ENGLISH']
    data = data.drop(['Idiom','SLink'], axis=1)

    data['Popularity'] = pool.map(Change_pop,data['ALink'])
    data['Songs'] = pool.map(Change_song,data['ALink'])
    data['ALink'] = pool.map(Change_artist,data['ALink'])
    data = data.dropna(axis=0).reset_index(drop=True)
    data.to_csv(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\data.csv', index=False)
    '''



    '''
    №2
    pool = mp.Pool(12) 
    #обрабатываем песню 
    data['Lyric_clear'] = pool.map(do_clear,data['Lyric'])
    data.to_csv(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\data.csv', index=False)
    #Мне понадобилось всего  370.12341237068176 seconds, чтобы перебрать такой большой массив, это чудечно!
    '''

    '''
    создание инвертированного индекса
    №3
    pool = mp.Pool(12) 
    Очень долго 
    reverse_index = pd.Series(list(pool.map(find_word_docs, vocabulary[:1000])), index=vocabulary[:1000])

    with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\reverse_index.pkl', 'wb') as file:
        p.dump(reverse_index,file)
    

    Решение проблемы
    ищем документы для запроса на ходу
    
    pool = mp.Pool(4) 
    start_time = time.time()
    reverse_index = pd.Series(list(pool.map(find_word_docs, vocabulary[:15])), index=vocabulary[:15])
    print("--- %s seconds ---" % (time.time() - start_time))

    with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\reverse_index.pkl', 'wb') as file:
        p.dump(reverse_index,file)
    '''

    pool = mp.Pool(4) 

    pool.close()




'''
def Change_pop(link):
    global data_artists, data_lyrics, data
    try:
        itog = data_artists[data_artists['Link'] == link]['Popularity'].values[0]
        return itog
    except:
        pass

def Change_song(link):
    global data_artists, data_lyrics, data
    try:
        itog = data_artists[data_artists['Link'] == link]['Songs'].values[0]
        return itog
    except:
        pass

def Change_artist(link):
    global data_artists, data_lyrics, data
    try:
        itog = data_artists[data_artists['Link'] == link]['Artist'].values[0]
        return itog
    except:
        pass


'''


if(__name__ == '__main__'):
    main()
