from nltk import tokenize
from operator import itemgetter
import math
from tqdm import tqdm
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



def create_corpus(a):
    #print('==',a)
    corpus = []
    ds_count = len(a)
    for i in range(0, ds_count):
        # Remove punctuation
        text = re.sub('[^a-zA-Z]', ' ', str(a[i]))

        # Convert to lowercase
        text = text.lower()

        # Remove tags
        #text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

        # Remove special characters and digits
        #text=re.sub("(\\d|\\W)+"," ",text)

        # Convert to list from string
        text = text.split()

        # Stemming
        ps=PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
        return corpus

## frequency based extraction
def freq_keyword(corpus):

    # View most frequently occuring keywords
    def get_top_n_words(corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                       vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                           reverse=True)
        return words_freq[:n]

    # Convert most freq words to dataframe for plotting bar plot, save as CSV
    top_words = get_top_n_words(corpus, n=20)
    top_df = pd.DataFrame(top_words)
    top_df.columns=["Keyword", "Frequency"]
    
    return top_df


## Load data

df = pd.read_csv('/home/amansinha/extractpubmed/timeline/tdm1_all.csv')
df = df.drop_duplicates()
df = df.drop(columns=['Unnamed: 0'], axis=1)
df['year'] = df['publication_date'].apply(lambda x: x.split('-')[0])



df1 = df.groupby('year')['abstract'].apply(list).reset_index(name='year_abstract')



all_keys = []
all_freq = []
for y,a in zip(df1['year'],df1['year_abstract']):
    #if len(a) ==1 : continue
    print(y,len(a))
    corpus = create_corpus(a)
    #print(corpus)
    df2 = freq_keyword(corpus)
    #print(df2)
    all_keys.append(df2['Keyword'])
    all_freq.append((*df2['Frequency'].tolist(), y))
    
    #print(tfidf_keyword(corpus,a))
cols = [f'#{i+1}' for i in range(20)]+['year']
cols
df2 = pd.DataFrame(all_freq, columns=[f'#{i+1}' for i in range(20)]+['year'])    


columns = df2['year'].tolist()
df2.drop('year', axis=1, inplace=True)
df3 = df2.values


# 3D

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

rows = [f'#{i+1}' for i in range(20)]
lx = len(df3[0])
ly = len(df3[:,0])

xpos = np.arange(0,lx,1)#list(range(20))#df2['Keyword'].tolist()#
ypos = np.arange(0,ly,1)#[2020]*10 + [2019]*10

xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

"""Setting x, y and z positions"""
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(lx*ly)

#print(list(zip(xpos,ypos,zpos)))
#print(len(xpos), len(ypos), len(zpos))


dx = 0.25*np.ones_like(zpos)
dy = dx.copy()
dz = df3.flatten()#np.zeros(20*20)#list(range(2000,2020))

#print(len(dx), len(dy), len(dz))
#print(dx,dy,dz)
colors = ['red', 'yellow', 'brown', 'cyan', 
          'green', 'pink', 'orange', 'blue',
          'purple', 'gray']#, 'black', 'white',
          #'magenta', 'darkred']*ly
ax1.bar3d(xpos, ypos, zpos, dx, dy,dz, color=colors)#'red')

"""Set the x and y ticks"""
ax1.w_xaxis.set_ticklabels(columns)
ax1.w_yaxis.set_ticklabels(rows)

#ax1.bar3d(xpos[:10], ypos[:10], zpos[:10], dx[:10], dy[:10],dz[:10], color='r')
#ax1.bar3d(xpos[10:], ypos[10:], zpos[10:], dx[10:], dy[10:],dz[10:], color='b')
ax1.set_xlabel('year')
ax1.set_ylabel('Keywords')
ax1.set_zlabel('Freq')

plt.show()

