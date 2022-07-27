import nltk
import io
import re
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
from nltk.corpus import brown
import requests
nltk.download('popular')
import string 
import numpy as np
from PIL import Image
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from textblob import TextBlob
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nrclex import NRCLex
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from glob import glob

path = r'C:\Users\delta\AppData\Local\Programs\Python\Python37\nlp\sentiment_spw\parts'

chapters = []

def readchap(folder):
    files = glob(folder+"/*.txt") # read all text files
    for i in files :
        infile = open(i,"r",encoding="utf-8")
        data = infile.read()
        infile.close()
        chapters.append([data])
    return chapters

readchap(path)

chapter_name = ["I", "II", "III", "IV", "V", "VI",
            "VII", "VIII", "IX", "X", "XI"]

# text
spw_df = pd.DataFrame(data=[k for k in chapters], columns=['text'])
spw_df = spw_df.replace('\n',' ', regex=True)

# chapter name
chapter_name = pd.DataFrame(chapter_name, columns=['chapter'])
spw_df = pd.merge(spw_df, chapter_name, left_index=True, right_index=True)

# stopwords
sw = nltk.corpus.stopwords.words('english')
with open('clean_stop.txt','r') as f:
    newStopWords = f.readlines()
sw.extend(newStopWords)
pers = ['el','ibn','n','would','could','abd','aba','u']
sw.extend(pers)

# contractions
with open('clean_map.txt','r') as f:
    mapping = f.readlines()

# clean (from https://github.com/072arushi/Movie_review_analysis)
punct = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punct))
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in sw])
def word_replace(text):
    return text.replace('<br />','')
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
def preprocess(text):
    text=clean_contractions(text,mapping)
    text=text.lower()
    text=word_replace(text)
    text=remove_urls(text)
    text=remove_html(text)
    text=remove_stopwords(text)
    text=remove_punctuation(text)
    text=lemmatize_words(text)
    return text

# save to dataframe
spw_df['clean_text'] = spw_df['text'].apply(lambda text: preprocess(text))
spw_df = spw_df.replace('\n',' ', regex=True)

# save cleaned chapters 
chapters_clean = []
for i in range(11):
    chapters_clean.append(spw_df.loc[i,'clean_text'])

# densities def
def lexical_density(text):
    return len(set(text)) / len(text)

# get lexical density
dens = []
for i in chapters_clean:
    tokens = RegexpTokenizer(r'\w+').tokenize(i)    
    lex_dens = lexical_density(tokens)
    dens.append(lex_dens)
lex_dens = pd.DataFrame(dens) 
lex_dens = lex_dens.rename(columns={0: 'lex_dens'})

# get lexical ensity normalized to shortest cleaned text length (6238)
dens_norm = []
for i in chapters_clean:
    tokens = RegexpTokenizer(r'\w+').tokenize(i)
    tokens = tokens[0:6238]     
    lex_dens_norm = lexical_density(tokens)
    dens_norm.append(lex_dens_norm)  
lex_dens_norm = pd.DataFrame(dens_norm) 
lex_dens_norm = lex_dens_norm.rename(columns={0: 'lex_dens_norm'})

#########################################################################
# clean text data
#########################################################################
# char count
spw_df['char_count']=spw_df['clean_text'].str.len()
# word count
spw_df['word_count']=spw_df['clean_text'].apply(lambda x: len(str(x).split()))
# unique count
spw_df['unique_count']=spw_df['clean_text'].apply(lambda x: len(set(str(x).split())))
# average word length
spw_df['avg_word_len']=spw_df['clean_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# lexical density
spw_df = pd.merge(spw_df, lex_dens, left_index=True, right_index=True)
# lexical density normalized
spw_df = pd.merge(spw_df, lex_dens_norm, left_index=True, right_index=True)
# convert to and save as csv
spw_df.to_csv(r'spw_df.csv')
