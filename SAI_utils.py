# -*- coding: utf-8 -*-
# Authors : Hanna Julienne

"""
preprocessing function for the Natural language processing competitions:
"""
import pandas as pd
import nltk
import seaborn as sns
import plotly

import sklearn.feature_extraction.text as sktxt
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

my_steamy = LancasterStemmer()
my_lemmy = WordNetLemmatizer()
my_lemmy.lemmatize("fumbling","v")

# detect words them
word_tokenizer = nltk.RegexpTokenizer(r"\w+")
# detect ponctuation sign (might be a stylistic element)
punct_tokenizer = nltk.RegexpTokenizer(r"[,;.:!?]+")
def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def lemm_wth_pos (x,y):
    wn_t = penn2morphy(y[:2])
    if wn_t != '':
        return my_lemmy.lemmatize(x, wn_t)
    else:
        return my_lemmy.lemmatize(x)

def preprocessing_wth_lemmatizer(sentence):
    tokens = word_tokenizer.tokenize(sentence)
    tok_wth_tag = nltk.pos_tag(tokens)
    return(' '.join([lemm_wth_pos(t[0], t[1]) for t  in tok_wth_tag]))

def preprocessing_wth_stemmer(sentence):
    tokens = word_tokenizer.tokenize(sentence)
    # filter stop words :
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    print tokens
    res =(' '.join([my_steamy.stem(t) for t  in tokens]))
    return(' '.join([my_steamy.stem(t) for t  in tokens]))
