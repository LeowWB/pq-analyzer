import sys
sys.path.extend(['D:\\Documents\\GitHub\\pq-analyzer\\env', 'D:\\Documents\\GitHub\\pq-analyzer\\env\\lib\\site-packages'])

import pandas as pd
import numpy as np
import pickle
from ast import literal_eval

from pdb import set_trace as st
import string
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def _get_stopwords():
    print('getting stopwords')
    gin_stopwords = ['about', 'actually', 'ago', 'agree', 'already', 'also', 'always',
                     'am', 'because', 'bring', 'can', 'cannot', 'cant',
                     'could', 'couldnt', 'did', 'didnt', 'different', 'dont',
                     'eg', 'ei', 'etc', 'even', 'every', 'everywhere', 'example', 'feel', 'felt', 'get',
                     'go', 'good', 'hence', 'however', 'ie', 'im', 'important', 
                     'know', 'like', 'make', 'may', 'maybe', 'mid', 'need', 'no', 'not', 'often', 'oh', 'ok',
                     'one', 'per', 'pre', 'put', 'quite', 'rather', 'really', 'reduce', 'say', 'singapore', 
                     'something', 'su', 'space',
                     'still', 'sum', 'th', 'that', 'thats', 'they', 'theyre', 'theyve',
                     'think', 'three', 'total', 'try', 'us', 'use', 've', 'via', 'wa', 'want', 'way', 'we',
                     'well', 'were', 'weve', 'whatsoever', 'whether', 'wont',
                     'would', 'wouldnt', 'yall', 'yea', 'yeah', 'yes', 'yet',
                     'youd', 'youll', 'youre', 'youve', 'yr']

    honorific_stopwords = ['mrs', 'mr', 'ms', 'miss', 'mdm', 'madam', 'mister', 'er', 'assoc', 'prof', 'dr', 'bg']

    number_stopwords = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'last', 'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'trillion']

    # "ministry" is a stopword because, if someone asks about ministry of education, then he's interested in education. so we dunnid the "ministry".
    hansard_stopwords = ['speaker', 'leader', 'chairman', 'chairwoman', 'chair', 'deputy', 'beg', 'permission', 'move', 'withdraw', 'allocate', 'head', 'cut', 'thank', 'member', 'honorable', 'question', 'clarification', 'grateful', 'year', 'month', 'day', 'reply', 'many', 'ministry', 'minister', 'consider', 'take', 'ask', 'sir', 'maam', 'since', 'especially', 'reason', 'see', 'non', 'due', 'mention', 'respectively', 'respective', 'must', 'expect', 'mean', 'exist', 'cause', 'hon']

    adverb_stopwords = ['annually', 'currently', 'early', 'finally', 'firstly', 'fully', 'increasingly', 'likely', 'monthly', 'particularly', 'previously', 'secondly', 'specifically']

    return gin_stopwords + honorific_stopwords + hansard_stopwords + number_stopwords + adverb_stopwords
    
stop_words.extend(_get_stopwords())

# Saving stopwords and phrases into files
with open('stopwords.pkl', 'wb') as f:
    pickle.dump(stop_words, f)
    
    
    
    
    

def lemmatize(lemmatizer, word):
    for pos in ('a', 'n', 'v', 'r', 's'):
        word = lemmatizer.lemmatize(word, pos=pos)
    return word

def clean_text(text, stop_words = stop_words):
    words_to_sub = {
        'ministry of communications and information': 'mci',
        'ministry of culture community and youth': 'mccy',
        'ministry of defense': 'mindef',
        'ministry of education': 'moe',
        'ministry of finance': 'mof',
        'ministry of foreign affairs': 'mfa', 
        'ministry of health': 'moh',
        'ministry of home affairs': 'mha',
        'ministry of law': 'minlaw',
        'ministry of manpower': 'mom',
        'ministry of national development': 'mnd',
        'ministry of social and family development': 'msf',
        'ministry of sustainability and the environment': 'mse',
        'ministry of trade and industry': 'mti',
        'ministry of transport': 'mot',
        'prime ministers office': 'pmo'
    }

    lemmatizer = WordNetLemmatizer()
    
    if text != text: # np.nan
        return np.nan
    
    else:
        text = str(text)
        text = text.lower()
        
        # Replacing symbols
        for to_replace in ('\n', '&gt; ', '&gt', '/', '-'):
            text = text.replace(to_replace, ' ')
        text = re.sub('\s\s+', ' ', text)
        text = re.sub('[^a-z ]', '', text)
        
        text = f' {text} '
        
        for word_to_sub in words_to_sub.keys():
            text = text.replace(f' {word_to_sub} ', f' {words_to_sub[word_to_sub]} ')
        
        # lemmatize and normalize and remove stopwords
        words = [
            lemmatize(lemmatizer, word) 
            for word in text.strip().split(' ') 
            if word != '' and word not in stop_words]

        text = ' '.join(words).strip()

        return text
