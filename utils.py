import sys

# if env gives any issues
sys.path.extend(['D:\\Documents\\GitHub\\pq-analyzer\\env', 'D:\\Documents\\GitHub\\pq-analyzer\\env\\lib\\site-packages'])

from pdb import set_trace as st
import pickle
import re
import string

from bertopic import BERTopic
import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')

def _get_stopwords():
    print('getting stopwords')
    general_stopwords = ['about', 'actually', 'ago', 'agree', 'already', 'also', 'always',
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

    number_stopwords = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                        'tenth', 'last', 'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'one', 'two', 'three',
                        'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                        'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 
                        'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 
                        'million', 'billion', 'trillion']

    hansard_stopwords = ['speaker', 'leader', 'chairman', 'chairwoman', 'chair', 'deputy', 'beg', 'permission',
                         'move', 'withdraw', 'allocate', 'head', 'cut', 'thank', 'member', 'honorable', 
                         'question', 'clarification', 'grateful', 'year', 'month', 'day', 'reply', 'many', 
                         'ministry', 'minister', 'consider', 'take', 'ask', 'sir', 'maam', 'since', 'especially', 
                         'reason', 'see', 'non', 'due', 'mention', 'respectively', 'respective', 'must', 'expect',
                         'mean', 'exist', 'cause', 'hon']

    adverb_stopwords = ['annually', 'currently', 'early', 'finally', 'firstly', 'fully', 'increasingly', 'likely',
                        'monthly', 'particularly', 'previously', 'secondly', 'specifically']

    return general_stopwords + honorific_stopwords + hansard_stopwords + number_stopwords + adverb_stopwords
    
stop_words.extend(_get_stopwords())

# Saving stopwords and phrases into files
with open('stopwords.pkl', 'wb') as f:
    pickle.dump(stop_words, f)
    
def _lemmatize(lemmatizer, word):
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
            _lemmatize(lemmatizer, word) 
            for word in text.strip().split(' ') 
            if word != '' and word not in stop_words]

        text = ' '.join(words).strip()

        return text

def _get_topics_from_model(topic_model, data):
    topics, probs = topic_model.fit_transform(data)
    frequency = topic_model.get_topic_freq()
    print(f'topics: {len(frequency["Topic"])}')
    
    # can't use "-1 in frequency['Topic']". idk why but it just doesn't work.
    if len(frequency[frequency['Topic'] == -1]) > 0: 
        num_rows_with_no_topic = frequency['Count'][frequency['Topic'] == -1].values[0]
        pct_rows_with_no_topic = round(num_rows_with_no_topic/len(data)*100,1)
        print('Number of question rows with no topic: ', num_rows_with_no_topic,
              ' (', pct_rows_with_no_topic, '%)',
              sep = '')
    else:
        print('all questions were successfully assigned to a topic')

    return topic_model, frequency, topics, probs

def get_bert_topics(data, seed, anchors=None):
    #sentence_model = SentenceTransformer("paraphrase-mpnet-base-v2") 
    umap_model = UMAP(random_state=seed)

    if anchors:
        topic_model = BERTopic(
        # embedding_model = sentence_model,
            min_topic_size = 10,
            seed_topic_list = anchors,
            umap_model = umap_model,
            vectorizer_model = CountVectorizer(max_df = 0.5,
                                               min_df = 0.001,
                                               ngram_range = (1,4))
        )
    else:
        topic_model = BERTopic(
        # embedding_model = sentence_model,
            min_topic_size = 10,
            umap_model = umap_model,
            vectorizer_model = CountVectorizer(max_df = 0.5,
                                               min_df = 0.001,
                                               ngram_range = (1,4))
        )

    return _get_topics_from_model(topic_model, data)
