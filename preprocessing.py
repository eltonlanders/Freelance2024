# -*- coding: utf-8 -*-
"""
Created on Wed May 8 21:00:56 2024

@author: Elton Landers
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import spacy


data = pd.read_csv('data.csv')


missing_count = data['rating'].isnull().sum()
missing_percentage = (missing_count / len(data)) * 100
print(f"Missing values in 'rating': {missing_count} ({missing_percentage:.2f}%)")



rating_median = data['rating'].median()
data['rating'] = data['rating'].fillna(rating_median)


qualitative_data = data[(data['sentiment'].notnull()) & (data['content'].notnull())]

missing_after = data['rating'].isnull().sum()


data['content'] = qualitative_data['content'].str.lower() 

data['content'] = data['content'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

data['content'] = data['content'].apply(lambda x: re.sub(r'[^\w\s,]', '', x))

# Remove HTML tags
data['content'] = data['content'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Remove URLs
data['content'] = data['content'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))

# Remove extra whitespace
data['content'] = data['content'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['content'] = data['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


nltk.download('punkt')

data['content_tokens'] = data['content'].apply(lambda x: word_tokenize(x)) 



nltk.download('punkt', download_dir='C:/nltk_data')



sample_text = "This is a test sentence. Tokenization should work now!"
tokens = word_tokenize(sample_text)
print(tokens)


data['content'] = data['content'].fillna('')

data['content_tokens'] = data['content'].apply(lambda x: word_tokenize(x))


import shutil
nltk_data_path = 'C:/Users/Elton Landers/AppData/Roaming/nltk_data/tokenizers/punkt'
shutil.rmtree(nltk_data_path)  # Remove existing data
nltk.download('punkt')         # Re-download


nlp = spacy.load('en_core_web_sm')





import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

data['content'] = data['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))



from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
data['content_tokens'] = data['content'].apply(lambda x: word_tokenize(x))


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
data['content_stemmed'] = data['content_tokens'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
data['content_lemmatized'] = data['content_tokens'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])







