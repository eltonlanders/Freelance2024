# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:25:41 2024

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





# NER

import spacy

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm



# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply the function to extract entities
data['entities'] = data['content'].apply(extract_entities)



# Filter entities to include only relevant labels
def filter_relevant_entities(entities):
    relevant_labels = ['FOOD', 'GPE', 'ORG', 'PRODUCT']
    return [ent for ent in entities if ent[1] in relevant_labels]

data['filtered_entities'] = data['entities'].apply(filter_relevant_entities)

# Display results
print(data[['content', 'filtered_entities']].head())





# Define keyword lists
cuisine_keywords = ["pizza", "vegan", "desserts", "pasta", "coffee", "biryani"]
theme_keywords = ["cozy", "dim lighting", "friendly staff", "romantic", "noisy"]

# Function to match keywords
def match_keywords(text, keywords):
    tokens = text.split()
    return [token for token in tokens if token in keywords]

# Match cuisines and themes
data['cuisine_mentions'] = data['content_lemmatized'].apply(lambda tokens: match_keywords(' '.join(tokens), cuisine_keywords))
data['theme_mentions'] = data['content_lemmatized'].apply(lambda tokens: match_keywords(' '.join(tokens), theme_keywords))

# Display results
print(data[['content', 'cuisine_mentions', 'theme_mentions']].head())


"""

from transformers import pipeline 
import torch

# Load a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")

# Function to analyze sentiment for aspects
def analyze_aspect_sentiment(aspects, text):
    results = {}
    for aspect in aspects:
        if aspect in text:
            sentiment = sentiment_analyzer(aspect)
            results[aspect] = sentiment[0]['label']  # 'LABEL' is either POSITIVE, NEGATIVE, or NEUTRAL
    return results

# Apply aspect sentiment analysis
data['aspect_sentiments'] = data.apply(lambda row: analyze_aspect_sentiment(row['cuisine_mentions'] + row['theme_mentions'], row['content']), axis=1)

# Display results
print(data[['content', 'aspect_sentiments']].head())



import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True if GPU support is available 






from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Convert lemmatized content to text format
data['cleaned_text'] = data['content_lemmatized'].apply(' '.join)

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data['cleaned_text'])

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display topics
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}: {[feature_names[i] for i in topic.argsort()[-10:]]}")








from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate a word cloud for cuisines
cuisine_text = ' '.join(data['cuisine_mentions'].sum())
wordcloud = WordCloud(background_color="white").generate(cuisine_text)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Popular Cuisines")
plt.show()


import pandas as pd

# Prepare data for sentiment analysis
aspect_sentiment_counts = pd.DataFrame(
    data['aspect_sentiments'].explode().value_counts().reset_index(),
    columns=['Aspect', 'Sentiment', 'Count']
)

# Plot sentiment distribution
import seaborn as sns
sns.barplot(data=aspect_sentiment_counts, x='Aspect', y='Count', hue='Sentiment')
plt.title("Aspect Sentiment Distribution")
plt.xticks(rotation=45)
plt.show()



"""
data.to_csv("processed.csv", index=False)



from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Tokenize data
train_encodings = tokenize_function(data['content'].tolist())
train_labels = data['sentiment'].tolist()  # Assuming sentiment column exists

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0, 'mixed': 2})




from sklearn.model_selection import train_test_split

# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['content'],  # Text data
    data['sentiment'],  # Sentiment labels
    test_size=0.2, 
    random_state=42
)


from transformers import BertTokenizer

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize train and validation data
train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="tf"
)

val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="tf"
)


import tensorflow as tf

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),  # Input IDs, attention masks
    train_labels.values     # Labels
)) 

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels.values
))

# Batch and shuffle the datasets
train_dataset = train_dataset.shuffle(1000).batch(16)
val_dataset = val_dataset.batch(16)



from transformers import TFBertForSequenceClassification

# Load pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)


# Evaluate the model
results = model.evaluate(val_dataset)
print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")


# Save the model
model.save_pretrained('./bert_sentiment_model_tf')
tokenizer.save_pretrained('./bert_sentiment_model_tf')



# Function to predict sentiment
def predict_sentiment(texts):
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
    # Get predictions
    outputs = model(encodings)
    predictions = tf.argmax(outputs.logits, axis=1).numpy()
    return predictions

# Predict sentiments for new examples
new_texts = ["The ambiance was fantastic!", "I hated the long wait."]
print(predict_sentiment(new_texts))














