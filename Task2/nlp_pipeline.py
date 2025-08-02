# nlp_pipeline.py

import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, FreqDist, bigrams, trigrams
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# 1. Clean and preprocess text
def clean_text(text):
    text = text.lower()                          # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)                 # Tokenize
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
    return lemmas


# 2. POS tagging
def get_pos_tags(tokens):
    return pos_tag(tokens)


# 3. Frequency Distribution
def get_word_frequencies(tokens, top_n=20):
    freq_dist = FreqDist(tokens)
    return pd.DataFrame(freq_dist.most_common(top_n), columns=['Word', 'Frequency'])


# 4. N-grams
def get_ngrams(tokens, n=2, top_n=20):
    if n == 2:
        n_grams = bigrams(tokens)
    elif n == 3:
        n_grams = trigrams(tokens)
    else:
        return pd.DataFrame([], columns=["N-gram", "Frequency"])
    
    freq_dist = FreqDist(n_grams)
    return pd.DataFrame([(" ".join(gram), freq) for gram, freq in freq_dist.most_common(top_n)],
                        columns=["N-gram", "Frequency"])


# 5. Sentiment Analysis (using TextBlob)
def get_sentiment_scores(text):
    blob = TextBlob(text)
    return {
        "Polarity (-1 to 1)": round(blob.sentiment.polarity, 3),
        "Subjectivity (0 to 1)": round(blob.sentiment.subjectivity, 3)
    }
