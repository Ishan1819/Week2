**NLTK-Powered Text Analytics Web App
Overview**
This project is a web-based natural language processing (NLP) tool built using Streamlit, NLTK, and pandas. It allows users to upload a .txt file and explore various NLP analyses like tokenization, POS tagging, sentiment scoring, and N-gram frequency distributions through an interactive UI.


**Features Implemented**
Text Preprocessing:
Cleaning, punctuation removal, lowercasing
Stopword removal and lemmatization

**NLP Techniques:**
Tokenization and Part-of-Speech (POS) tagging
Word frequency distributions
Bigram and Trigram collocations
Sentiment analysis using TextBlob

**Visualizations:**
Bar charts for word frequencies and N-grams (via Streamlit/matplotlib)

**Interactive Web Interface:**
Built using Streamlit

**Two pages:**
Data Explorer to view raw text, tokens, POS, and sentiment
Analysis Dashboard to visualize frequency and N-gram stats


**Requirements.txt**
Clone or download the repository
pip install streamlit nltk pandas matplotlib textblob
After installing, put below command in the terminal
streamlit run streamlit_app.py
Upload a .txt file from the sidebar to start exploring the text.


**Pages Overview**
Data Explorer
Shows original text
Displays first 50 cleaned tokens
POS-tagged word list
Sentiment polarity and subjectivity scores

**Analysis Dashboard**
Top 20 frequent words (table + bar chart)
Top Bigrams (table + bar chart)
Top Trigrams (table + bar chart)


**Key Learnings:**
Developed a complete NLP pipeline with basic and intermediate text analysis.
Learned to integrate NLTK, pandas, and Streamlit effectively.
Understood how to visualize textual patterns using N-grams and sentiment.
Gained experience in building modular NLP functions for real-time web applications.

