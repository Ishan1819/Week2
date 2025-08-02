# streamlit_app.py

import streamlit as st
from nlp_pipeline import clean_text, get_pos_tags, get_word_frequencies, get_ngrams, get_sentiment_scores

# Set Streamlit page config
st.set_page_config(page_title="Text Analytics Web App", layout="wide")

st.title("NLP-Powered Text Analytics")

# Sidebar navigation
page = st.sidebar.radio("Go to:", ["Data Explorer", "Analysis Dashboard"])

# Uploading text file
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

# If file is uploaded
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    tokens = clean_text(raw_text)
    pos_tags = get_pos_tags(tokens)
    sentiment = get_sentiment_scores(raw_text)
    word_freq_df = get_word_frequencies(tokens)
    bigram_df = get_ngrams(tokens, n=2)
    trigram_df = get_ngrams(tokens, n=3)

    # Data Explorer
    if page == "Data Explorer":
        st.subheader("Original Text")
        st.text_area("Raw Text", value=raw_text, height=250)

        st.subheader("Cleaned Tokens")
        st.write(tokens[:50])  # show first 50 tokens

        st.subheader("POS Tagging")
        st.dataframe(pos_tags, use_container_width=True)

        st.subheader("Sentiment Scores")
        st.json(sentiment)

    # Analysis Dashboard
    elif page == "Analysis Dashboard":
        st.subheader("Top 20 Most Frequent Words")
        st.dataframe(word_freq_df, use_container_width=True)
        st.bar_chart(word_freq_df.set_index("Word"))

        st.subheader("Top Bigrams")
        st.dataframe(bigram_df)
        st.bar_chart(bigram_df.set_index("N-gram"))

        st.subheader("Top Trigrams")
        st.dataframe(trigram_df)
        st.bar_chart(trigram_df.set_index("N-gram"))
else:
    st.warning("Please upload a .txt file to begin.")
