import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text_without_sc = [t for t in text if t.isalnum()]
    text = text_without_sc[:]
    text_without_sc.clear()
    transformed_text = [
        i
        for i in text
        if i not in stopwords.words('english') and i not in punctuation
    ]

    text = transformed_text[:]
    transformed_text.clear()

    transformed_text.extend(ps.stem(i) for i in text)
    return " ".join(transformed_text)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email Spam Classifier")
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vectorized_sms = tfidf.transform([transform_sms])
    result = model.predict(vectorized_sms)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
