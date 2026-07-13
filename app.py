import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def train_model():
    real = pd.read_csv(r'C:/Users/yeses/Downloads/True.csv', on_bad_lines='skip')
    fake = pd.read_csv(r'C:/Users/yeses/Downloads/Fake.csv', on_bad_lines='skip')
    real['label'] = 1
    fake['label'] = 0
    df = pd.concat([real, fake], ignore_index=True)
    df['content'] = df['title'] + ' ' + df['text']
    df['content'] = df['content'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    X = tfidf.fit_transform(df['content'])
    y = df['label']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, tfidf

st.title("AI-Powered Fake News Detector")
st.write("Paste any news article below and I will tell you if it is fake or real!")

with st.spinner("Loading model... please wait"):
    model, tfidf = train_model()

article = st.text_area("Paste your news article here:", height=200)

if st.button("Analyze Article"):
    if article.strip() == "":
        st.warning("Please paste an article first!")
    else:
        cleaned = clean_text(article)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        if prediction == 1:
            st.success(f"REAL NEWS ✅ (Confidence: {round(probability[1]*100, 1)}%)")
        else:
            st.error(f"FAKE NEWS ⚠️ (Confidence: {round(probability[0]*100, 1)}%)")