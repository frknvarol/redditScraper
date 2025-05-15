import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import streamlit as st
import nltk
from collections import Counter
from nltk.corpus import stopwords
import string
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

df = pd.read_csv("reddit_data.csv")

stop_words = set(stopwords.words('english')).union({
    'im', 'ive', 'dont', 'youre', 'the', 'i', 'am', 'like', 'would', 'could', 'also', 'one', 'new',
    'r', 'get', 'know', 'think', 'make', 'used', 'use', 'really', 'anyone', 'something', 'want',
    'many', 'well', 'much', 'even', 'need', 'still', 'might', 'good', 'lot', 'thing', 'way', 'see',
    'first', 'best', 'post', 'people', 'however', 'two', 'without', 'full', 'less', 'seems', 'help', 'etc',
    'what', 'whats', 'feel', 'show', 'context', 'step', 'size', 'eg', 'large', 'p', 'approach', 'system',
    'example', 'inference', 'find', 'better', 'working', 'different', 'year', 'attention', 'thought', 'idea',
    'idea', 'based', 'link', 'please', 'similar', 'other', 'others', 'series', 'number', 'understand',
    'discussion', 'let', 'current', 'review', 'learn', 'value', 'look', 'looking', 'share', 'everyone',
    'since', 'next', 'interested', 'found', 'state', 'read', 'single', 'instead', 'v', 'bit', 'hi', 'faster',
    'u', 'take', 'actually', 'specific', 'author', 'who', 'whom', 'made', 'available', 'experience', 'understanding',
    'may', 'simple', 'n', 'thanks', 'thank', 'video', 'part', 'source', 'around', 'challenge', 'scale', 'come',
    'recently', 'small', 'big', 'compare', 'compared', 'create', 'say', 'cost', 'recent', 'often',
    'quite', 'go', 'id', 'multiple', 'including', 'current', 'currently', 'got', 'x'
})
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = (df['title'] + " " + df['selftext']).fillna('').apply(clean_text)

vectorizer = TfidfVectorizer(max_df=0.9, min_df=10)
X = vectorizer.fit_transform(df['cleaned_text'])

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
df.set_index('datetime', inplace=True)
trends = df.groupby([pd.Grouper(freq='D'), 'cluster']).size().unstack().fillna(0)

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

st.title("Reddit Trend Catcher (From CSV)")

cluster = st.selectbox("Select Topic Cluster", sorted(df['cluster'].unique()))
filtered = df[df['cluster'] == cluster]

st.subheader("Top Posts")
st.write(filtered[['title', 'upvotes', 'sentiment']].sort_values(by='upvotes', ascending=False).head(10))

st.subheader("Word Cloud")
wc = WordCloud(width=800, height=400).generate(" ".join(filtered['cleaned_text']))
st.image(wc.to_array())

st.subheader("Trend Over Time")
st.line_chart(trends)


pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())

df['pca_x'] = reduced[:, 0]
df['pca_y'] = reduced[:, 1]

st.subheader("PCA Cluster Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='tab10', legend='full', s=50)
st.pyplot(fig)



st.subheader("Top Words Per Cluster")

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(4):
    top_words = [terms[ind] for ind in order_centroids[i, :4]]
    st.markdown(f"**Cluster {i}:** " + ", ".join(top_words))



st.subheader("Sentiment Distribution")

fig2, ax2 = plt.subplots()
sns.histplot(df['sentiment'], kde=True, bins=30)
st.pyplot(fig2)



all_text = ' '.join(df['cleaned_text'])

tokens = all_text.lower().translate(str.maketrans('', '', string.punctuation)).split()

filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

word_freq = Counter(filtered_tokens)

top_100 = word_freq.most_common(100)


with open("top_words.txt", "w") as f:
    for word, freq in top_100:
        f.write(f"{word}: {freq}\n")

# streamlit run main.py
