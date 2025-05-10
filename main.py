import praw
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



reddit = praw.Reddit("default")



posts = []
subreddit = reddit.subreddit("MachineLearning")
for post in subreddit.top(time_filter='week', limit=1000):
    posts.append({
        "title": post.title,
        "selftext": post.selftext,
        "subreddit": post.subreddit.display_name,
        "upvotes": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "id": post.id,
        "author": str(post.author),
        "url": post.url,
        "flair": str(post.link_flair_text)
    })



    df = pd.DataFrame(posts)
    df.to_csv("reddit_data.csv", index=False)




stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = (df['title'] + " " + df['selftext']).fillna('').apply(clean_text)



vectorizer = TfidfVectorizer(max_df=0.9, min_df=10, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])



kmeans = KMeans(n_clusters=10, random_state=0)
df['cluster'] = kmeans.fit_predict(X)


df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
df.set_index('datetime', inplace=True)
trends = df.groupby([pd.Grouper(freq='D'), 'cluster']).size().unstack().fillna(0)



analyzer = SentimentIntensityAnalyzer()

df['sentiment'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


st.title("Reddit Trend Catcher")

cluster = st.selectbox("Select Topic Cluster", sorted(df['cluster'].unique()))
filtered = df[df['cluster'] == cluster]

st.subheader("Top Posts")
st.write(filtered[['title', 'upvotes', 'sentiment']].sort_values(by='upvotes', ascending=False).head(10))

st.subheader("Word Cloud")
wc = WordCloud(width=800, height=400).generate(" ".join(filtered['cleaned_text']))
st.image(wc.to_array())

st.subheader("Trend Over Time")
st.line_chart(trends)

