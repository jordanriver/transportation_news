import requests
from bs4 import BeautifulSoup
from langchain import LangChain, Pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
import sqlite3
import streamlit as st

# Function to scrape website
def scrape_website(url):
    response = requests.get(url)
    response.encoding = 'utf-8'  # Ensure the response is interpreted as UTF-8
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    return articles

# Initialize LangChain
lc = LangChain()
text_splitter = CharacterTextSplitter()
llm = OpenAI()

# Define pipelines for keyword search, summarization, and categorization
keyword_pipeline = Pipeline(
    components=[
        text_splitter,
        llm
    ]
)

summarization_pipeline = Pipeline(
    components=[
        text_splitter,
        llm
    ]
)

categorization_pipeline = Pipeline(
    components=[
        text_splitter,
        llm
    ]
)

# Function to search, summarize, and categorize articles
def search_articles(articles, keywords):
    results = []
    for article in articles:
        for keyword in keywords:
            if keyword.lower() in article.text.lower():
                summary = summarization_pipeline.run(article.text)
                category = categorization_pipeline.run(article.text)
                results.append({'article': article, 'summary': summary, 'category': category})
                break
    return results

# Function to store data in SQLite
def store_data(articles):
    conn = sqlite3.connect('news.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles (title TEXT, summary TEXT, category TEXT)''')
    for article in articles:
        title = article['article'].find('h2').text
        summary = article['summary']
        category = article['category']
        c.execute("INSERT INTO articles (title, summary, category) VALUES (?, ?, ?)", (title, summary, category))
    conn.commit()
    conn.close()

# Streamlit app
st.set_page_config(page_title="News Search", layout="wide", initial_sidebar_state="expanded")
st.title('חיפוש חדשות')

keywords = st.text_input('הכנס מילות מפתח (מופרדות בפסיק):')

if st.button('חפש'):
    articles = scrape_website('https://example.com/news')
    if keywords:
        keywords_list = [keyword.strip() for keyword in keywords.split(',')]
        results = search_articles(articles, keywords_list)
        store_data(results)
        if results:
            st.write(f'נמצאו {len(results)} מאמרים:')
            for result in results:
                st.subheader(result['article'].find('h2').text)
                st.write(f"תקציר: {result['summary']}")
                st.write(f"קטגוריה: {result['category']}")
        else:
            st.write('לא נמצאו מאמרים.')
