import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from langchain import LangChain, Pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
import sqlite3
import streamlit as st

# Function to check robots.txt
def can_scrape(url):
    parsed_url = requests.compat.urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp.can_fetch("*", url)

# Function to scrape the main page for article links
def scrape_main_page(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'html.parser')
    article_links = [a['href'] for a in soup.find_all('a', href=True) if 'article' in a['href']]
    return article_links

# Function to scrape content from an article link
def scrape_article(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1').text if soup.find('h1') else 'No title'
    content = soup.find('div', {'class': 'content'}).text if soup.find('div', {'class': 'content'}) else 'No content'
    return {'title': title, 'content': content}

# Function to scrape the website and collect articles
def scrape_website(url):
    if not can_scrape(url):
        return []

    article_links = scrape_main_page(url)
    articles = []
    for link in article_links:
        full_url = requests.compat.urljoin(url, link)
        article = scrape_article(full_url)
        articles.append(article)
    return articles

# Initialize LangChain
lc = LangChain()
text_splitter = CharacterTextSplitter()
llm = OpenAI()

# Define pipelines for summarization and categorization
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
def process_articles(articles, keywords):
    results = []
    for article in articles:
        for keyword in keywords:
            if keyword.lower() in article['content'].lower():
                summary = summarization_pipeline.run(article['content'])
                category = categorization_pipeline.run(article['content'])
                results.append({'title': article['title'], 'summary': summary, 'category': category})
                break
    return results

# Function to store data in SQLite
def store_data(articles):
    conn = sqlite3.connect('news.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles (title TEXT, summary TEXT, category TEXT)''')
    for article in articles:
        c.execute("INSERT INTO articles (title, summary, category) VALUES (?, ?, ?)", 
                (article['title'], article['summary'], article['category']))
    conn.commit()
    conn.close()

# Streamlit app
st.set_page_config(page_title="News Search", layout="wide", initial_sidebar_state="expanded")
st.title('חיפוש חדשות')

url = st.text_input('הכנס כתובת אתר חדשות:')
keywords = st.text_input('הכנס מילות מפתח (מופרדות בפסיק):')

if st.button('חפש'):
    if url:
        articles = scrape_website(url)
        if keywords:
            keywords_list = [keyword.strip() for keyword in keywords.split(',')]
            results = process_articles(articles, keywords_list)
            store_data(results)
            if results:
                st.write(f'נמצאו {len(results)} מאמרים:')
                for result in results:
                    st.subheader(result['title'])
                    st.write(f"תקציר: {result['summary']}")
                    st.write(f"קטגוריה: {result['category']}")
            else:
                st.write('לא נמצאו מאמרים.')
    else:
        st.write('אנא הכנס כתובת אתר.')
        