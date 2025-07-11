import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import json
from datetime import datetime
from collections import Counter

# Load the fine-tuned FinBERT model and tokenizer
finetuned_model_path = "finbert-finetuned-latest"  # Ensure this folder exists in the working directory
finbert = BertForSequenceClassification.from_pretrained(finetuned_model_path)
tokenizer = BertTokenizer.from_pretrained(finetuned_model_path)
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# Function to get specific stock-related news from Google News
def getSpecificStockNews(ticker):
    headers = {
        "User-Agent": 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }
    
    url = f'https://www.google.com/search?q={ticker}+share+price&tbm=nws'
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    news_articles = []
    
    for el in soup.select("div.SoaBEf"):
        title = el.select_one("div.MBeuO").get_text()
        time = el.select_one(".LfVVr").get_text()
        news_articles.append((title, time))
           
        if len(news_articles) >= 5:
            break
    
    return news_articles

# Function to apply FinBERT sentiment analysis using the fine-tuned model
def get_finbert_sentiment(text):
    results = nlp(text)
    return results[0]  # returns dict: {'label': 'Positive'/'Neutral'/'Negative', 'score': float}

# Streamlit app
def main():
    st.title("📈 Stock News Sentiment Analysis using Fine-tuned FinBERT")

    # Load stock data (ensure 'nifty_stocks.xlsx' exists with 'Company Name' and 'Symbol' columns)
    stock_data = pd.read_excel('nifty_stocks.xlsx')
    company_to_symbol = stock_data.set_index('Company Name')['Symbol'].to_dict()

    selected_company = st.selectbox("Select a Company:", list(company_to_symbol.keys()))
    ticker_symbol = company_to_symbol[selected_company]

    st.write(f"Fetching news for **{selected_company}** (Ticker: {ticker_symbol})")
    
    news_articles = getSpecificStockNews(ticker_symbol)

    sentiment_labels = []
    news_sentiment = 0
    
    if news_articles:
        st.subheader(f"📰 Latest News for {selected_company} (Ticker: {ticker_symbol})")
        
        # Identify the most recent article (based on the time field)
        latest_article = None
        latest_time = None
        
        for title, time in news_articles:
            try:
                parsed_time = datetime.strptime(time, "%b %d, %Y") if ',' in time else datetime.strptime(time, "%b %d")
            except ValueError:
                continue
            
            if latest_time is None or parsed_time > latest_time:
                latest_time = parsed_time
                latest_article = (title, time)

        if latest_article:
            latest_title, latest_time_str = latest_article
            st.markdown(f"**🕒 Latest Article:** {latest_title} ({latest_time_str})")
        
        st.markdown("---")
        for title, time in news_articles:
            if latest_article and (title, time) == latest_article:
                continue
            
            finbert_result = get_finbert_sentiment(title)
            sentiment_labels.append(finbert_result['label'])

            st.write(f"**{title}** ({time})")
            st.write(f"→ Sentiment: `{finbert_result['label']}` with score `{finbert_result['score']:.2f}`")
            st.markdown("---")

        # Determine overall sentiment (excluding the latest article)
        if sentiment_labels:
            if all(sentiment == 'Neutral' for sentiment in sentiment_labels):
                most_common_sentiment = 'Negative'
            else:
                most_common_sentiment = max(set(sentiment_labels), key=sentiment_labels.count)

            if most_common_sentiment in ['Positive', 'Neutral']:
                news_sentiment = 1
            else:
                news_sentiment = 0

            st.success(f"🧠 Most Common Sentiment (Excluding Latest): **{most_common_sentiment}** → Final Sentiment Score: `{news_sentiment}`")
        else:
            st.warning(f"No valid sentiment found for news related to {selected_company}.")
    
    # Save the sentiment value to a JSON file
    with open('stock_news_out.json', 'w') as f:
        json.dump(news_sentiment, f)

if __name__ == "__main__":
    main()
