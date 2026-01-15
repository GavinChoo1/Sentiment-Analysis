import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Get correct label mapping from model config (safe from label reordering)
id2label = finbert_model.config.id2label
labels = [id2label[i] for i in range(len(id2label))]

vader_analyzer = SentimentIntensityAnalyzer()


# ============================================================================
# NEWS FETCHING
# ============================================================================

def fetch_news(query, num_articles=10, region="SG", language="en"):
    """
    Fetch news articles from Google News RSS with locale parameters.
    
    Args:
        query: Search query string
        num_articles: Max articles to fetch per query
        region: Region code (SG for Singapore, US for USA, etc.)
        language: Language code (en for English)
    
    Returns:
        List of article dictionaries with title, link, published, content
    """
    # Add locale parameters for consistency
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl={language}&gl={region}&ceid={region}:{language}"
    
    try:
        feed = feedparser.parse(rss_url)
        news_items = feed.entries[:num_articles]
    except Exception as e:
        print(f"Error fetching RSS feed for '{query}': {e}")
        return []

    articles = []
    for item in news_items:
        title = item.get("title", "N/A")
        link = item.get("link", "N/A")
        published = item.get("published", "N/A")
        
        # Try to fetch content, but don't block if it fails
        content = fetch_article_content(link)
        
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles


def fetch_article_content(url, timeout=10):
    """
    Attempt to fetch article content from URL.
    Google News links may not resolve to full article content.
    
    Args:
        url: Article URL
        timeout: Request timeout in seconds
    
    Returns:
        Article content as string, or empty string if unavailable
    """
    if not url or url == "N/A":
        return ""
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        return content.strip()
    except (requests.RequestException, Exception) as e:
        # Return empty string instead of error message for cleaner sentiment analysis
        return ""


# ============================================================================
# SENTIMENT ANALYSIS - VADER
# ============================================================================

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Best for social media and finance headlines.
    
    Args:
        text: Text to analyze
    
    Returns:
        Tuple of (polarity_score, sentiment_label)
    """
    if not text or not text.strip():
        return 0.0, "Neutral"
    
    scores = vader_analyzer.polarity_scores(text)
    polarity = scores['compound']

    # Standard VADER thresholds
    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment


# ============================================================================
# SENTIMENT ANALYSIS - FINBERT
# ============================================================================

def analyze_sentiment_finbert(text):
    """
    Analyze sentiment using FinBERT (BERT fine-tuned for financial tone).
    More accurate for financial news but slower and GPU-intensive.
    
    Args:
        text: Text to analyze (will be truncated to 512 tokens)
    
    Returns:
        Tuple of (confidence_score, sentiment_label)
    """
    if not text or not text.strip():
        return 0.0, "Neutral"

    try:
        inputs = finbert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = finbert_model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        max_index = np.argmax(probabilities)
        
        # Use id2label for safe label mapping
        sentiment = id2label[max_index]
        confidence = probabilities[max_index]

        return confidence, sentiment
    except Exception as e:
        print(f"Error in FinBERT analysis: {e}")
        return 0.0, "Neutral"


# ============================================================================
# DEDUPLICATION & ANALYSIS
# ============================================================================

def deduplicate_articles(articles):
    """
    Remove duplicate articles by link.
    
    Args:
        articles: List of article dictionaries
    
    Returns:
        List of unique articles (preserving order, keeping first occurrence)
    """
    seen_links = set()
    unique_articles = []
    
    for article in articles:
        link = article.get("link", "")
        if link not in seen_links:
            seen_links.add(link)
            unique_articles.append(article)
    
    return unique_articles


def analyze_article_sentiment(article, use_finbert=False, include_content=True):
    """
    Analyze sentiment of an article using title and optionally content.
    
    Args:
        article: Article dictionary
        use_finbert: If True, use FinBERT; else use VADER
        include_content: If True, combine title + content for analysis
    
    Returns:
        Tuple of (score, sentiment_label)
    """
    title = article.get("title", "")
    content = article.get("content", "")
    
    # Combine title and content for better signal (only if content is available)
    if include_content and content:
        text = f"{title}. {content}"
    else:
        text = title
    
    if use_finbert:
        return analyze_sentiment_finbert(text)
    else:
        return analyze_sentiment_vader(text)


# ============================================================================
# REPORTING
# ============================================================================

def print_article_details(idx, article, score, sentiment):
    """Pretty print article details and sentiment."""
    print(f"\n{'='*80}")
    print(f"Article {idx}: {article['title']}")
    print(f"{'='*80}")
    print(f"Link: {article['link']}")
    print(f"Published: {article['published']}")
    print(f"Sentiment: {sentiment} (Score: {score:.4f})")
    if article['content']:
        print(f"Content Preview: {article['content'][:200]}...")
    else:
        print("Content: [Not available]")


def summarize_sentiments(articles, sentiments):
    """
    Print market sentiment summary.
    
    Args:
        articles: List of articles analyzed
        sentiments: List of (score, label) tuples in same order
    """
    summary = {label: 0 for label in labels}
    scores = {label: [] for label in labels}

    for _, sentiment in sentiments:
        summary[sentiment] += 1

    total = len(articles)
    
    print(f"\n{'='*80}")
    print("MARKET SENTIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total articles analyzed: {total}")
    print(f"Unique articles (after deduplication): {total}")
    
    for sentiment in labels:
        count = summary[sentiment]
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{sentiment:10s}: {count:3d} ({percent:5.2f}%)")
    
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline: fetch news, deduplicate, analyze, report."""
    
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]
    
    num_articles_per_query = 10
    all_articles = []

    # Fetch news for all queries
    print("\n" + "="*80)
    print("FETCHING NEWS ARTICLES")
    print("="*80)
    for query in queries:
        print(f"\nFetching articles for '{query}'...")
        articles = fetch_news(query, num_articles_per_query, region="SG", language="en")
        print(f"  → Retrieved {len(articles)} articles")
        all_articles.extend(articles)

    # Deduplicate
    print(f"\nTotal articles before deduplication: {len(all_articles)}")
    all_articles = deduplicate_articles(all_articles)
    print(f"Total articles after deduplication: {len(all_articles)}")

    # Analyze sentiment
    print("\n" + "="*80)
    print("ANALYZING SENTIMENT")
    print("="*80)
    
    sentiments = []
    use_finbert = False  # Set to True to use FinBERT instead of VADER
    
    for idx, article in enumerate(all_articles, 1):
        score, sentiment = analyze_article_sentiment(
            article, 
            use_finbert=use_finbert,
            include_content=True  # Include article content if available
        )
        sentiments.append((score, sentiment))
        print_article_details(idx, article, score, sentiment)

    # Print summary
    summarize_sentiments(all_articles, sentiments)


if __name__ == "__main__":
    main()