from textblob import TextBlob

def analyze_sentiment(text: str) -> float:
    """
    Analyzes the sentiment of the given text using TextBlob.
    Returns a polarity score between -1.0 (Negative) and 1.0 (Positive).
    """
    if not isinstance(text, str):
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity
