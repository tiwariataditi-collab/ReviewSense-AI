import pandas as pd
import os
import sys
from sentiment import analyze_sentiment

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE = os.path.join(BASE_DIR, "data", "Reviews.csv")

def evaluate_sentiment_accuracy():
    print(f"Loading dataset from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found.")
        return

    df = df.dropna(subset=['Text', 'Score'])
    
    # Sample 10,000 rows so it evaluates quickly
    df_sample = df.sample(n=10000, random_state=42)
    
    print("Mapping True Sentiments based on Star Ratings...")
    def get_true_sentiment(score):
        if score >= 4: return "Positive"
        elif score <= 2: return "Negative"
        else: return "Neutral"
    
    df_sample['True_Sentiment'] = df_sample['Score'].apply(get_true_sentiment)
    
    print("Analyzing text with TextBlob...")
    def get_pred_sentiment(text):
        polarity = analyze_sentiment(text)
        # Using a small threshold to define neutrality
        if polarity > 0.1: return "Positive"
        elif polarity < -0.1: return "Negative"
        else: return "Neutral"
        
    df_sample['Pred_Sentiment'] = df_sample['Text'].apply(get_pred_sentiment)
    
    # Calculate Accuracy
    accuracy = (df_sample['True_Sentiment'] == df_sample['Pred_Sentiment']).mean()
    print(f"\n📊 Sentiment Analysis (TextBlob) Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_sentiment_accuracy()