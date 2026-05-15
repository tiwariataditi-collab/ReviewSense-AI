import plotly.express as px
import pandas as pd

def plot_top_products(df: pd.DataFrame, top_n: int = 10):
    """Plots the highest-rated products based on average score."""
    top_products = df.groupby('ProductId')['Score'].mean().sort_values(ascending=False).head(top_n).reset_index()
    
    fig = px.bar(
        top_products, 
        x='ProductId', 
        y='Score', 
        title="Top Rated Products", 
        color='Score', 
        color_continuous_scale='Viridis', 
        text_auto='.2f'
    )
    fig.update_layout(xaxis_title="Product ID", yaxis_title="Average Score")
    return fig

def plot_sentiment_distribution(df: pd.DataFrame):
    """Plots the distribution of sentiment scores."""
    fig = px.histogram(
        df, 
        x='Sentiment', 
        nbins=30, 
        title="Sentiment Score Distribution", 
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(xaxis_title="Sentiment Polarity (-1.0 to 1.0)", yaxis_title="Number of Reviews")
    return fig

def plot_trending_products(df: pd.DataFrame, top_n: int = 10):
    """Plots trending products based on total review counts."""
    trending = df['ProductId'].value_counts().head(top_n).reset_index()
    trending.columns = ['ProductId', 'ReviewCount']
    
    fig = px.bar(
        trending, 
        x='ProductId', 
        y='ReviewCount', 
        title="Trending Products (Most Reviews)", 
        color='ReviewCount', 
        color_continuous_scale='Magma', 
        text_auto=True
    )
    fig.update_layout(xaxis_title="Product ID", yaxis_title="Total Reviews")
    return fig
