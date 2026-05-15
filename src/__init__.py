"""
Visualizations package for the Product Recommendation System.
Contains Plotly interactive charting functions for the Streamlit dashboard.
"""

from .charts import plot_top_products, plot_sentiment_distribution, plot_trending_products

__all__ = ["plot_top_products", "plot_sentiment_distribution", "plot_trending_products"]
