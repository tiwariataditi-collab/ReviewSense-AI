import streamlit as st
import pandas as pd
import sys
import os
import json
import hashlib
import pickle

# Add parent directory to path to allow absolute imports from root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.recommender import NLPRecommender
from src.sentiment import analyze_sentiment
from visuals.charts import plot_top_products, plot_sentiment_distribution, plot_trending_products

# -- Configuration --
st.set_page_config(page_title="AI Product Recommender", layout="wide", page_icon="🛍️")

USERS_FILE = os.path.join(BASE_DIR, "users.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "data", "Reviews.csv")