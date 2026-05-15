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

# ==========================================
# 1. AUTHENTICATION LOGIC
# ==========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def auth():
    st.title("🔐 Login / Signup to AI Recommender")
    choice = st.radio("Choose Option", ["Login", "Signup"])
    users = load_users()

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("👤 Username", key="username")
    with col2:
        password = st.text_input("🔑 Password", type="password", key="password")

    if choice == "Signup":
        if st.button("Create Account"):
            if not username or not password:
                st.error("Username and password cannot be empty")
            elif username in users:
                st.warning("User already exists ❗")
            else:
                users[username] = hash_password(password)
                save_users(users)
                st.success("Account created ✅ Please switch to Login.")

    if choice == "Login":
        if st.button("Login"):
            if not username or not password:
                st.error("Username and password cannot be empty")
            elif username in users and users[username] == hash_password(password):
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")
                
# ==========================================
# 2. DATA & MODEL LOADING LOGIC
# ==========================================
@st.cache_data
def load_and_prep_data():
    """
    Loads dataset from a CSV file and computes sentiments.
    """
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {DATA_FILE}. Please check your 'data/' folder.")
        st.stop()
        
    # IMPORTANT: Drop rows with missing text first
    df = df.dropna(subset=['Text', 'Summary'])
    
    # Sample data to make it load instantly (fixes the black screen issue)
    # Adjust 'n' if you want more or fewer rows to process
    df = df.sample(n=1000, random_state=42)
    
    # Calculate Sentiment mapping applying our sentiment model
    df['Sentiment'] = df['Text'].apply(analyze_sentiment)
    return df

@st.cache_resource
def load_nlp_model(df):
    """Trains and returns the NLP Recommender."""
    recommender = NLPRecommender()
    recommender.fit(df)
    return recommender

@st.cache_resource
def load_svd_model():
    """Loads the pre-trained Collaborative Filtering SVD models."""
    try:
        with open(os.path.join(MODELS_DIR, "svd_recommender_model.pkl"), "rb") as f:
            svd_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "user_rated_products_map.pkl"), "rb") as f:
            user_map = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "all_products.pkl"), "rb") as f:
            all_products = pickle.load(f)
        return svd_model, user_map, all_products
    except Exception as e:
        return None, None, None

def get_svd_recommendations(user_id, model, user_map, all_products, n=5):
    if model is None:
        return []
    rated = user_map.get(user_id, [])
    preds = []
    for product in all_products:
        if product not in rated:
            pred = model.predict(user_id, product)
            preds.append((product, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:n]]
