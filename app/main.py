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
