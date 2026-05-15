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

# ==========================================
# 3. MAIN DASHBOARD UI
# ==========================================
def main():
    # Setup session state for login
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        auth()
        st.stop()

    # Sidebar Profile
    st.sidebar.title("Profile")
    st.sidebar.write(f"Welcome, **{st.session_state['user']}** 👋")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

    st.title("🛍️ NLP-Powered Product Recommendation Platform")
    st.markdown("Discover the best products driven by **TF-IDF Cosine Similarity** and **Sentiment Analysis**.")
    
    # Load all models and data
    df = load_and_prep_data()
    nlp_recommender = load_nlp_model(df)
    svd_model, user_map, all_products = load_svd_model()
    
    tabs = st.tabs(["🤖 For You", "🔍 Search Similar Products", "📊 Analytics Dashboard", "📈 Top & Trending"])
    
    # Tab 1: Personalized SVD Recommendations
    with tabs[0]:
        st.subheader("Personalized Picks for You")
        if svd_model:
            num_recs = st.slider("Number of recommendations", 1, 10, 5, key='svd_slider')
            if st.button("🎯 Get Recommendations", use_container_width=True):
                with st.spinner("Analyzing your profile..."):
                    recs = get_svd_recommendations(st.session_state['user'], svd_model, user_map, all_products, num_recs)
                    if recs:
                        st.success("✅ Top Personalized Recommendations based on Collaborative Filtering:")
                        for i, product_id in enumerate(recs, 1):
                            st.info(f"{i}. **{product_id}**")
                    else:
                        st.warning("Not enough user history to provide SVD recommendations.")
        else:
            st.warning("Collaborative filtering model files are missing from the 'models/' folder.")

    # Tab 2: NLP Similar Products
    with tabs[1]:
        st.subheader("Search Content-Based Recommendations")
        product_list = df['ProductId'].unique()
        selected_product = st.selectbox("Select a Product ID to get recommendations:", product_list)
        
        if selected_product:
            st.write(f"### Similar to {selected_product}")
            recs = nlp_recommender.recommend(selected_product, top_n=3)
            
            if not recs.empty:
                for idx, row in recs.iterrows():
                    with st.expander(f"Product: {row['ProductId']} | Match: {row['SimilarityScore']:.2%} | ⭐ {row['Score']}/5"):
                        st.markdown(f"**Summary:** {row['Summary']}")
                        st.markdown(f"**Review Preview:** {row['Text']}")
                        st.markdown(f"**Helpfulness:** {row['HelpfulnessNumerator']} / {row['HelpfulnessDenominator']}")
                        
                        sentiment_color = "green" if row['Sentiment'] > 0 else "red" if row['Sentiment'] < 0 else "gray"
                        st.markdown(f"**Sentiment Score:** :{sentiment_color}[{row['Sentiment']:.2f}]")
                        st.progress(float(row['SimilarityScore']))
                        
                        st.info(f"**Why recommended?** This product shares similar textual review content based on NLP TF-IDF features with a {row['SimilarityScore']:.2%} similarity match.")
            else:
                st.warning("Not enough data to recommend similar products.")
                
    # Tab 3: Analytics
    with tabs[2]:
        st.subheader("Sentiment Insights & Analytics")
        st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)
        st.info("The **Sentiment Polarity** score ranges from **-1.0** (Negative) to **+1.0** (Positive). By analyzing the text of reviews, we can understand the overall customer satisfaction beyond just the star rating.")
            
    # Tab 4: Trending
    with tabs[3]:
        st.subheader("Product Performance Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_top_products(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_trending_products(df), use_container_width=True)

if __name__ == "__main__":
    main()

