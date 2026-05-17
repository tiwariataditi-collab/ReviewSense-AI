import streamlit as st
import pandas as pd
import sys
import os
import hashlib
import pickle
import sqlite3
import subprocess

# Add parent directory to path to allow absolute imports from root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.recommender import NLPRecommender
from src.sentiment import analyze_sentiment
from visuals.charts import plot_top_products, plot_sentiment_distribution, plot_trending_products

# -- Configuration --
st.set_page_config(page_title="AI Product Recommender", layout="wide", page_icon="🛍️")

def set_custom_style():
    """Injects custom CSS to make the app look premium and professional."""
    st.markdown("""
    <style>
    /* Hide default Streamlit elements for a clean UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid #ff4b4b;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 4px 12px rgba(255, 75, 75, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "data", "Reviews.csv")
DB_FILE = os.path.join(BASE_DIR, "data", "users.db")

# ==========================================
# 1. AUTHENTICATION LOGIC
# ==========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    """Initializes the SQLite Database for user management."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def auth():
    st.title(" Login / Signup to AI Recommender")
    st.markdown("---")
    
    # Ensure DB exists
    init_db()
    
    choice = st.radio("Choose Option", ["Login", "Signup"])

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input(" Username", key="username")
    with col2:
        password = st.text_input(" Password", type="password", key="password")

    if choice == "Signup":
        if st.button("Create Account"):
            if not username or not password:
                st.error("Username and password cannot be empty")
            else:
                try:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                              (username, hash_password(password)))
                    conn.commit()
                    conn.close()
                    st.success("Account created. Please switch to Login.")
                except sqlite3.IntegrityError:
                    st.warning("User already exists !")

    if choice == "Login":
        if st.button("Login"):
            if not username or not password:
                st.error("Username and password cannot be empty")
            else:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                conn.close()
                
                if result and result[0] == hash_password(password):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials !")

# ==========================================
# 2. DATA & MODEL LOADING LOGIC
# ==========================================
@st.cache_data
def load_and_prep_data():
    """
    Loads dataset from a CSV file and computes sentiments.
    """
    try:
        # Use nrows to prevent Out-Of-Memory crashes on the 1GB Cloud Server
        df = pd.read_csv(DATA_FILE, nrows=20000)
    except Exception as e:
        st.error(f"Failed to load dataset at: {DATA_FILE}. Error details: {str(e)}")
        st.stop()
        
    # IMPORTANT: Drop rows with missing text first
    df = df.dropna(subset=['Text', 'Summary'])
    
    # Sample data to ensure lightning-fast UI performance
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
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
    svd_model_path = os.path.join(MODELS_DIR, "svd_recommender_model.pkl")
    user_map_path = os.path.join(MODELS_DIR, "user_rated_products_map.pkl")
    all_prod_path = os.path.join(MODELS_DIR, "all_products.pkl")
    
    missing = []
    if not os.path.exists(svd_model_path): missing.append("svd_recommender_model.pkl")
    if not os.path.exists(user_map_path): missing.append("user_rated_products_map.pkl")
    if not os.path.exists(all_prod_path): missing.append("all_products.pkl")
    
    if missing:
        return None, None, None, f"Missing files: {', '.join(missing)}"
        
    try:
        with open(svd_model_path, "rb") as f:
            svd_model = pickle.load(f)
        with open(user_map_path, "rb") as f:
            user_map = pickle.load(f)
        with open(all_prod_path, "rb") as f:
            all_products = pickle.load(f)
        return svd_model, user_map, all_products, "OK"
    except Exception as e:
        return None, None, None, f"Error loading models: {str(e)}"

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
    set_custom_style()
    
    # Setup session state for login
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        auth()
        st.stop()

    # Sidebar Profile
    st.sidebar.title("Profile")
    st.sidebar.write(f"Welcome, **{st.session_state['user']}** ")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

    st.title(" Product Recommendation Platform")
    st.markdown("Discover the best products driven by **TF-IDF Cosine Similarity** and **Sentiment Analysis**.")
    
    # Load all models and data
    df = load_and_prep_data()
    nlp_recommender = load_nlp_model(df)
    svd_model, user_map, all_products, svd_status = load_svd_model()
    
    tabs = st.tabs(["🤖 For You", "🔍 Search Similar Products", "📊 Analytics Dashboard", "📈 Top & Trending"])
    
    # Tab 1: Personalized SVD Recommendations
    with tabs[0]:
        st.subheader("Personalized Picks for You")
        if svd_model:
            num_recs = st.slider("Number of recommendations", 1, 10, 5, key='svd_slider')
            if st.button(" Get Recommendations", use_container_width=True):
                with st.spinner("Analyzing your profile..."):
                    recs = get_svd_recommendations(st.session_state['user'], svd_model, user_map, all_products, num_recs)
                    if recs:
                        st.success(" Top Personalized Recommendations based on Collaborative Filtering:")
                        for i, product_id in enumerate(recs, 1):
                            st.info(f"{i}. **{product_id}**")
                    else:
                        st.warning("Not enough user history to provide SVD recommendations.")
        else:
            st.warning("Collaborative filtering model files are missing or failed to load.")
            st.error(f"System Check Report: {svd_status}")
            
            st.write("###  Diagnostics & Auto-Fix")
            if os.path.exists(MODELS_DIR):
                files = os.listdir(MODELS_DIR)
                st.code(f"Contents of models/ folder:\n{files if files else 'Folder is empty'}")
            else:
                st.code("models/ folder does not exist.")
                
            if st.button(" Auto-Train Models Now"):
                with st.spinner("Training models in background... This may take up to a minute."):
                    script_path = os.path.join(BASE_DIR, "src", "train_svd.py")
                    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.cache_resource.clear()
                        st.success("  Models successfully generated! Reloading...")
                        st.rerun()
                    else:
                        st.error(f" Training Failed!\nError details:\n{result.stderr or result.stdout}")

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
        st.subheader("  Business Intelligence & Analytics")
        
        # High-level KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Unique Products", f"{df['ProductId'].nunique():,}")
        col3.metric("Average Rating", f"{df['Score'].mean():.2f} ")
        positive_pct = (len(df[df['Sentiment'] > 0.1]) / len(df)) * 100
        col4.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        
        st.divider()
        st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)
        st.info("The **Sentiment Polarity** score ranges from **-1.0** (Negative) to **+1.0** (Positive). By analyzing the text of reviews, we can understand the overall customer satisfaction beyond just the star rating.")
            
    # Tab 4: Trending
    with tabs[3]:
        st.subheader(" Product Performance Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_top_products(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_trending_products(df), use_container_width=True)

if __name__ == "__main__":
    main()
