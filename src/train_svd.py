import pandas as pd
import pickle
import os
import sys

try:
    from surprise import Reader, Dataset, SVD
except ImportError:
    print("Error: 'scikit-surprise' is not installed. Please run: pip install scikit-surprise")
    sys.exit(1)

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE = os.path.join(BASE_DIR, "data", "Reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_and_save_svd():
    print(f"Loading dataset from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found. Please ensure 'Reviews.csv' is inside {DATA_FILE}")
        return

    print("Preparing data...")
    # Drop missing rows
    df = df.dropna(subset=['UserId', 'ProductId', 'Score'])
    
    # Taking a sample of 10,000 rows for fast training (you can increase this later)
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df

    print("Training SVD Model (this might take a few seconds)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_sample[['UserId', 'ProductId', 'Score']], reader)
    trainset = data.build_full_trainset()
    
    svd_model = SVD()
    svd_model.fit(trainset)

    print("Generating user maps and product lists...")
    user_map = df_sample.groupby('UserId')['ProductId'].apply(list).to_dict()
    all_products = df_sample['ProductId'].unique().tolist()

    print(f"Saving model files to {MODELS_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(os.path.join(MODELS_DIR, "svd_recommender_model.pkl"), "wb") as f: pickle.dump(svd_model, f)
    with open(os.path.join(MODELS_DIR, "user_rated_products_map.pkl"), "wb") as f: pickle.dump(user_map, f)
    with open(os.path.join(MODELS_DIR, "all_products.pkl"), "wb") as f: pickle.dump(all_products, f)
    print("✅ Models successfully generated! You can now restart your Streamlit app.")

if __name__ == "__main__":
    train_and_save_svd()