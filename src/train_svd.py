import pandas as pd
import pickle
import os
import sys

try:
    from surprise import Reader, Dataset, SVD, accuracy
    from surprise.model_selection import train_test_split
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
    
    # Filter for density: SVD needs users and products with multiple ratings to find patterns
    print("Filtering out sparse data to improve accuracy...")
    min_user_ratings = 3
    min_product_ratings = 3
    
    df = df[df.groupby('UserId')['UserId'].transform('size') >= min_user_ratings]
    df = df[df.groupby('ProductId')['ProductId'].transform('size') >= min_product_ratings]

    # Taking a larger sample of 50,000 rows for better accuracy
    if len(df) > 50000:
        df_sample = df.sample(n=50000, random_state=42)
    else:
        df_sample = df

    print("Training SVD Model (this might take a few seconds)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_sample[['UserId', 'ProductId', 'Score']], reader)
    
    print("Evaluating Model Accuracy (Train/Test Split)...")
    trainset_eval, testset_eval = train_test_split(data, test_size=0.2, random_state=42)
    eval_model = SVD(n_factors=50, reg_all=0.05)  # Tuned hyperparameters
    eval_model.fit(trainset_eval)
    predictions = eval_model.test(testset_eval)
    print("Model Evaluation Metrics:")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    print("Training Final SVD Model on full dataset...")
    trainset = data.build_full_trainset()
    
    svd_model = SVD(n_factors=50, reg_all=0.05)
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