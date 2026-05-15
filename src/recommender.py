import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NLPRecommender:
    """
    Content-Based NLP Recommender using TF-IDF and Cosine Similarity.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame):
        """
        Fits the TF-IDF vectorizer on the combined text features.
        """
        self.df = df.copy().reset_index(drop=True)
        self.df['Summary'] = self.df['Summary'].fillna('')
        self.df['Text'] = self.df['Text'].fillna('')
        
        # Combining summary and review text for rich feature extraction
        self.df['combined_features'] = self.df['Summary'] + " " + self.df['Text']
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])

    def recommend(self, product_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Returns top_n similar products based on cosine similarity of reviews.
        """
        if product_id not in self.df['ProductId'].values:
            return pd.DataFrame()
        
        # Get index of the product
        idx = self.df[self.df['ProductId'] == product_id].index[0]
        
        # Compute similarities
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
        
        # Sort by highest similarity, ignoring the item itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        product_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        # Prepare recommendation dataframe
        rec_df = self.df.iloc[product_indices].copy()
        rec_df['SimilarityScore'] = scores
        
        # Deduplicate recommendations by ProductId in case of multiple reviews for the same product
        return rec_df.drop_duplicates(subset=['ProductId']).head(top_n)
