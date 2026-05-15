#  ReviewSense AI

Welcome to **ReviewSense AI**! This is a production-ready, machine-learning-driven web application that provides personalized product recommendations using a combination of **Collaborative Filtering (SVD)** and **Content-Based Filtering (NLP)**.

Recently upgraded to a professional standard, this project now features a secure **SQLite-based authentication system**, natural language review processing using **TF-IDF & Cosine Similarity**, and a premium interactive dashboard with business KPIs, built using **Streamlit** and **Plotly**.

## Features

- ** Premium UI & Business KPIs**: A custom-styled, clean interface that displays high-level metrics like Total Reviews, Average Ratings, and Positive Sentiment percentages.
- **Secure User Authentication**: A robust Login and Signup system backed by a local **SQLite database** and SHA-256 password hashing.
- **  Auto-Healing SVD Engine**: A collaborative filtering engine that recommends products based on user rating history, complete with an **Auto-Train** feature that automatically generates missing models in the background.
- **  Content-Based NLP Recommendations**: Suggests similar products by analyzing the textual similarity of reviews and summaries using TF-IDF Vectorization.
- **  Sentiment Analysis**: Evaluates customer reviews using `TextBlob` to assign polarity scores, easily differentiating genuinely positive feedback from negative ones.
- **  Analytics Dashboard**: Interactive Plotly charts visualizing sentiment distributions, top-rated products, and trending items.

##   Project Structure

```text
ReviewSense-AI/
├── app/
│   └── main.py                      # Main Streamlit dashboard and UI logic
├── data/
│   ├── Reviews.csv                  # Dataset containing product reviews
│   └── users.db                     # SQLite database for secure user credentials
├── models/                          
│   ├── all_products.pkl             # Serialized list of all products
│   ├── svd_recommender_model.pkl    # Trained SVD Collaborative Filtering model
│   └── user_rated_products_map.pkl  # User-to-product rating mapping
├── src/
│   ├── __init__.py
│   ├── recommender.py               # Core NLP TF-IDF Cosine Similarity logic
│   ├── sentiment.py                 # Sentiment analysis model using TextBlob
│   └── train_svd.py                 # Script to automatically train/generate SVD models
├── visuals/
│   ├── __init__.py
│   └── charts.py                    # Plotly interactive charting functions
├── .gitignore                       # Git ignore file for security and cache exclusion
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

##  Tech Stack

- **Language:** Python 3.x
- **Frontend/UI:** Streamlit (with Custom CSS)
- **Machine Learning / NLP:** Scikit-Learn, scikit-surprise (SVD), TextBlob
- **Data Manipulation:** Pandas, NumPy
- **Database:** SQLite
- **Visualizations:** Plotly Express

##  Installation & Setup

Follow these steps to run the application on your local machine.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ReviewSense-AI.git
   cd ReviewSense-AI
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app/main.py
   ```

##  Future Enhancements

- Build a **Weighted Hybrid Recommendation Engine** combining both NLP similarity and SVD estimations into a single score.
- Implement caching for the pre-trained TF-IDF vectorizer matrix to further reduce app startup overhead.
- Deploy the application seamlessly to **Streamlit Community Cloud** or AWS/GCP.

---
*Developed by Aditi Tiwari*
