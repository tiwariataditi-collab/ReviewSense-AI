# 🛍️ ReviewSense AI

A professional, machine-learning-driven web application that provides personalized product suggestions using **Collaborative Filtering (SVD)** and **Content-Based Filtering (NLP)**. 

This project features a secure authentication system, natural language review processing using **TF-IDF & Cosine Similarity**, and a rich interactive dashboard built with **Streamlit** and **Plotly** to provide insightful product analytics.

## 🚀 Features

- **🔐 User Authentication**: Secure Login and Signup system with SHA-256 password hashing.
- **🤖 Personalized Picks (SVD)**: Collaborative filtering engine that recommends products based on a user's past rating history.
- **🔍 Content-Based NLP Recommendations**: Suggests similar products by analyzing the textual similarity of product summaries and reviews using TF-IDF Vectorization.
- **🎭 Sentiment Analysis**: Evaluates customer reviews using `TextBlob` to assign polarity scores, helping differentiate genuinely positive feedback from sarcasm or mixed reviews.
- **📊 Analytics Dashboard**: Interactive histograms and bar charts visualizing sentiment distribution, top-rated products, and trending products.

## 📂 Project Structure

```text
ReviewSense-AI/
├── app/
│   └── main.py                      # Main Streamlit dashboard and UI logic
├── data/
│   └── Reviews.csv                  # Dataset containing product reviews
├── models/                          
│   ├── all_products.pkl             # Serialized list of all products
│   ├── svd_recommender_model.pkl    # Trained SVD Collaborative Filtering model
│   └── user_rated_products_map.pkl  # User-to-product rating mapping
├── src/
│   ├── __init__.py
│   ├── recommender.py               # Core NLP TF-IDF Cosine Similarity logic
│   └── sentiment.py                 # Sentiment analysis model using TextBlob
├── visuals/
│   ├── __init__.py
│   └── charts.py                    # Plotly interactive charting functions
├── users.json                       # Local JSON database for user credentials
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Frontend/UI:** Streamlit
- **Machine Learning / NLP:** Scikit-Learn (TF-IDF, Cosine Similarity), TextBlob
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express

## ⚙️ Installation & Setup

Follow these steps to run the application on your local machine.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ReviewSense-AI.git
   cd ReviewSense-AI
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app/main.py
   ```

## 🔮 Future Enhancements

- Integrate Collaborative Filtering (using `scikit-surprise`) for a Hybrid Recommendation System.
- Save and load the pre-trained TF-IDF vectorizer matrix to reduce startup overhead.
- Connect to an external SQL/NoSQL database for real-time review additions.

---
*Developed by Aditi Tiwari*
