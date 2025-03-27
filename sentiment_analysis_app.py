import streamlit as st
import pickle
import numpy as np

# Load the trained model, vectorizer, and label encoder
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stTextArea textarea { 
            font-size: 16px; 
            border-radius: 10px; 
            padding: 10px;
        }
        .sentiment-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            margin-top: 20px;
        }
        .positive { background-color: #d4edda; color: #155724; }
        .neutral { background-color: #fff3cd; color: #856404; }
        .negative { background-color: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("💬 Sentiment Analysis App")
st.write("🔍 Enter a tweet to analyze its sentiment.")

# User input
user_input = st.text_area("✏️ Type your tweet here:")

if st.button("🔮 Predict Sentiment"):
    if user_input:
        # Transform input text
        input_tfidf = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_tfidf)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        # Define colors & emojis for sentiments
        sentiment_map = {
            "positive": ("🟢 Positive 😊", "positive"),
            "neutral": ("🟡 Neutral 😐", "neutral"),
            "negative": ("🔴 Negative 😠", "negative")
        }
        
        # Display result
        label, css_class = sentiment_map.get(sentiment, ("Unknown", "neutral"))
        st.markdown(f'<div class="sentiment-box {css_class}">{label}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a tweet to analyze.")

