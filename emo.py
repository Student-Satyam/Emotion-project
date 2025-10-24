import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define your emotion mapping (0–5)
emotion_labels = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

# Emoji map for each emotion
emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️"
}

# Streamlit app
st.set_page_config(page_title="Emotion Detection", page_icon="🎭")
st.title("🎭 Sattu's Emotion Detection from Text")
st.write("Enter a sentence and the satyam's model will predict the emotion it conveys.")

# User input
text = st.text_area("Type your text here:")

if st.button("Predict Emotion"):
    if text.strip():
        # Transform and predict
        X = vectorizer.transform([text])
        pred_num = model.predict(X)[0]
        pred_label = emotion_labels[pred_num]  # number -> emotion name

        proba = model.predict_proba(X)[0]
        prob_df = pd.DataFrame({
            "Emotion": [emotion_labels[i] for i in model.classes_],
            "Probability": proba
        }).set_index("Emotion")

        # Display predicted emotion with emoji
        st.subheader(f"Predicted Emotion: {emoji_map.get(pred_label,'')} **{pred_label.capitalize()}**")

        # Probability bar chart
        st.bar_chart(prob_df)
    else:
        st.warning("Please enter some text first.")
