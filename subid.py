import streamlit as st
from transformers import pipeline


st.title("Subject Identifier")


text = st.text_area("Enter your paragraph:", 
                          value="Write your text here...", 
                          height=200, 
                          max_chars=500,
                          key="paragraph")
scan = st.button("Scan")

if scan:
    classifier = pipeline("zero-shot-classification")

    # User input for the paragraph
    user_paragraph = text

    # Define candidate labels representing different topics or subjects
    candidate_labels = ["educational", "sports", "technology", "science", "business", 
                    "politics", "entertainment", "health", "travel", "food"]  # Added labels

    # Classify the input paragraph into one of the candidate labels
    classification_result = classifier(user_paragraph, candidate_labels)

    # Get the predicted label with the highest score
    predicted_label = classification_result['labels'][0]

    st.write(f"Text is talking about  {predicted_label}")
