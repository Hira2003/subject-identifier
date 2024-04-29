import pandas as pd
import streamlit as st
from transformers import pipeline

# web page setting
st.set_page_config(page_title="Subject Identifier", page_icon="📚", layout="wide")

# Title
st.title("Subject Identifier")
st.divider()

# Sidebar
side_bar = st.sidebar
with side_bar:
	st.title("About", anchor="about")
	st.divider()
	st.write("This app uses the Zero-shot classification model to identify the subject of a given paragraph.")

# Main content
r1c1, r1c2 = st.columns([3, 2])

with r1c1:
	input_text = st.text_area("Enter your paragraph:", value="Write your text here...", height=200, max_chars=500, key="paragraph")
	scan = st.button("Scan")
	if scan:
		classifier = st.cache_data(pipeline)("zero-shot-classification")
		# classifier = pipeline("zero-shot-classification")

		# Define candidate labels representing different topics or subjects
		candidate_labels = ["educational", "sports", "technology", "science", "business", "politics", "entertainment", "health", "travel", "food"]  # Added labels

		# Classify the input paragraph into one of the candidate labels
		classification_result = classifier(input_text, candidate_labels)

		# Get the predicted label with the highest score
		predicted_label = classification_result['labels'][0]
		df = pd.DataFrame(classification_result)

		st.subheader(f"""Text is talking about: """, divider='rainbow')
		st.subheader(predicted_label)

with r1c2:
	if scan:
		st.dataframe(data=df[['labels', 'scores']], hide_index=True, use_container_width=True)
