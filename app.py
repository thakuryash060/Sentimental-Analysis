import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline

# Load the saved model and tokenizer
@st.cache_resource
def load_local_model():
    # Load the tokenizer and model from your saved directory
    tokenizer = RobertaTokenizer.from_pretrained("C:\\Users\\thaku\\Desktop\\Teachnook_AI\\hit_and_trail\\roberta_saved_model")
    model = RobertaForSequenceClassification.from_pretrained("C:\\Users\\thaku\\Desktop\\Teachnook_AI\\hit_and_trail\\roberta_saved_model")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Map the model's label output to human-readable sentiments
label_mapping = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

# Function to perform sentiment analysis and map the output labels
def analyze_sentiment(text):
    model_pipeline = load_local_model()
    raw_result = model_pipeline(text)
    
    # Map the label to POSITIVE, NEUTRAL, or NEGATIVE
    for result in raw_result:
        result["label"] = label_mapping.get(result["label"], result["label"])  # Map label
    
    return raw_result

# Streamlit UI
st.title("ROBERT Sentiment Analysis (Custom Model)")
st.write("Enter a sentence to analyze its sentiment using your custom-trained model.")

# Text input from the user
user_input = st.text_area("Input Text", "")

# Analyze button
if st.button("Analyze"):
    if user_input:
        result = analyze_sentiment(user_input)
        st.write("Sentiment Analysis Result:", result)
    else:
        st.write("Please enter some text for analysis.")
