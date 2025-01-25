import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained Logistic Regression model and CountVectorizer
model = joblib.load('log_reg.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags=re.MULTILINE)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Predictor", layout="wide")

# Custom CSS for dynamic background color
st.markdown(
    """
    <style>
    .positive {
        background-color: #1d8348; /* Dark green */
        color: #fdfefe; /* Light text */
    }
    .negative {
         background-color: #943126; /* Dark red */
        color: #fdfefe; /* Light text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sentiment Analysis Predictor: Machine Learning Project")

st.write("Enter a movie review below, and the model will predict whether the sentiment is **positive** or **negative**.")

# User input
user_input = st.text_area("Movie Review", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Preprocess the input
        cleaned_input = preprocess_text(user_input)

        # Convert input to feature vector
        input_vector = vectorizer.transform([cleaned_input])

        # Predict sentiment
        prediction = model.predict(input_vector)[0]

        # Determine sentiment and set background color
        sentiment = "Positive" if prediction == 1 else "Negative"
        background_class = "positive" if prediction == 1 else "negative"

        # Display result with dynamic background
        st.markdown(
            f"""
            <div class="{background_class}" style="padding: 20px; border-radius: 5px;">
                <h2 style="text-align: center;">The predicted sentiment is: <b>{sentiment}</b></h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a valid review.")

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align: center;">
        <b>Made by:</b> <b><br>Hardiksinh Solanki(249531990),</br></b>
                        <b>Dikshaben Patel(249432540)</b>
    </footer>
    """,
    unsafe_allow_html=True,
)
