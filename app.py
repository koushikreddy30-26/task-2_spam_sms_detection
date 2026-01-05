import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)
st.title("📩 SMS Spam Detection App")
st.markdown("""
This application uses **Machine Learning & NLP** to classify SMS messages as  
**Spam 🚫** or **Legitimate (Ham) ✅**.

### 🔍 How it works:
- Converts text into numbers using **TF-IDF**
- Uses **Support Vector Machine (SVM)** for classification
""")

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()
@st.cache_resource
@st.cache_resource
def train_model():
    X = df['message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words='english', min_df=5)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    return tfidf, model, y_test, y_pred


tfidf, model, y_test, y_pred = train_model()


st.subheader("✍️ Enter an SMS message")

user_input = st.text_area(
    "Type your message here:",
    height=150,
    placeholder="Enter the SMS message you want to classify..."
)

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        input_vector = tfidf.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is LEGITIMATE")


st.markdown("---")
st.markdown("👨‍💻 Built using **TF-IDF + SVM** | Dataset: Kaggle SMS Spam Collection")
