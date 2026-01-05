# SMS Spam Detection Application

This project is a Machine Learning and NLP based web application that classifies SMS messages as Spam or Legitimate (Ham). The application uses the SMS Spam Collection Dataset and provides real-time predictions through an interactive Streamlit interface.

---

## Features

- Classifies SMS messages as Spam or Ham
- Real-time prediction using Machine Learning
- TF-IDF based text vectorization
- Support Vector Machine (SVM) classifier
- Displays evaluation metrics
- Clean and simple Streamlit UI

---

## How the Application Works

1. The user enters an SMS message.
2. The message is converted into numerical features using TF-IDF.
3. A trained Support Vector Machine (SVM) model predicts the class.
4. The result is displayed as Spam or Legitimate.

---

## Dataset Details

- Dataset Name: SMS Spam Collection Dataset
- Source: Kaggle
- Total Messages: 5,574
- Classes:
  - Ham (Legitimate)
  - Spam

Columns:
- v1: Message label (ham or spam)
- v2: SMS text

---

## Machine Learning Techniques Used

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Support Vector Machine (SVM)

---

## Evaluation Metrics

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help measure the effectiveness of the spam classification model.

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Project Structure

sms-spam-app/
│
├── app.py
├── spam.csv
├── requirements.txt
└── README.md

---

## Installation and Execution

1. Clone the repository:
   git clone https://github.com/koushikreddy30-26/task-2_spam_sms_detection.git

2. Navigate to the project directory:
   cd sms-spam-detection

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   streamlit run app.py

5. Open your browser and go to:
   http://localhost:8501

---

## Sample Messages for Testing

Spam Message:
Congratulations! You have won a FREE prize. Click now to claim.

Legitimate Message:
Hey, are we meeting at 6 pm today?

---

## Future Improvements

- Add more ML models for comparison
- Show prediction confidence score
- Deploy the application online
- Improve UI design

---

