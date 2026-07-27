# SMS Spam Classifier

A machine learning project built with Python, Pandas, and scikit-learn that classifies SMS text messages as either **Spam** or **Ham** (Not Spam). 

This project uses Natural Language Processing (NLP) techniques to analyze text data and predict its category in real-time.

---

## 🧠 How it Works

The classifier is built using a pipeline of two classic NLP techniques:

1. **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency): 
   Converts the raw text messages into a numerical format that the machine learning algorithm can understand. It highlights important words while down-weighting common words like "the" or "is".
   
2. **Multinomial Naive Bayes (NB)**: 
   A probabilistic classifier based on applying Bayes' theorem. It's incredibly fast, highly scalable, and traditionally the standard baseline for text classification tasks like spam filtering.

---

## 📂 Project Structure

```
sms_spam_classifier/
├── data/
│   └── spam.csv          # Sample dataset of messages with spam/ham labels
├── data_loader.py        # Handles loading and cleaning text (removing punctuation, lowercasing)
├── model.py              # Contains the SpamClassifier class (Pipeline setup, training, evaluation)
├── main.py               # Interactive Command Line Interface (CLI)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3 installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the App

Start the interactive CLI to test the model:

```bash
python main.py
```

You will see the model train instantly and output an evaluation report (Precision, Recall, F1-Score). 
Then, you can type in any custom message and it will predict whether it's SPAM or HAM!

### Sample Output

```
==================================================
 SMS SPAM CLASSIFIER
 Powered by TF-IDF & Naive Bayes
==================================================

Loading dataset...
Loaded 82 SMS messages.
Training the model...

--- Model Evaluation Report ---
              precision    recall  f1-score   support

         ham       0.91      1.00      0.95        10
        spam       1.00      0.86      0.92         7

    accuracy                           0.94        17
   macro avg       0.95      0.93      0.94        17
weighted avg       0.95      0.94      0.94        17
-------------------------------

Model is ready!

Type an SMS message to check if it's SPAM or HAM.
Type 'exit' or 'quit' to close the program.

Enter message: Congratulations! You've won a $1000 gift card. Call now!
--> Result: [ SPAM ] 🚨 Be careful!
------------------------------
Enter message: Hey, what time are we grabbing coffee?
--> Result: [ HAM ] ✅ Safe message.
------------------------------
```

---

## 🛠️ Built With
- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
