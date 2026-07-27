import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SpamClassifier:
    def __init__(self):
        # We use a pipeline that first vectorizes the text (TF-IDF) and then applies Naive Bayes
        self.model = make_pipeline(
            TfidfVectorizer(stop_words='english'),
            MultinomialNB()
        )
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Train the classifier using the provided DataFrame."""
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")
            
        X = df['text']
        y = df['label']
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True

    def evaluate(self, df: pd.DataFrame):
        """Evaluate the model and print a classification report."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
            
        X = df['text']
        y = df['label']
        
        # Split into training and testing just for evaluation stats
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train on the split
        temp_model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
        temp_model.fit(X_train, y_train)
        
        # Predict and print
        y_pred = temp_model.predict(X_test)
        print("\n--- Model Evaluation Report ---")
        print(classification_report(y_test, y_pred))
        print("-------------------------------\n")

    def predict(self, text: str) -> str:
        """Predict whether a single message is 'spam' or 'ham'."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        
        # Pipeline expects an iterable of strings
        prediction = self.model.predict([text])
        return prediction[0]
