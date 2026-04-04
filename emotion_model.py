"""
Emotion Detection Model Training Module
Trains a Logistic Regression model with TF-IDF vectorization
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import os
from data_preprocessing import preprocess_dataset


class EmotionModel:
    """
    Emotion Detection Model using TF-IDF + Logistic Regression
    """
    
    def __init__(self, random_state=42, max_features=5000, max_iter=1000):
        """
        Initialize the model components
        
        Args:
            random_state (int): Random seed for reproducibility
            max_features (int): Maximum number of features for TF-IDF
            max_iter (int): Maximum iterations for Logistic Regression
        """
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )
        self.emotions = None
        self.accuracy = None
        self.metrics = {}
        
    def load_dataset(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        return df
    
    def prepare_data(self, dataset, text_column='text', label_column='emotion'):
        """
        Prepare and preprocess data
        
        Args:
            dataset (pd.DataFrame): Raw dataset
            text_column (str): Name of text column
            label_column (str): Name of label column
        
        Returns:
            tuple: Preprocessed texts and labels
        """
        print(f"\n🔄 Preprocessing dataset...")
        
        # Check if columns exist
        if text_column not in dataset.columns or label_column not in dataset.columns:
            raise ValueError(f"Columns '{text_column}' or '{label_column}' not found in dataset")
        
        # Remove null values
        dataset = dataset.dropna(subset=[text_column, label_column])
        print(f"  - Samples after removing nulls: {len(dataset)}")
        
        # Preprocess texts
        dataset = preprocess_dataset(dataset, text_column)
        
        # Store unique emotions
        self.emotions = dataset[label_column].unique()
        print(f"  - Unique emotions: {list(self.emotions)}")
        print(f"  - Emotion distribution:\n{dataset[label_column].value_counts()}")
        
        return dataset[text_column].values, dataset[label_column].values
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        
        Args:
            X (array): Features (texts)
            y (array): Labels (emotions)
            test_size (float): Proportion of test set
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"  - Training set: {len(X_train)} samples")
        print(f"  - Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train TF-IDF vectorizer and Logistic Regression model
        
        Args:
            X_train (array): Training texts
            y_train (array): Training labels
        """
        print(f"\n🚀 Training model...")
        
        # Fit TF-IDF vectorizer
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  - TF-IDF vectorizer fitted")
        print(f"  - Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train logistic regression model
        self.model.fit(X_train_tfidf, y_train)
        print(f"  - Logistic Regression model trained ✓")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test (array): Test texts
            y_test (array): Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n📈 Evaluating model...")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        self.accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.metrics = {
            'accuracy': self.accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"  - Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📋 Confusion Matrix:")
        print(cm)
        
        # Print classification report
        print(f"\n📝 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.emotions))
        
        return self.metrics
    
    def save_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Save trained model and vectorizer to disk
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved to {model_path}")
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"✓ Vectorizer saved to {vectorizer_path}")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    
    def predict_emotion(self, text):
        """
        Predict emotion for a given text
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Predicted emotion and confidence scores
        """
        from data_preprocessing import preprocess_text
        
        # Preprocess input
        processed_text = preprocess_text(text)
        
        # Vectorize
        X_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        emotion = self.model.predict(X_tfidf)[0]
        probabilities = self.model.predict_proba(X_tfidf)[0]
        
        # Create result dictionary
        result = {
            'predicted_emotion': emotion,
            'confidence': float(max(probabilities)),
            'all_probabilities': {
                emotion_label: float(prob)
                for emotion_label, prob in zip(self.emotions, probabilities)
            }
        }
        
        return result
    
    def print_summary(self):
        """Print model training summary"""
        print("""
        ================================================
                    MODEL TRAINING SUMMARY
        ================================================
        
        Algorithm: Logistic Regression
        Vectorization: TF-IDF
        
        Features:
        - TF-IDF max features: 5000
        - N-gram range: (1, 2)
        - Min document frequency: 2
        - Max document frequency: 0.8
        
        Model Parameters:
        - Max iterations: 1000
        - Multi-class: multinomial
        
        """)
        
        if self.metrics:
            print("Performance Metrics:")
            for metric, value in self.metrics.items():
                print(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("================================================")


def main():
    """
    Main function to train and save the emotion model
    """
    print("=" * 60)
    print("EMOTION-BASED MUSIC RECOMMENDATION SYSTEM")
    print("ML Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Initialize model
        emotion_model = EmotionModel()
        emotion_model.print_summary()
        
        # Load dataset
        print("\n📁 Loading dataset...")
        dataset = emotion_model.load_dataset('dataset.csv')
        
        # Prepare data
        X, y = emotion_model.prepare_data(dataset, text_column='text', label_column='emotion')
        
        # Split data
        X_train, X_test, y_train, y_test = emotion_model.split_data(X, y, test_size=0.2)
        
        # Train model
        emotion_model.train(X_train, y_train)
        
        # Evaluate model
        emotion_model.evaluate(X_test, y_test)
        
        # Save model
        print("\n💾 Saving model and vectorizer...")
        emotion_model.save_model()
        
        # Test prediction
        print("\n🧪 Testing model with sample input...")
        test_input = "I am very happy and excited today!"
        result = emotion_model.predict_emotion(test_input)
        print(f"Input: '{test_input}'")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print("\n✓ Model training completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please ensure 'dataset.csv' exists in the project folder")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
