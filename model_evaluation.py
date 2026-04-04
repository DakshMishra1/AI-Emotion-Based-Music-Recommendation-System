"""
Model Evaluation and Analysis Module
Provides detailed model evaluation, comparison, and visualization utilities
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis
    """
    
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Load trained model and vectorizer
        
        Args:
            model_path (str): Path to saved model
            vectorizer_path (str): Path to saved vectorizer
        """
        self.model = self.load_model(model_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)
        
    def load_model(self, model_path):
        """Load model from pickle file"""
        if not os.path.exists(model_path):
            print(f"⚠ Model file not found at {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
        return model
    
    def load_vectorizer(self, vectorizer_path):
        """Load vectorizer from pickle file"""
        if not os.path.exists(vectorizer_path):
            print(f"⚠ Vectorizer file not found at {vectorizer_path}")
            return None
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✓ Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    
    def evaluate_predictions(self, X_test, y_test, emotions):
        """
        Evaluate model predictions on test set
        
        Args:
            X_test (array): Test texts
            y_test (array): True labels
            emotions (list): List of emotion labels
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None or self.vectorizer is None:
            print("✗ Model or vectorizer not loaded")
            return None
        
        # Transform and predict
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_test, y_pred, emotions, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix
        
        Args:
            y_test (array): True labels
            y_pred (array): Predicted labels
            emotions (list): Emotion labels
            save_path (str): Path to save figure
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Emotion Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_classification_report(self, y_test, y_pred, emotions, save_path='classification_report.png'):
        """
        Plot classification report metrics per emotion
        
        Args:
            y_test (array): True labels
            y_pred (array): Predicted labels
            emotions (list): Emotion labels
            save_path (str): Path to save figure
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Calculate metrics per emotion
        precision = precision_score(y_test, y_pred, average=None, labels=emotions, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, labels=emotions, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, labels=emotions, zero_division=0)
        
        # Create visualization
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Report - Metrics per Emotion', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Classification report saved to {save_path}")
        plt.close()
    
    def plot_emotion_distribution(self, y_true, save_path='emotion_distribution.png'):
        """
        Plot distribution of emotions in dataset
        
        Args:
            y_true (array): True labels
            save_path (str): Path to save figure
        """
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        plt.bar(unique, counts, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Emotion Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for i, (emotion, count) in enumerate(zip(unique, counts)):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Emotion distribution saved to {save_path}")
        plt.close()
    
    def get_feature_importance(self, top_n=10):
        """
        Get top N important features (words) for each emotion
        
        Args:
            top_n (int): Number of top features to return
        
        Returns:
            dict: Top features per emotion
        """
        if self.model is None or self.vectorizer is None:
            print("✗ Model or vectorizer not loaded")
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_
        
        feature_importance = {}
        for idx, emotion in enumerate(self.model.classes_):
            top_indices = np.argsort(np.abs(coefficients[idx]))[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = [coefficients[idx][i] for i in top_indices]
            
            feature_importance[emotion] = {
                'features': top_features,
                'scores': top_scores
            }
        
        return feature_importance
    
    def print_feature_importance(self, top_n=10):
        """
        Print top important features for each emotion
        
        Args:
            top_n (int): Number of top features to display
        """
        importance = self.get_feature_importance(top_n)
        
        if importance is None:
            return
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} IMPORTANT FEATURES PER EMOTION")
        print(f"{'='*60}\n")
        
        for emotion, data in importance.items():
            print(f"🎯 {emotion.upper()}")
            print("-" * 40)
            for feature, score in zip(data['features'], data['scores']):
                print(f"  {feature:20} : {score:8.4f}")
            print()
    
    def print_evaluation_summary(self, metrics):
        """
        Print detailed evaluation summary
        
        Args:
            metrics (dict): Evaluation metrics dictionary
        """
        if metrics is None:
            return
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION SUMMARY")
        print(f"{'='*60}\n")
        
        print("📊 METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\n{'='*60}\n")


def main():
    """
    Main evaluation function
    """
    print("=" * 60)
    print("MODEL EVALUATION AND ANALYSIS")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    
    if evaluator.model is not None:
        print("\n✓ Model and vectorizer loaded successfully")
        print("Ready for evaluation with test data")
        
        # Example: Print feature importance
        evaluator.print_feature_importance(top_n=10)


if __name__ == "__main__":
    main()
