"""
Data Preprocessing Module for Emotion-Based Music Recommendation System
Handles text cleaning, tokenization, and feature extraction
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def remove_special_characters(text):
    """
    Remove special characters, URLs, and extra whitespace
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def convert_to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


def remove_stopwords(tokens):
    """
    Remove common stopwords from tokens
    
    Args:
        tokens (list): List of word tokens
    
    Returns:
        list: Filtered tokens without stopwords
    """
    return [token for token in tokens if token not in stop_words]


def lemmatize_tokens(tokens):
    """
    Apply lemmatization to normalize word forms
    
    Args:
        tokens (list): List of word tokens
    
    Returns:
        list: Lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize_text(text):
    """
    Tokenize text into individual words
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of word tokens
    """
    return word_tokenize(text)


def preprocess_text(text):
    """
    Complete preprocessing pipeline for a single text sample
    
    Args:
        text (str): Raw input text
    
    Returns:
        str: Fully preprocessed text
    """
    # Step 1: Remove special characters
    text = remove_special_characters(text)
    
    # Step 2: Convert to lowercase
    text = convert_to_lowercase(text)
    
    # Step 3: Tokenize
    tokens = tokenize_text(text)
    
    # Step 4: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Step 5: Lemmatize
    tokens = lemmatize_tokens(tokens)
    
    # Join tokens back to string
    return ' '.join(tokens)


def preprocess_dataset(dataframe, text_column='text'):
    """
    Preprocess entire dataset
    
    Args:
        dataframe (pd.DataFrame): Input dataset with text column
        text_column (str): Name of the column containing text data
    
    Returns:
        pd.DataFrame: Dataset with preprocessed text
    """
    dataframe[text_column] = dataframe[text_column].apply(preprocess_text)
    return dataframe


def get_preprocessing_summary():
    """Print summary of preprocessing operations"""
    print("""
    ============================================
    TEXT PREPROCESSING PIPELINE SUMMARY
    ============================================
    
    Operations performed:
    1. Remove special characters, URLs, emails
    2. Convert to lowercase
    3. Tokenization
    4. Remove stopwords
    5. Lemmatization
    
    ============================================
    """)


if __name__ == "__main__":
    # Test the preprocessing functions
    get_preprocessing_summary()
    
    sample_text = "Hey! I'm feeling AMAZING today!!! Check out https://example.com #happy 😊"
    print(f"\nOriginal text: {sample_text}")
    print(f"Preprocessed: {preprocess_text(sample_text)}")
