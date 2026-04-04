# ML Engineering - Student 1 Quick Start Guide

## 📋 Project Overview

This directory contains all the machine learning components for the Emotion-Based Music Recommendation System. Your responsibilities include emotion detection model development and training.

---

## 🎯 Your Tasks (Student 1 - ML Engineer)

### Task 1: Dataset Collection & Cleaning ✓
- **File**: `dataset.csv`
- **Status**: Sample dataset provided with emotion labels
- **Format**: CSV with columns: `text` (input), `emotion` (label: joy, sadness, anger, fear)
- **Steps**:
  1. Collect emotion-labeled text data
  2. Clean and standardize the data
  3. Remove duplicates and null values
  4. Add more samples for better accuracy

### Task 2: Text Preprocessing ✓
- **File**: `data_preprocessing.py`
- **Functions**:
  - `remove_special_characters()` - Clean URLs, emails, special chars
  - `convert_to_lowercase()` - Normalize text case
  - `tokenize_text()` - Split text into words
  - `remove_stopwords()` - Remove common words
  - `lemmatize_tokens()` - Normalize word forms
  - `preprocess_text()` - Complete pipeline

### Task 3: Apply TF-IDF ✓
- **Implementation**: In `emotion_model.py`
- **Parameters**:
  - Max features: 5000
  - N-grams: (1, 2) - unigrams and bigrams
  - Min doc frequency: 2
  - Max doc frequency: 80%
- **Purpose**: Convert text to numerical vectors

### Task 4: Train Model (Logistic Regression) ✓
- **File**: `emotion_model.py`
- **Class**: `EmotionModel`
- **Algorithm**: Logistic Regression with multinomial classification
- **Training Process**:
  1. Load and preprocess dataset
  2. Split data (80% train, 20% test)
  3. Fit TF-IDF vectorizer
  4. Train logistic regression model

### Task 5: Evaluate Accuracy ✓
- **File**: `model_evaluation.py`
- **Metrics Calculated**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - Classification Report

### Task 6: Save Model ✓
- **Output Files**:
  - `model.pkl` - Trained logistic regression model
  - `vectorizer.pkl` - TF-IDF vectorizer
- **Method**: Python pickle serialization

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd "path/to/EMOTION BASED MUSIC RECOMMENDATION SYSTEM"

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Run the emotion model training
python emotion_model.py
```

**Expected Output**:
- Dataset loading confirmation
- Preprocessing progress
- Train/test split info
- Training progress
- Model performance metrics
- Saved models: `model.pkl`, `vectorizer.pkl`

### 3. Evaluate Model

```bash
# Evaluate trained model
python model_evaluation.py
```

**Output**:
- Loads saved model and vectorizer
- Provides evaluation utilities
- Feature importance analysis

### 4. Test Prediction

```python
from emotion_model import EmotionModel
from data_preprocessing import preprocess_text

# Load model
model = EmotionModel()
model.model = pickle.load(open('model.pkl', 'rb'))
model.vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Test prediction
result = model.predict_emotion("I am very happy!")
print(result)
```

---

## 📁 File Structure

```
ML Backend Files:
├── data_preprocessing.py      # Text preprocessing functions
├── emotion_model.py           # Model training (TF-IDF + Logistic Regression)
├── model_evaluation.py        # Model evaluation and analysis
├── dataset.csv               # Sample dataset (update with real data)
├── model.pkl                 # Trained model (generated after training)
├── vectorizer.pkl            # TF-IDF vectorizer (generated after training)
└── requirements.txt          # Python dependencies
```

---

## 📊 Dataset Format

Your `dataset.csv` should have exactly 2 columns:

| text | emotion |
|------|---------|
| "I am feeling super happy!" | joy |
| "This is the worst day ever" | sadness |
| "I am so angry right now!" | anger |
| "I am terrified and scared" | fear |

### Supported Emotions:
1. **joy** - Happy, excited, pleased
2. **sadness** - Sad, depressed, blue
3. **anger** - Angry, furious, enraged
4. **fear** - Afraid, scared, anxious

---

## 🔧 Configuration Parameters

You can modify these in `emotion_model.py`:

```python
EmotionModel(
    random_state=42,        # Seed for reproducibility
    max_features=5000,      # Max TF-IDF features
    max_iter=1000          # Max Logisitc Regression iterations
)
```

---

## 📈 Expected Performance

With the sample dataset:
- **Accuracy**: 70-85% (depends on dataset size and quality)
- **Precision**: 0.70-0.85
- **Recall**: 0.70-0.85
- **F1-Score**: 0.70-0.85

To improve performance:
1. **Increase dataset size**: More samples = better accuracy
2. **Balance emotions**: Equal representation for each emotion
3. **Improve data quality**: Remove noisy/ambiguous samples
4. **Tune hyperparameters**: Experiment with different values

---

## 🐛 Troubleshooting

### Issue: "nltk data not found"
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: "dataset.csv not found"
- Ensure `dataset.csv` is in the same directory as `emotion_model.py`
- Check file path and column names

### Issue: Low accuracy
- Collect more training data
- Balance emotion distribution
- Review and clean dataset
- Try different preprocessing techniques

---

## 📚 Key Concepts

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Converts text to numerical vectors
- Weights important words higher
- Removes common/rare terms
- Better than simple word count

### Logistic Regression
- Binary/multiclass classification algorithm
- Fast and interpretable
- Works well with TF-IDF features
- Provides probability scores

### Train-Test Split
- 80% data for training
- 20% data for validation
- Prevents overfitting
- Stratified to maintain class balance

---

## ✅ Checklist

Complete each task in order:

- [ ] Setup virtual environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Review and expand `dataset.csv`
- [ ] Run `python emotion_model.py` to train
- [ ] Check if `model.pkl` and `vectorizer.pkl` are created
- [ ] Review performance metrics
- [ ] Run `python model_evaluation.py` for detailed analysis
- [ ] Test with sample predictions
- [ ] Push code to GitHub

---

## 🔗 Integration with Student 2 (Frontend)

Your trained models (`model.pkl`, `vectorizer.pkl`) will be used by:
- Your friend's Flask API to receive text
- Return predictions to the frontend
- Processed emotions sent to Spotify recommendation engine

Make sure models are well-validated before handoff!

---

## 📞 Support Resources

- Scikit-learn docs: https://scikit-learn.org/
- NLTK docs: https://www.nltk.org/
- TF-IDF explanation: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression

---

## 🎯 Next Steps

1. ✓ Complete the tasks above
2. ✓ Achieve >80% accuracy on test set
3. ✓ Document any data preprocessing decisions
4. ✓ Commit code to GitHub
5. → Handoff to Student 2 for integration

Good luck! 🚀
