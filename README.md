# 🎵 AI Emotion-Based Music Recommendation System

## 📌 Project Description

The **AI Emotion-Based Music Recommendation System** is a machine learning project that analyzes a user's emotional state based on text input and recommends suitable music accordingly. The system uses Natural Language Processing (NLP) techniques to classify emotions and provides personalized song suggestions to enhance the user’s mood.

---

## 💡 Features

* 🎯 Detects user emotions from text input
* 🎵 Recommends songs based on detected emotion using Spotify API
* 🤖 Uses Machine Learning (TF-IDF + Logistic Regression)
* 📊 Mood tracking and history visualization
* 🧠 AI-based suggestions for improving mood
* 💾 Persistent mood tracking database
* 📈 Visual analytics of emotional patterns over time

---

## 🧠 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* NLTK
* Spotify Web API
* Flask (or Django) for backend API
* Database (SQLite/PostgreSQL) for mood history
* Matplotlib/Seaborn for visualization

---

## ⚙️ How It Works

1. User enters a sentence describing their current mood
2. The system preprocesses the text using NLP techniques
3. A trained ML model predicts the emotion category
4. Based on the emotion, Spotify API is queried for relevant songs
5. Mood entry is stored in the database with timestamp
6. System analyzes mood history and identifies patterns
7. AI generates personalized suggestions for mood improvement
8. Results including music recommendations and mood analytics are displayed
9. User can view mood tracking history and visualization charts

---

## 📂 Project Structure

```
music-emotion-project/
│
├── dataset.csv
├── model.pkl
├── app.py
├── emotion_model.py
├── recommender.py
├── spotify_integration.py
├── mood_tracker.py
├── mood_history.db
├── analytics.py
├── ai_suggestions.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

1. Clone the repository

```
git clone https://github.com/your-username/music-emotion-project.git
```

2. Navigate to the project folder

```
cd music-emotion-project
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Set up Spotify API credentials
   - Create a Spotify Developer account at https://developer.spotify.com/
   - Create an application and get your Client ID and Secret
   - Set environment variables: `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`

5. Run the application

```
python app.py
```

The application will start a local server (typically at http://localhost:5000) where you can interact with the system.

---

## 🎯 Future Enhancements

* Voice-based emotion detection from audio input
* Advanced deep learning models (BERT, RoBERTa) for better emotion classification accuracy
* Multi-language emotion detection support
* Real-time mood prediction from facial expressions using computer vision
* Social features: Share mood stories and music recommendations with friends
* Integrations with other music platforms (YouTube Music, Apple Music)
* Wearable device integration for automatic mood tracking
* Collaborative filtering for group music recommendations

---

## 👨‍💻 Contributors

* Piyush Singh Jimiwal
* Daksh Mishra 

---

## 📌 Conclusion

This project demonstrates the use of Machine Learning and NLP to build a real-world application that enhances user experience by combining emotion detection with personalized music recommendations.

---
