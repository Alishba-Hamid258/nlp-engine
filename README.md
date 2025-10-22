NLP Engine
A comprehensive NLP application for Named Entity Recognition (NER), Sentiment Analysis, Aspect-Based Sentiment Analysis (ABSA), and Profanity Detection. Built with FastAPI for the backend, Streamlit for the frontend, and BERT for ABSA modeling.

Features

NER: Identifies entities (e.g., people, organizations) using SpaCy.

Sentiment Analysis: Computes polarity scores using NLTK's VADER.

Aspect-Based Sentiment Analysis: Analyzes sentiments for specific text aspects using a fine-tuned BERT model.

Profanity Detection: Flags profane words from a predefined list.

Visualizations: Displays results as HTML tables and entity highlights.

API: RESTful FastAPI endpoints for text analysis.

Frontend: Interactive Streamlit interface for text input and results.

File Structure
nlp_engine/
├── data/
│   └── aspect_sentiment_data.csv
├── models/
│   └── nlp_models.py
├── routes/
│   └── nlp_routes.py
├── api.py
├── config.py
├── frontend.py
├── requirements.txt
├── train_absa.py
├── .gitignore
└── README.md

Prerequisites

Python 3.8+

Git

A virtual environment (recommended)

Installation

Clone the repository:

git clone https://github.com/YOUR_GITHUB_USERNAME/nlp-engine.git

cd nlp-engine


Create and activate a virtual environment:

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


(Optional) Train the ABSA model:
python train_absa.py

This generates the model in models/absa_model/ (excluded from repo due to size; see Notes).

Usage
Backend (API)

Start the FastAPI server:
python api.py


Access the API at http://localhost:8000/api/analyze (POST).
Example request:{"text": "The battery life is amazing but the screen is damn blurry."}


Response includes NER, sentiment, ABSA, and profanity results.



Frontend

Start the Streamlit app:streamlit run frontend.py


Open http://localhost:8501 in your browser.
Enter text and click "Analyze" to view results (dataframes, visualizations).

Example Output
Input: "The battery life is amazing but the screen is damn blurry."Output:

NER: Entities like "battery life" (if detected).
Sentiment: Positive/negative/neutral scores.
ABSA: "battery life": Positive, "screen": Negative.
Profanity: Flags "damn".

Notes

Model Files: The trained BERT model (models/absa_model/) is not included due to size. Run train_absa.py to generate it.

Profanity: Expand PROFANE_WORDS in nlp_models.py for better detection.

API Port: Configurable via .env (default: 8000).

Security: No API authentication; add for production use.

Dependencies
See requirements.txt for details (e.g., FastAPI, Streamlit, Transformers, SpaCy, NLTK, Torch).

Contributing
Fork, make changes, and submit pull requests. Issues welcome!


