
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
from spacy import displacy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Download NLTK data if not already done
download('vader_lexicon', quiet=True)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load trained ABSA model
absa_tokenizer = BertTokenizer.from_pretrained("models/absa_model")
absa_model = BertForSequenceClassification.from_pretrained("models/absa_model")
absa_model.eval()

# Simple profanity list (expand as needed)
PROFANE_WORDS = {"badword", "curse", "profanity", "damn", "hell"}

def perform_ner(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
    return entities

def perform_sentiment_analysis(text: str):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def visualize_sentiment(scores):
    html = f"""
    <table style="border-collapse: collapse; width: 50%;">
        <tr style="background-color: #4CAF50; color: white;">
            <th style="padding: 8px; border: 1px solid #ddd;">Sentiment Type</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Score</th>
        </tr>
        <tr style="border: 1px solid #ddd;">
            <td style="padding: 8px;">Negative</td>
            <td style="padding: 8px;">{scores['neg']:.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd;">
            <td style="padding: 8px;">Neutral</td>
            <td style="padding: 8px;">{scores['neu']:.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd;">
            <td style="padding: 8px;">Positive</td>
            <td style="padding: 8px;">{scores['pos']:.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd;">
            <td style="padding: 8px;">Compound</td>
            <td style="padding: 8px;">{scores['compound']:.2f}</td>
        </tr>
    </table>
    """
    return html

def create_sentiment_dataframe(scores):
    df_data = {
        "Sentiment Type": ["Negative", "Neutral", "Positive", "Compound"],
        "Score": [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
    }
    df = pd.DataFrame(df_data)
    return df.to_dict('records')

def perform_aspect_based_sentiment(text: str):
    doc = nlp(text)
    aspects = set()
    for chunk in doc.noun_chunks:
        aspect = chunk.text.lower()
        if len(aspect.split()) <= 3:  # Limit to shorter, meaningful chunks
            aspects.add(aspect)
    results = {}
    for aspect in aspects:
        inputs = absa_tokenizer(text, aspect, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = absa_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}[predicted_class]
            results[aspect] = {"sentiment": sentiment}
    return results

def visualize_aspect_sentiment(aspect_results):
    if not aspect_results:
        return "<p>No aspects detected.</p>"
    html = "<table style='border-collapse: collapse; width: 70%;'><tr style='background-color: #4CAF50; color: white;'><th style='padding: 8px; border: 1px solid #ddd;'>Aspect</th><th style='padding: 8px; border: 1px solid #ddd;'>Sentiment</th></tr>"
    for aspect, scores in aspect_results.items():
        html += f"<tr style='border: 1px solid #ddd;'><td style='padding: 8px;'>{aspect}</td><td style='padding: 8px;'>{scores['sentiment']}</td></tr>"
    html += "</table>"
    return html

def create_aspect_sentiment_dataframe(aspect_results):
    if not aspect_results:
        return []
    data = [{"Aspect": aspect, "Sentiment": scores["sentiment"]} for aspect, scores in aspect_results.items()]
    return data

def detect_profanity(text: str):
    words = set(text.lower().split())
    profane = words.intersection(PROFANE_WORDS)
    return list(profane) if profane else None

def visualize_profanity(profane_words):
    if not profane_words:
        return "<p>No profanity detected.</p>"
    html = f"<table style='border-collapse: collapse; width: 30%;'><tr style='background-color: #4CAF50; color: white;'><th style='padding: 8px; border: 1px solid #ddd;'>Profane Word</th></tr>"
    for word in profane_words:
        html += f"<tr style='border: 1px solid #ddd;'><td style='padding: 8px;'>{word}</td></tr>"
    html += "</table>"
    return html

def create_profanity_dataframe(profane_words):
    if not profane_words:
        return []
    df_data = {"Profane Word": profane_words}
    df = pd.DataFrame(df_data)
    return df.to_dict('records')

def visualize_entities(text: str):
    doc = nlp(text)
    html = displacy.render(doc, style="ent", jupyter=False, options={"compact": True})
    return html

def create_entity_dataframe(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
    df = pd.DataFrame(entities, columns=['text', 'type', 'lemma'])
    return df.to_dict('records')

def process_text(text: str):
    return {
        "NER": perform_ner(text),
        "Sentiment Analysis": perform_sentiment_analysis(text),
        "Aspect Based Sentiment Analysis": perform_aspect_based_sentiment(text),
        "Profanity Detection": detect_profanity(text),
        "Entity Visualization": visualize_entities(text),
        "Entity DataFrame": create_entity_dataframe(text),
        "Sentiment Visualization": visualize_sentiment(perform_sentiment_analysis(text)),
        "Sentiment DataFrame": create_sentiment_dataframe(perform_sentiment_analysis(text)),
        "Aspect Visualization": visualize_aspect_sentiment(perform_aspect_based_sentiment(text)),
        "Aspect DataFrame": create_aspect_sentiment_dataframe(perform_aspect_based_sentiment(text)),
        "Profanity Visualization": visualize_profanity(detect_profanity(text)),
        "Profanity DataFrame": create_profanity_dataframe(detect_profanity(text))
    }
