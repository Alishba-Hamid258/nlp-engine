
import streamlit as st
import requests
import pandas as pd
from config import API_PORT

API_URL = f"http://localhost:{API_PORT}/api/analyze"

st.title("NLP Engine")

text_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text_input:
        try:
            response = requests.post(API_URL, json={"text": text_input})
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("NER (Named Entity Recognition)")
                st.write(result["NER"])
                
                st.subheader("Entity Visualization")
                st.markdown(result["Entity Visualization"], unsafe_allow_html=True)
                
                st.subheader("Entity DataFrame")
                if result["Entity DataFrame"]:
                    df = pd.DataFrame(result["Entity DataFrame"])
                    st.dataframe(df)
                else:
                    st.write("No entities found.")
                
                st.subheader("Sentiment Analysis")
                st.write(result["Sentiment Analysis"])
                
                st.subheader("Sentiment Visualization")
                st.markdown(result["Sentiment Visualization"], unsafe_allow_html=True)
                
                st.subheader("Sentiment DataFrame")
                if result["Sentiment DataFrame"]:
                    df = pd.DataFrame(result["Sentiment DataFrame"])
                    st.dataframe(df)
                else:
                    st.write("No sentiment data to display.")
                
                st.subheader("Aspect Based Sentiment Analysis")
                st.write(result["Aspect Based Sentiment Analysis"])
                
                st.subheader("Aspect Visualization")
                st.markdown(result["Aspect Visualization"], unsafe_allow_html=True)
                
                st.subheader("Aspect DataFrame")
                if result["Aspect DataFrame"]:
                    df = pd.DataFrame(result["Aspect DataFrame"])
                    st.dataframe(df)
                else:
                    st.write("No aspects detected.")
                
                st.subheader("Profanity Detection")
                st.write(result["Profanity Detection"] or "No profanity detected")
                
                st.subheader("Profanity Visualization")
                st.markdown(result["Profanity Visualization"], unsafe_allow_html=True)
                
                st.subheader("Profanity DataFrame")
                if result["Profanity DataFrame"]:
                    df = pd.DataFrame(result["Profanity DataFrame"])
                    st.dataframe(df)
                else:
                    st.write("No profanity detected.")
            else:
                st.error("Error from API: " + response.text)
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")
    else:
        st.warning("Please enter some text.")
