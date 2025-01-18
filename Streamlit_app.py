import streamlit as st
import requests
import plotly.graph_objects as go
import os
import json

# Get the backend URL from environment variable
BACKEND_URL = st.secrets["BACKEND_URL"]

st.title("Transcript Sentiment Analysis")

def create_sentiment_chart(analysis_results):
    if 'method' not in analysis_results:
        return None
    
    if analysis_results['method'] == 'textblob':
        # Create gauge chart for TextBlob
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = analysis_results['polarity'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Polarity"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "gray"},
                    {'range': [0.3, 1], 'color': "green"}
                ]
            }
        ))
    else:
        # Create bar chart for Transformers
        fig = go.Figure(data=[
            go.Bar(
                x=['Positive', 'Negative'],
                y=[analysis_results['positive_chunks'], analysis_results['negative_chunks']],
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(title="Sentiment Distribution")
    
    return fig

def analyze_text(content, method):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/analyze",  # Correct endpoint URL
            json={"text": content, "method": method.lower()}  # Send text instead of filepath
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

uploaded_file = st.file_uploader("Upload a transcript", type=["txt"])
method = st.selectbox("Choose Sentiment Analysis Method", ["TextBlob", "Transformers"])

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')  # Read file content as text
    st.text_area("Transcript Content", content, height=150)
    
    with st.spinner("Processing..."):
        try:
            # Analyze sentiment directly without uploading the file
            analysis_results = analyze_text(content, method)
            
            if analysis_results:
                st.success("Analysis Complete!")
                
                # Display results
                st.json(analysis_results)
                
                # Display visualization
                fig = create_sentiment_chart(analysis_results)
                if fig:
                    st.plotly_chart(fig)
                
                # Additional metrics
                st.subheader("Detailed Metrics")
                if method == "TextBlob":
                    st.metric("Subjectivity", f"{analysis_results['subjectivity']:.2f}")
                else:
                    st.metric("Average Positive Score", f"{analysis_results['avg_positive_score']:.2f}")
                    st.metric("Average Negative Score", f"{analysis_results['avg_negative_score']:.2f}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
