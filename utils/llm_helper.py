# utils/llm_helper.py
from openai import AzureOpenAI
import streamlit as st
import numpy as np

client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_KEY"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
)

def count_tokens(text):
    """Simple token counter - rough estimate"""
    words = text.split()
    # Rough estimate: average token is 4 characters
    return sum(len(word) // 4 + 1 for word in words)

def truncate_text_to_token_limit(text, token_limit):
    """Truncate text to stay within token limit"""
    words = text.split()
    current_text = ""
    current_tokens = 0
    
    for word in words:
        word_tokens = count_tokens(word)
        if current_tokens + word_tokens > token_limit:
            break
        current_text += word + " "
        current_tokens += word_tokens
    
    return current_text.strip()

def summarize_cluster_themes(cluster_data, token_limit=32000):
    """
    Summarize themes for a cluster using Azure OpenAI with token limit
    """
    system_prompt = """You are an expert analyst who identifies key themes and patterns in technical data. 
    Your task is to analyze the provided text data and:
    1. Identify 3-5 main themes or patterns
    2. Provide a brief explanation for each theme
    3. Present the information in a clear, bulleted format
    Focus on technical aspects and be specific about the patterns you observe."""

    # Calculate tokens for system prompt
    system_tokens = count_tokens(system_prompt)
    
    # Reserve tokens for response and formatting
    reserved_tokens = 4000
    
    # Calculate available tokens for user data
    available_tokens = token_limit - system_tokens - reserved_tokens
    
    # Truncate cluster data if needed
    truncated_data = truncate_text_to_token_limit(cluster_data, available_tokens)

    user_prompt = f"""Analyze the following technical descriptions and identify the main themes:
    {truncated_data}
    
    Provide your analysis in the following format:
    • Theme 1: [theme name]
      - [brief explanation]
    • Theme 2: [theme name]
      - [brief explanation]
    And so on..."""

    try:
        response = client.chat.completions.create(
            model=st.secrets["AZURE_OPENAI_DEPLOYMENT_CHAT"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=30000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "Could not generate themes for this cluster."

    except Exception as e:
        st.error(f"Error in theme summarization: {str(e)}")
        return f"Error generating themes: {str(e)}"

def process_cluster_themes(df_clusters, selected_column, token_limit):
    """
    Process and summarize themes for each cluster with token limit
    """
    cluster_themes = {}
    
    for cluster in df_clusters['Cluster'].unique():
        # Get cluster data
        cluster_data = df_clusters[df_clusters['Cluster'] == cluster]
        
        # Sample if more than 50 entries
        if len(cluster_data) > 50:
            cluster_data = cluster_data.sample(n=50, random_state=42)
        
        # Concatenate text data
        text_data = ' '.join(cluster_data[selected_column].astype(str))
        
        # Get themes for this cluster with token limit
        themes = summarize_cluster_themes(text_data, token_limit)
        cluster_themes[cluster] = themes
    
    return cluster_themes