# Home.py
import streamlit as st
from databricks.sdk.core import Config  
from databricks.sdk import WorkspaceClient 
from utils.config import load_config
import auth

st.set_page_config(page_title="Failure Mode Identification Tool", page_icon="📊", layout="wide")

# Get the user's information
user_info = auth.get_user_info()
print(user_info)

def main():    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Failure Mode Identification Tool</p>', unsafe_allow_html=True)

    # Display the user's information in the sidebar  
    st.sidebar.title("User Info")  
    st.sidebar.write(f"User Name: {user_info.get('user_name')}")  
    st.sidebar.write(f"User ID: {user_info.get('user_id')}") 
    st.markdown("""
    ### Welcome to the Failure Mode Identification Tool
    
    This tool allows you to identify themes and understand the failure modes from vehicle claims
    
    #### Getting Started
    1. Choose "Identify Failure Modes" from the sidebar.
    2. Input the required parameters in the sidebar 
    3. Click "Analyze"
    4. Follow through instructio to perform analysis and understand faolure modes on demand 
    """)
    
    
    st.markdown("""
    #### Need Help?
    If you need assistance or have any questions, please contact the support team at example@example.com.
    """)

if __name__ == "__main__":
    main()