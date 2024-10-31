# utils/config.py
import json
import os
import sys
import streamlit as st

def load_config():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(script_dir)
    config_path = os.path.join(script_dir, 'input_config.json')
    print(config_path)
    
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        st.error(f"Config file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in config file at {config_path}")
        sys.exit(1)