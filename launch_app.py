#!/usr/bin/env python3
"""
Launch script for the Exoplanet Data Hunter Streamlit app
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        from serpapi import GoogleSearch
        from dotenv import load_dotenv
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_env():
    """Check if .env file exists and has API key"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("1. Copy .env.template to .env")
        print("2. Add your Serp API key from serpapi.com")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('SERPAPI_KEY')
    if not api_key or api_key == 'your_api_key_here':
        print("❌ Serp API key not configured")
        print("Edit .env file and add your API key")
        return False
    
    print("✅ Environment configured correctly")
    return True

def main():
    print("🚀 Launching Exoplanet Data Hunter...")
    
    if not check_requirements():
        return
    
    if not check_env():
        return
    
    print("🌟 Starting Streamlit app...")
    print("📱 App will open in your browser at http://localhost:8501")
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_serp_test.py", 
        "--server.port", "8501",
        "--server.headless", "false"
    ])

if __name__ == "__main__":
    main()