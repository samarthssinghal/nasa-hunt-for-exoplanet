#!/usr/bin/env python3
"""
Quick test of Serp API functionality
"""

import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

def test_serp_api():
    api_key = os.getenv('SERPAPI_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ Please set SERPAPI_KEY in .env file")
        return
    
    print("🔍 Testing Serp API connection...")
    
    # Simple test query
    params = {
        "engine": "google",
        "q": "exoplanet data NASA",
        "api_key": api_key,
        "num": 3
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if 'error' in results:
            print(f"❌ API Error: {results['error']}")
            return
        
        organic_results = results.get('organic_results', [])
        print(f"✅ API Working! Found {len(organic_results)} results")
        
        for i, result in enumerate(organic_results[:3], 1):
            title = result.get('title', 'No title')
            link = result.get('link', 'No link')
            print(f"{i}. {title}")
            print(f"   {link}")
        
        # Check remaining quota
        search_info = results.get('search_information', {})
        print(f"\n📊 Search completed in {search_info.get('time_taken_displayed', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_serp_api()