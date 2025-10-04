#!/usr/bin/env python3
"""
Streamlit Test Page for Serp API Functionality
Interactive interface for testing and searching latest exoplanet datasets
"""

import streamlit as st
import pandas as pd
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
import requests
from urllib.parse import urlparse
# import plotly.express as px
# import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🔍 Exoplanet Data Hunter",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSerpTester:
    def __init__(self):
        self.api_key = os.getenv('SERPAPI_KEY')
        self.data_dir = os.getenv('DATA_DIR', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Predefined search configurations for different mission types
        self.search_configs = {
            "Kepler Objects of Interest (KOI)": {
                "keywords": ["Kepler Objects of Interest", "KOI", "Kepler exoplanet", "kepler disposition"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov"],
                "filetypes": ["csv", "txt"],
                "description": "Comprehensive list of confirmed exoplanets, planetary candidates, and false positives from Kepler mission"
            },
            "TESS Objects of Interest (TOI)": {
                "keywords": ["TESS Objects of Interest", "TOI", "TESS exoplanet", "TFOPWG Disposition"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "tess.mit.edu"],
                "filetypes": ["csv", "txt"],
                "description": "Confirmed exoplanets, planetary candidates, and false positives from TESS mission"
            },
            "K2 Planets and Candidates": {
                "keywords": ["K2 exoplanet candidates", "K2 planets", "Archive Disposition", "K2 mission"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov"],
                "filetypes": ["csv", "txt"],
                "description": "Comprehensive list from K2 mission with Archive Disposition classifications"
            },
            "JWST Exoplanet Data": {
                "keywords": ["JWST exoplanet", "James Webb exoplanet", "JWST atmospheric", "JWST transit"],
                "sites": ["mast.stsci.edu", "jwst.nasa.gov", "exoplanetarchive.ipac.caltech.edu"],
                "filetypes": ["csv", "fits", "txt"],
                "description": "Latest exoplanet observations and atmospheric data from James Webb Space Telescope"
            },
            "NEOSSat Astronomy Data": {
                "keywords": ["NEOSSat exoplanet", "NEOSSat astronomy", "Canadian Space Agency exoplanet"],
                "sites": ["donnees-data.asc-csa.gc.ca", "asc-csa.gc.ca"],
                "filetypes": ["csv", "txt"],
                "description": "Astronomical data from Canada's Near-Earth Object Surveillance Satellite"
            },
            "Latest Confirmed Exoplanets": {
                "keywords": ["latest confirmed exoplanets", "new exoplanet discoveries", "exoplanet confirmation"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov"],
                "filetypes": ["csv", "txt"],
                "description": "Most recent confirmed exoplanet discoveries and updates"
            }
        }
    
    def build_search_query(self, mission_type, custom_keywords="", year="2024-2025"):
        """Build optimized search query for specific mission type"""
        config = self.search_configs.get(mission_type, {})
        
        # Start with mission-specific keywords
        keywords = config.get("keywords", [])
        if custom_keywords:
            keywords.extend([kw.strip() for kw in custom_keywords.split(",")])
        
        # Build query components
        keyword_part = f'"{keywords[0]}"' if keywords else ""
        if len(keywords) > 1:
            keyword_part += f" OR {' OR '.join([f'"{kw}"' for kw in keywords[1:3]])}"
        
        # Add temporal constraint
        temporal_part = f"{year}"
        
        # Add site restrictions
        sites = config.get("sites", [])
        site_part = f"site:{' OR site:'.join(sites)}" if sites else ""
        
        # Add filetype preference
        filetypes = config.get("filetypes", ["csv"])
        filetype_part = f"filetype:{filetypes[0]}" if filetypes else ""
        
        # Combine query parts
        query_parts = [p for p in [keyword_part, temporal_part, site_part, filetype_part] if p]
        return " ".join(query_parts)
    
    def search_serp(self, query, num_results=10):
        """Execute Serp API search"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if 'error' in results:
                st.error(f"API Error: {results['error']}")
                return []
            
            return results.get('organic_results', [])
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def analyze_search_results(self, results):
        """Analyze and categorize search results"""
        categorized = {
            "data_files": [],
            "research_papers": [],
            "official_sites": [],
            "other": []
        }
        
        for result in results:
            title = result.get('title', '').lower()
            link = result.get('link', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Categorize based on content
            if any(ext in link for ext in ['.csv', '.txt', '.fits']) or 'download' in link:
                categorized["data_files"].append(result)
            elif any(term in title + snippet for term in ['paper', 'article', 'research', 'arxiv', 'doi']):
                categorized["research_papers"].append(result)
            elif any(site in link for site in ['nasa.gov', 'exoplanetarchive', 'asc-csa.gc.ca', 'jwst.nasa.gov']):
                categorized["official_sites"].append(result)
            else:
                categorized["other"].append(result)
        
        return categorized
    
    def display_search_results(self, results, query):
        """Display search results in organized format"""
        if not results:
            st.warning("No results found. Try different keywords or check your API key.")
            return
        
        categorized = self.analyze_search_results(results)
        
        # Results summary
        st.markdown("### 📊 Search Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📁 {len(categorized['data_files'])}</h3>
                <p>Data Files</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 {len(categorized['research_papers'])}</h3>
                <p>Research Papers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏛️ {len(categorized['official_sites'])}</h3>
                <p>Official Sites</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 {len(categorized['other'])}</h3>
                <p>Other Results</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display categorized results
        for category, items in categorized.items():
            if items:
                st.markdown(f"### 📋 {category.replace('_', ' ').title()}")
                
                for i, result in enumerate(items):
                    title = result.get('title', 'No title')
                    link = result.get('link', '#')
                    snippet = result.get('snippet', 'No description')
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4><a href="{link}" target="_blank">{title}</a></h4>
                        <p><strong>URL:</strong> <code>{link}</code></p>
                        <p>{snippet}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add download button for data files
                    if category == "data_files":
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button(f"📥 Try Download", key=f"download_{i}_{category}"):
                                self.attempt_download(link, f"search_result_{i}.csv")
    
    def attempt_download(self, url, filename):
        """Attempt to download and preview a file"""
        try:
            with st.spinner("Downloading..."):
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Try to read as CSV
                try:
                    df = pd.read_csv(url)
                    if len(df) > 0:
                        st.success(f"✅ Successfully downloaded: {len(df)} rows, {len(df.columns)} columns")
                        
                        # Save to data directory
                        filepath = os.path.join(self.data_dir, filename)
                        df.to_csv(filepath, index=False)
                        
                        # Show preview
                        st.markdown("#### 👀 Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Show basic stats
                        st.markdown("#### 📈 Basic Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Rows", len(df))
                            st.metric("Total Columns", len(df.columns))
                        with col2:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            st.metric("Numeric Columns", len(numeric_cols))
                            st.metric("Missing Values", df.isnull().sum().sum())
                        
                        return df
                    else:
                        st.warning("Downloaded file is empty")
                        return None
                except Exception as csv_error:
                    st.error(f"Could not parse as CSV: {str(csv_error)}")
                    return None
                    
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Exoplanet Data Hunter</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Serp API Testing for Latest Exoplanet Datasets**")
    
    # Initialize tester
    tester = StreamlitSerpTester()
    
    # Check API key
    if not tester.api_key or tester.api_key == 'your_api_key_here':
        st.error("⚠️ Please configure your Serp API key in the .env file")
        st.info("1. Copy .env.template to .env\n2. Add your API key from serpapi.com\n3. Restart the app")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Search Configuration")
    
    # Mission type selector
    mission_type = st.sidebar.selectbox(
        "🎯 Select Mission/Dataset Type:",
        options=list(tester.search_configs.keys()),
        help="Choose the type of exoplanet data you want to search for"
    )
    
    # Display mission description
    if mission_type:
        config = tester.search_configs[mission_type]
        st.sidebar.markdown(f"**📝 Description:**")
        st.sidebar.info(config["description"])
        
        st.sidebar.markdown(f"**🔍 Default Keywords:** {', '.join(config['keywords'][:3])}")
        st.sidebar.markdown(f"**🌐 Target Sites:** {', '.join(config['sites'][:2])}")
    
    # Custom search options
    st.sidebar.markdown("---")
    custom_keywords = st.sidebar.text_input(
        "➕ Additional Keywords (comma-separated):",
        placeholder="e.g., atmospheric composition, transit photometry"
    )
    
    year_range = st.sidebar.selectbox(
        "📅 Time Range:",
        ["2024-2025", "2023-2024", "2022-2025", "latest"],
        index=0
    )
    
    num_results = st.sidebar.slider(
        "📊 Number of Results:",
        min_value=5, max_value=20, value=10
    )
    
    # Main search interface
    st.markdown("---")
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_button = st.button("🚀 Research Latest Datasets", type="primary")
    
    if search_button:
        # Build and display query
        query = tester.build_search_query(mission_type, custom_keywords, year_range)
        
        st.markdown("### 🔎 Search Query")
        st.code(query, language="text")
        
        # Execute search
        with st.spinner("🔍 Searching for latest datasets..."):
            results = tester.search_serp(query, num_results)
        
        if results:
            st.success(f"✅ Found {len(results)} results!")
            tester.display_search_results(results, query)
            
            # Save search history
            search_record = {
                "timestamp": datetime.now().isoformat(),
                "mission_type": mission_type,
                "query": query,
                "num_results": len(results),
                "custom_keywords": custom_keywords
            }
            
            # Load existing history
            history_file = os.path.join(tester.data_dir, "search_history.json")
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
            
            history.append(search_record)
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history[-20:], f, indent=2)  # Keep last 20 searches
        
        else:
            st.warning("No results found. Try adjusting your search parameters.")
    
    # Search history section
    st.markdown("---")
    if st.expander("📜 Recent Search History"):
        history_file = os.path.join(tester.data_dir, "search_history.json")
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if history:
                history_df = pd.DataFrame(history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                st.dataframe(history_df[['timestamp', 'mission_type', 'num_results', 'custom_keywords']])
            else:
                st.info("No search history yet.")
        except:
            st.info("No search history yet.")
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📈 Quick API Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧪 Test API Connection"):
            with st.spinner("Testing..."):
                test_results = tester.search_serp("exoplanet NASA", 3)
                if test_results:
                    st.success("✅ API Working!")
                else:
                    st.error("❌ API Issues")
    
    with col2:
        data_files = [f for f in os.listdir(tester.data_dir) if f.endswith('.csv')]
        st.metric("📁 Downloaded Files", len(data_files))
    
    with col3:
        if st.button("🗂️ View Data Directory"):
            st.write("**Files in data directory:**")
            for file in os.listdir(tester.data_dir):
                st.write(f"- {file}")


if __name__ == "__main__":
    main()