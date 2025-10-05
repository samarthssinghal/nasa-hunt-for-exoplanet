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
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from data_preprocessing_enhanced import (
    standardize_koi_data, standardize_toi_data, standardize_k2_data,
    engineer_features, handle_missing_values
)

# Load environment variables
load_dotenv()

# Note: Page configuration is handled by app.py in multi-page mode
# This configuration is only used when running this file standalone

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
        
        # Initialize session state
        self.init_session_state()
        
        # Enhanced search configurations based on NASA Space Apps Challenge resources
        self.search_configs = {
            "All Relevant Exoplanet Data": {
                "keywords": ["exoplanet dataset csv", "exoplanet catalog download", "planetary systems data",
                            "NASA exoplanet archive", "confirmed planets list", "exoplanet database",
                            "transit photometry data", "radial velocity measurements", "exoplanet parameters",
                            "KOI TOI catalog", "habitable zone planets", "JWST TESS Kepler data"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov", "stsci.edu", "exoplanet.eu"],
                "filetypes": ["csv", "txt", "fits", "json"],
                "description": "Comprehensive search for all exoplanet datasets across missions and catalogs"
            },
            "Kepler Objects of Interest (KOI)": {
                "keywords": ["Kepler Objects of Interest", "KOI", "Kepler exoplanet", "kepler disposition", 
                            "Disposition Using Kepler Data", "kepler transits", "kepler planetary candidates",
                            "kepler false positives", "kepler confirmed planets"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov", "mast.stsci.edu"],
                "filetypes": ["csv", "txt", "fits"],
                "description": "Comprehensive list of confirmed exoplanets, planetary candidates, and false positives from Kepler mission. See column 'Disposition Using Kepler Data' for classification."
            },
            "TESS Objects of Interest (TOI)": {
                "keywords": ["TESS Objects of Interest", "TOI", "TESS exoplanet", "TFOPWG Disposition",
                            "TESS planetary candidates", "TESS false positives", "TESS ambiguous planetary candidates",
                            "TESS known planets", "TESS transits", "TESS PC", "TESS FP", "TESS APC", "TESS KP"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "tess.mit.edu", "mast.stsci.edu"],
                "filetypes": ["csv", "txt", "fits"],
                "description": "All confirmed exoplanets, planetary candidates (PC), false positives (FP), ambiguous planetary candidates (APC), and known planets (KP) from TESS. See column 'TFOPWG Disposition' for classification."
            },
            "K2 Planets and Candidates": {
                "keywords": ["K2 exoplanet candidates", "K2 planets", "Archive Disposition", "K2 mission",
                            "K2 transits", "K2 confirmed exoplanets", "K2 false positives", "K2 planetary candidates"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "nasa.gov", "keplerscience.arc.nasa.gov"],
                "filetypes": ["csv", "txt", "fits"],
                "description": "Comprehensive list from K2 mission with Archive Disposition classifications for confirmed exoplanets, candidates, and false positives."
            },
            "JWST Exoplanet Data": {
                "keywords": ["JWST exoplanet", "James Webb exoplanet", "JWST atmospheric", "JWST transit",
                            "JWST spectroscopy", "JWST transmission spectra", "JWST emission spectra",
                            "Webb exoplanet observations", "JWST exoplanet atmospheres", "JWST biosignatures"],
                "sites": ["mast.stsci.edu", "jwst.nasa.gov", "exoplanetarchive.ipac.caltech.edu", "stsci.edu"],
                "filetypes": ["csv", "fits", "txt", "json"],
                "description": "Atmospheric and transit observations from James Webb Space Telescope, including spectroscopy data."
            },
            "NEOSSat Exoplanet Targets": {
                "keywords": ["NEOSSat", "NEOSSAT exoplanet", "Canadian Space Agency", "microvariability",
                            "NEOSSat astronomy data", "space telescope asteroids", "NEOSSat photometry",
                            "CSA exoplanet", "Near Earth Object Surveillance Satellite"],
                "sites": ["donnees-data.asc-csa.gc.ca", "asc-csa.gc.ca", "open.canada.ca"],
                "filetypes": ["csv", "txt", "fits"],
                "description": "Astronomical images and exoplanet observations from Canadian NEOSSat mission."
            },
            "Machine Learning Datasets": {
                "keywords": ["exoplanet machine learning", "exoplanet detection ML", "ensemble algorithms exoplanet",
                            "supervised learning exoplanets", "exoplanet classification dataset", "transit photometry ML",
                            "exoplanet preprocessing techniques", "exoplanet feature engineering"],
                "sites": ["arxiv.org", "github.com", "kaggle.com", "exoplanetarchive.ipac.caltech.edu"],
                "filetypes": ["csv", "txt", "json", "pkl", "h5"],
                "description": "Preprocessed datasets and ML-ready data for exoplanet detection and classification."
            },
            "Ground-Based Transit Surveys": {
                "keywords": ["SuperWASP exoplanet", "WASP planets", "HATNet exoplanet", "HAT-P planets",
                            "CoRoT exoplanet", "OGLE transit", "XO exoplanet", "TrES exoplanet", 
                            "KELT exoplanet", "ground-based transit survey", "photometric survey exoplanets"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "superwasp.org", "hatsurveys.org"],
                "filetypes": ["csv", "txt", "fits", "dat"],
                "description": "Data from ground-based transit surveys: SuperWASP, HATNet, CoRoT, OGLE, XO, TrES, KELT."
            },
            "TCE & Pipeline Data": {
                "keywords": ["TCE threshold crossing event", "Kepler pipeline", "TESS pipeline",
                            "Data Validation Report", "DVR exoplanet", "vetting metrics", "transit SNR",
                            "koi_score", "planet confidence", "false positive probability"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "mast.stsci.edu", "archive.stsci.edu"],
                "filetypes": ["csv", "xml", "pdf", "fits"],
                "description": "Pipeline outputs including TCEs, vetting metrics, and confidence scores."
            },
            "Alternative Planet Catalogs": {
                "keywords": ["EPIC ID K2", "TIC TESS Input Catalog", "Kepler numbers confirmed",
                            "validated planets", "planet host star", "stellar parameters exoplanet",
                            "habitable zone planets", "Earth-like exoplanets", "rocky planets catalog"],
                "sites": ["exoplanetarchive.ipac.caltech.edu", "exoplanet.eu", "exoplanets.org"],
                "filetypes": ["csv", "txt", "json", "xml"],
                "description": "Alternative naming systems: EPIC IDs (K2), TIC numbers (TESS), confirmed planet catalogs."
            }
        }
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'max_results': 50,
                'preferred_sites': ['exoplanetarchive.ipac.caltech.edu', 'nasa.gov'],
                'file_formats': ['csv', 'txt'],
                'save_results': True,
                'auto_download': False,
                'preprocessing_defaults': {
                    'standardize_cols': True,
                    'handle_missing': True,
                    'engineer_feats': False,
                    'normalize_data': False,
                    'remove_outliers': False,
                    'balance_classes': False
                }
            }
    
    def search_serp(self, query, max_results=10):
        """Search using SERP API with enhanced error handling and AI Overview"""
        if not self.api_key:
            st.error("SERPAPI_KEY not found in environment variables!")
            return None
        
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 100),  # Google search limit
                "hl": "en",  # Enable AI Overview for English
                "gl": "us"   # Country for AI Overview support
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                st.error(f"SERP API Error: {results['error']}")
                return None
            
            # Store AI Overview if available
            self.last_ai_overview = results.get("ai_overview", None)
            
            return results.get("organic_results", [])
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return None
    
    def extract_dataset_urls(self, search_results):
        """Extract potential dataset URLs from search results with better filtering"""
        dataset_candidates = []
        
        # Exoplanet-specific keywords to verify relevance
        exoplanet_keywords = ['exoplanet', 'koi', 'toi', 'kepler', 'tess', 'planet', 
                             'transit', 'radial velocity', 'habitable', 'stellar', 
                             'wasp', 'hat', 'corot', 'k2', 'jwst', 'disposition',
                             'candidate', 'confirmed', 'false positive']
        
        # Known good domains
        trusted_domains = ['exoplanetarchive.ipac.caltech.edu', 'nasa.gov', 'stsci.edu',
                          'tess.mit.edu', 'asc-csa.gc.ca', 'cerit-sc.cz', 'swarthmore.edu',
                          'github.com/nasa', 'kaggle.com']
        
        for result in search_results:
            link = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            domain = urlparse(link).netloc
            
            # Check for file extensions
            has_data_extension = any(ext in link.lower() for ext in ['.csv', '.txt', '.json', '.fits', '.dat'])
            
            # Check for exoplanet relevance
            content = (title + ' ' + snippet).lower()
            is_exoplanet_related = any(keyword in content for keyword in exoplanet_keywords)
            
            # Check if it's from a trusted domain
            is_trusted = any(trusted in domain for trusted in trusted_domains)
            
            # More strict filtering
            if has_data_extension and (is_exoplanet_related or is_trusted):
                # Verify it's not obviously unrelated
                bad_indicators = ['example', 'test', 'demo', 'sample', 'art history', 
                                'stopwords', 'embedding', 'validation.csv']
                if not any(bad in content for bad in bad_indicators):
                    dataset_candidates.append({
                        'url': link,
                        'title': title,
                        'snippet': snippet,
                        'domain': domain,
                        'trusted': is_trusted,
                        'relevance_score': sum(1 for kw in exoplanet_keywords if kw in content)
                    })
        
        # Sort by relevance score and trusted status
        dataset_candidates.sort(key=lambda x: (x['trusted'], x['relevance_score']), reverse=True)
        
        return dataset_candidates
    
    def download_file(self, url, filename):
        """Download file from URL with progress tracking"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ExoplanetSearch/1.0)'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(self.data_dir, filename)
            
            # Create progress bar
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = min(downloaded / total_size, 1.0)
                            progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
            return filepath
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None
    
    def analyze_csv(self, filepath):
        """Quick analysis of CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            analysis = {
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_cols': len(df.select_dtypes(include=['number']).columns),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'column_names': df.columns.tolist()[:10]  # First 10 columns
            }
            
            return analysis, df.head()
            
        except Exception as e:
            return {'error': str(e)}, None

def dataset_explorer_page():
    """Dataset Explorer page for multi-page app"""
    tester = StreamlitSerpTester()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Exoplanet Data Hunter</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("**Discover and download the latest exoplanet datasets using advanced search**")
    
    # Sidebar - Simplified for narrow screens
    st.sidebar.markdown("## 🎛️ Quick Search")
    
    # Mission type selection (simplified) - Default to "All Relevant Exoplanet Data"
    mission_type = st.sidebar.selectbox(
        "🚀 Select Mission:",
        list(tester.search_configs.keys()),
        index=0  # Default to first option (All Relevant Exoplanet Data)
    )
    
    # Advanced options in expander
    with st.sidebar.expander("⚙️ Options"):
        max_results = st.slider("Max results:", 5, 100, 20)
        custom_keywords = st.text_input("Extra keywords:", "")
        
        # Time frame filter
        st.markdown("**🗓️ Time Frame**")
        time_filter = st.selectbox(
            "Search period:",
            ["All time", "2025", "2024", "2023", "2022", "Last 2 years", "Last 5 years", "Custom range"]
        )
        
        if time_filter == "Custom range":
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.number_input("From:", 2000, 2025, 2020)
            with col2:
                end_year = st.number_input("To:", 2000, 2025, 2025)
        
        auto_download = st.checkbox("Auto-download CSVs", False)
    
    # Main content area
    st.markdown("---")
    
    # Compact action buttons
    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("🔍 Search SERP API", type="primary", use_container_width=True)
    with col2:
        direct_fetch_button = st.button("📥 Download NASA Data", use_container_width=True)
    
    # Handle search button
    if search_button:
        config = tester.search_configs[mission_type]
        
        # Build search query
        base_keywords = config['keywords'].copy()  # Make a copy to avoid modifying original
        if custom_keywords:
            base_keywords.append(custom_keywords)
        
        # Add time filter to query
        time_suffix = ""
        if time_filter == "2025":
            time_suffix = " after:2025-01-01"
        elif time_filter == "2024":
            time_suffix = " after:2024-01-01 before:2025-01-01"
        elif time_filter == "2023":
            time_suffix = " after:2023-01-01 before:2024-01-01"
        elif time_filter == "2022":
            time_suffix = " after:2022-01-01 before:2023-01-01"
        elif time_filter == "Last 2 years":
            time_suffix = f" after:{datetime.now().year - 2}-01-01"
        elif time_filter == "Last 5 years":
            time_suffix = f" after:{datetime.now().year - 5}-01-01"
        elif time_filter == "Custom range":
            time_suffix = f" after:{start_year}-01-01 before:{end_year + 1}-01-01"
        
        search_query = ' OR '.join(base_keywords) + ' filetype:csv' + time_suffix
        
        with st.spinner(f"Searching for {mission_type} datasets..."):
            results = tester.search_serp(search_query, max_results)
        
        if results:
            st.success(f"Found {len(results)} search results!")
            
            # Extract dataset candidates
            candidates = tester.extract_dataset_urls(results)
            
            # Show AI Overview if available
            if hasattr(tester, 'last_ai_overview') and tester.last_ai_overview:
                with st.expander("🤖 Google AI Overview"):
                    ai_overview = tester.last_ai_overview
                    if 'text_blocks' in ai_overview:
                        for block in ai_overview['text_blocks']:
                            if block['type'] == 'paragraph':
                                st.markdown(block.get('snippet', ''))
                            elif block['type'] == 'heading':
                                st.markdown(f"**{block.get('snippet', '')}**")
                            elif block['type'] == 'list' and 'list' in block:
                                for item in block['list']:
                                    st.markdown(f"- **{item.get('title', '')}** {item.get('snippet', '')}")
            
            if candidates:
                st.markdown(f"### 📊 Dataset Candidates ({len(candidates)})")
                
                # Separate trusted from untrusted sources
                trusted_candidates = [c for c in candidates if c.get('trusted', False)]
                other_candidates = [c for c in candidates if not c.get('trusted', False)]
                
                if trusted_candidates:
                    st.markdown("#### ✅ Trusted Sources")
                    for i, candidate in enumerate(trusted_candidates):
                        relevance_emoji = "🎯" if candidate.get('relevance_score', 0) > 2 else "🌟"
                        with st.expander(f"{relevance_emoji} {candidate['title'][:60]}..."):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Domain:** {candidate['domain']}")
                                st.write(f"**Description:** {candidate['snippet']}")
                                st.write(f"**Relevance Score:** {candidate.get('relevance_score', 0)} keywords matched")
                                st.markdown(f"**Direct Link:** `{candidate['url'][:80]}...`")
                            
                            with col2:
                                # Download button for each candidate
                                if st.button(f"📥 Download", key=f"download_trusted_{i}"):
                                    filename = f"dataset_{candidate['domain'].replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                    filepath = tester.download_file(candidate['url'], filename)
                                    
                                    if filepath:
                                        st.success(f"✅ Downloaded: {filename}")
                                        
                                        # Quick analysis
                                        analysis, preview = tester.analyze_csv(filepath)
                                        if 'error' not in analysis:
                                            st.write(f"📊 {analysis['rows']} rows, {analysis['columns']} cols, {analysis['file_size_mb']:.1f} MB")
                                            if preview is not None:
                                                st.dataframe(preview)
                
                if other_candidates:
                    st.markdown("#### 🔍 Other Potential Sources")
                    for i, candidate in enumerate(other_candidates):
                        with st.expander(f"❓ {candidate['title'][:60]}..."):
                            st.warning("⚠️ Unverified source - verify data relevance before use")
                            st.write(f"**Domain:** {candidate['domain']}")
                            st.write(f"**Description:** {candidate['snippet']}")
                            st.write(f"**Relevance Score:** {candidate.get('relevance_score', 0)} keywords matched")
                            
                            # Download button with warning
                            if st.button(f"📥 Download (Verify First)", key=f"download_other_{i}"):
                                filename = f"unverified_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                filepath = tester.download_file(candidate['url'], filename)
                                
                                if filepath:
                                    st.success(f"Downloaded: {filename}")
                                    st.info("⚠️ Please verify this is exoplanet data before use")
            
            # Save search to history
            history_file = os.path.join(tester.data_dir, "search_history.json")
            
            # Load existing history
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
            
            # Add current search
            search_record = {
                'timestamp': datetime.now().isoformat(),
                'mission_type': mission_type,
                'query': search_query,
                'num_results': len(results),
                'candidates_found': len(candidates) if candidates else 0,
                'custom_keywords': custom_keywords
            }
            
            history.append(search_record)
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history[-20:], f, indent=2)  # Keep last 20 searches
        
        else:
            st.warning("No results found. Try adjusting your search parameters.")
    
    # Handle direct fetch button
    if direct_fetch_button:
        with st.spinner("Downloading NASA datasets..."):
            try:
                from direct_dataset_fetcher import DirectDatasetFetcher
                fetcher = DirectDatasetFetcher()
                results = fetcher.fetch_all_datasets()
                
                st.success(f"✅ Downloaded {results['total_datasets']} datasets!")
                
                # Show download results
                if results['successful_downloads']:
                    st.markdown("### 📥 Downloaded:")
                    for download in results['successful_downloads']:
                        st.markdown(f"- **{download['dataset_id'].replace('_', ' ')}**")
                        st.caption(f"  📊 {download['rows']:,} rows, {download['columns']} columns")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                st.info("💡 Try: `pip install requests pandas` and ensure you have internet connection")
    
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
    
    # Data Management Section (Simplified for Narrow Screens)
    st.markdown("---")
    st.markdown("### 📊 Data Files")
    
    # List available CSV files
    data_files = [f for f in os.listdir(tester.data_dir) if f.endswith('.csv')]
    
    if data_files:
        st.markdown(f"**{len(data_files)} datasets available**")
        
        # Compact view for narrow screens
        selected_file = st.selectbox("📁 Select dataset:", data_files)
        
        if selected_file:
            file_path = os.path.join(tester.data_dir, selected_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV", 
                    data=open(file_path, 'rb').read(),
                    file_name=selected_file,
                    mime='text/csv',
                    use_container_width=True
                )
            with col2:
                if st.button("👁️ Preview", use_container_width=True):
                    try:
                        df = pd.read_csv(file_path)
                        st.write(f"📊 {len(df)} rows × {len(df.columns)} columns ({file_size:.1f} MB)")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")
            
            # Optional: Expandable analysis tools
            if st.expander("🔧 Advanced Tools"):
                st.markdown("**Quick Analysis**")
                if st.button("📊 Analyze Selected", use_container_width=True):
                    try:
                        df = pd.read_csv(file_path)
                        st.write(f"**Shape:** {len(df)} rows × {len(df.columns)} columns")
                        
                        # Show missing data
                        missing = df.isnull().sum()
                        if missing.any():
                            st.write(f"**Missing data:** {missing.sum()} total")
                        
                        # Show numeric summary
                        numeric_df = df.select_dtypes(include=['number'])
                        if not numeric_df.empty:
                            st.write(f"**Numeric columns:** {len(numeric_df.columns)}")
                            st.dataframe(numeric_df.describe())
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    else:
        st.info("📁 No CSV files found. Use the 'Download NASA Data' button above to fetch datasets.")
    
    # Quick Update Check Section
    st.markdown("---")
    st.markdown("### 🔄 Dataset Update Monitor")
    
    # Show current search keywords for transparency
    with st.expander("🔍 View Search Keywords Used by SERP API"):
        config = tester.search_configs[mission_type]
        st.markdown(f"**Current Mission: {mission_type}**")
        st.markdown("**Keywords:**")
        for keyword in config['keywords']:
            st.markdown(f"- `{keyword}`")
        st.markdown("**Target Sites:**")
        for site in config['sites']:
            st.markdown(f"- `{site}`")
        st.markdown("**File Types:**")
        for filetype in config['filetypes']:
            st.markdown(f"- `{filetype}`")
    
    # Always-monitored datasets based on NASA Space Apps Challenge
    st.markdown("#### 📊 Core NASA Datasets (Auto-Monitored)")
    
    core_datasets = {
        "KOI (Kepler Objects of Interest)": {
            "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv",
            "host_site": "NASA Exoplanet Archive (Caltech/IPAC)",
            "host_url": "https://exoplanetarchive.ipac.caltech.edu/",
            "description": "Kepler mission data with 'Disposition Using Kepler Data' classification column",
            "key_column": "koi_disposition",
            "last_updated": "Continuously updated"
        },
        "TOI (TESS Objects of Interest)": {
            "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&format=csv", 
            "host_site": "NASA Exoplanet Archive (Caltech/IPAC)",
            "host_url": "https://exoplanetarchive.ipac.caltech.edu/",
            "description": "TESS data with 'TFOPWG Disposition' for PC, FP, APC, KP classifications",
            "key_column": "tfopwg_disp",
            "last_updated": "Updated monthly with new TESS sectors"
        },
        "K2 Planets and Candidates": {
            "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2pandc&format=csv",
            "host_site": "NASA Exoplanet Archive (Caltech/IPAC)",
            "host_url": "https://exoplanetarchive.ipac.caltech.edu/",
            "description": "K2 mission data with 'Archive Disposition' classification",
            "key_column": "k2c_disp",
            "last_updated": "Mission complete, periodic updates"
        },
        "Confirmed Exoplanets": {
            "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=pscomppars&format=csv",
            "host_site": "NASA Exoplanet Archive (Caltech/IPAC)",
            "host_url": "https://exoplanetarchive.ipac.caltech.edu/",
            "description": "NASA confirmed exoplanet catalog with planetary parameters",
            "key_column": "pl_name",
            "last_updated": "Updated weekly"
        },
        "SuperWASP Variable Stars": {
            "url": "https://wasp.cerit-sc.cz/search",
            "host_site": "SuperWASP Public Archive",
            "host_url": "https://wasp.cerit-sc.cz/",
            "description": "Wide Angle Search for Planets - 18M+ light curves",
            "key_column": "swasp_id",
            "last_updated": "Archive complete"
        },
        "NEOSSat Exoplanet Data": {
            "url": "https://donnees-data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/NEOSSat/",
            "host_site": "Canadian Space Agency Open Data Portal",
            "host_url": "https://www.asc-csa.gc.ca/eng/open-data/",
            "description": "Canadian space telescope exoplanet observations",
            "key_column": "target_name",
            "last_updated": "Quarterly updates"
        }
    }
    
    # Additional Space Apps Challenge Resources
    with st.expander("📚 NASA Space Apps Challenge Resources"):
        st.markdown("**Research Papers & ML Resources:**")
        st.markdown("- [Exoplanet Detection Using Machine Learning](https://arxiv.org/) - Overview of ML methods")
        st.markdown("- [Ensemble-Based ML Algorithms for Exoplanet ID](https://arxiv.org/) - High accuracy techniques")
        st.markdown("")
        st.markdown("**Space Agency Partner Data:**")
        st.markdown("- **NEOSSat (CSA)** - World's first space telescope for asteroids & exoplanets")
        st.markdown("- **JWST** - Atmospheric spectroscopy and biosignature detection")
        st.markdown("")
        st.info("💡 The SERP API searches for these resources plus related datasets using enhanced keywords based on NASA Space Apps Challenge guidelines.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for name, info in core_datasets.items():
            st.markdown(f"**{name}**")
            st.caption(f"{info['description']}")
            st.markdown(f"🏛️ **Host:** [{info['host_site']}]({info['host_url']})")
            st.markdown(f"📍 **Direct Data:** `{info['url'][:50]}...`")
            st.markdown(f"🔄 **Updates:** {info['last_updated']}")
            st.markdown("")
    
    with col2:
        st.markdown("**Quick Actions**")
        if st.button("🔄 Check All Updates", type="primary", use_container_width=True):
            with st.spinner("Checking for dataset updates..."):
                try:
                    from direct_dataset_fetcher import DirectDatasetFetcher
                    fetcher = DirectDatasetFetcher()
                    results = fetcher.fetch_all_datasets()
                    st.success(f"✅ Updated! Found {results['total_datasets']} datasets")
                    st.rerun()
                except Exception as e:
                    st.error(f"Update check failed: {str(e)}")
        
        if st.button("📋 Show All Keywords", use_container_width=True):
            st.markdown("**All Mission Keywords:**")
            for mission, config in tester.search_configs.items():
                st.markdown(f"**{mission}:**")
                for keyword in config['keywords'][:2]:  # Show first 2
                    st.caption(f"• {keyword}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 About")
    st.info("This interface combines SERP API discovery with direct NASA dataset fetching for exoplanet research.")


# For standalone testing
if __name__ == "__main__":
    # Run page configuration only when running standalone
    st.set_page_config(
        page_title="🔍 Exoplanet Data Hunter",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    dataset_explorer_page()