#!/usr/bin/env python3
"""
Main Navigation App for NASA Exoplanet Explorer
Multi-page Streamlit application with dataset explorer and educational search
"""

import streamlit as st
from educational_search import educational_search_page

# Import the dataset explorer page
from streamlit_serp_test import dataset_explorer_page

# Page configuration
st.set_page_config(
    page_title="🌌 NASA Exoplanet Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for navigation
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .nav-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .page-description {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main navigation controller"""
    
    # Sidebar navigation
    st.sidebar.markdown('<p class="nav-header">🌌 Navigation</p>', unsafe_allow_html=True)
    
    # Define pages
    PAGES = {
        "🏠 Home": {
            "function": home_page,
            "description": "Welcome and overview"
        },
        "📊 Dataset Explorer": {
            "function": dataset_explorer_page,
            "description": "Search and download exoplanet datasets"
        },
        "🧠 Educational Search": {
            "function": educational_search_page,
            "description": "Academic papers, images, and educational content"
        }
    }
    
    # Page selection
    selection = st.sidebar.radio(
        "Select Page",
        list(PAGES.keys()),
        format_func=lambda x: x,
        help="Navigate between different sections of the application"
    )
    
    # Show page description
    if selection in PAGES:
        st.sidebar.markdown(f'<p class="page-description">{PAGES[selection]["description"]}</p>', 
                          unsafe_allow_html=True)
    
    # Add separator
    st.sidebar.markdown("---")
    
    # Quick links section
    st.sidebar.markdown("### 🔗 Quick Links")
    st.sidebar.markdown("- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)")
    st.sidebar.markdown("- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)")
    st.sidebar.markdown("- [TESS Mission](https://tess.mit.edu/)")
    st.sidebar.markdown("- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)")
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: Use the Dataset Explorer to download exoplanet data, "
        "and Educational Search to learn about the science behind the discoveries!"
    )
    
    # Run the selected page
    page = PAGES[selection]["function"]
    page()

def home_page():
    """Home page with welcome message and overview"""
    st.markdown('<h1 class="main-header">🌌 NASA Exoplanet Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the NASA Exoplanet Explorer Platform! 🚀
    
    This comprehensive platform combines powerful dataset discovery with educational resources
    for exoplanet research and learning.
    """)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>📊 Dataset Explorer</h3>
        <p>Discover and download the latest exoplanet datasets from NASA and partner agencies:</p>
        <ul>
        <li>KOI (Kepler Objects of Interest)</li>
        <li>TOI (TESS Objects of Interest)</li>
        <li>K2 Planets and Candidates</li>
        <li>JWST Observations</li>
        <li>NEOSSat Data</li>
        </ul>
        <p><b>Features:</b> SERP API search, direct NASA downloads, time-based filtering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>🧠 Educational Search</h3>
        <p>Access academic papers, visual content, and educational resources:</p>
        <ul>
        <li>Google Scholar integration</li>
        <li>Visual learning with images</li>
        <li>AI-generated summaries</li>
        <li>Research paper discovery</li>
        <li>Export capabilities</li>
        </ul>
        <p><b>Perfect for:</b> Students, researchers, and astronomy enthusiasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Current Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confirmed Exoplanets", "5,600+", "↑ 12 this month")
    
    with col2:
        st.metric("Candidate Planets", "9,900+", "↑ 47 this month")
    
    with col3:
        st.metric("Planetary Systems", "4,100+", "↑ 8 this month")
    
    with col4:
        st.metric("Active Missions", "4", "TESS, JWST, Hubble, Spitzer")
    
    # Getting started guide
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    with st.expander("How to use the Dataset Explorer"):
        st.markdown("""
        1. **Navigate** to the Dataset Explorer using the sidebar
        2. **Select** a mission type or search for all exoplanet data
        3. **Apply filters** like time range or custom keywords
        4. **Search** using SERP API or download directly from NASA
        5. **Preview** datasets and download them for analysis
        6. **Export** search results for later reference
        """)
    
    with st.expander("How to use Educational Search"):
        st.markdown("""
        1. **Navigate** to Educational Search using the sidebar
        2. **Enter** a topic or question about exoplanets
        3. **Choose** search type: Scholar, Web, or Images
        4. **Review** AI-generated summaries (when available)
        5. **Explore** academic papers, visual content, or articles
        6. **Export** results as JSON for further research
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit, powered by SERP API and NASA Exoplanet Archive")

# Add custom CSS for feature cards
st.markdown("""
<style>
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    }
    .feature-card h3 {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .feature-card ul {
        margin-left: 1.5rem;
    }
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()