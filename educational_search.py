#!/usr/bin/env python3
"""
Educational Search Page for Exoplanet Research
Uses SERP API to search Google Scholar, Images, and Web with AI Overview
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from serpapi import GoogleSearch
from typing import Dict, List, Optional
import requests
from urllib.parse import quote_plus

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class EducationalSearch:
    """Handles educational searches using SERP API"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPAPI_KEY')
        if not self.api_key:
            st.error("SERPAPI_KEY not found in environment variables!")
        
    def search_google_scholar(self, query: str, num_results: int = 10) -> Dict:
        """Search Google Scholar for academic papers"""
        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "hl": "en"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return results
            
        except Exception as e:
            st.error(f"Scholar search failed: {str(e)}")
            return {}
    
    def search_google_images(self, query: str, num_results: int = 10) -> Dict:
        """Search Google Images for visual content"""
        try:
            params = {
                "engine": "google_images",
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "hl": "en",
                "gl": "us"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return results
            
        except Exception as e:
            st.error(f"Image search failed: {str(e)}")
            return {}
    
    def search_google_with_ai(self, query: str, num_results: int = 10) -> Dict:
        """Search Google with AI Overview"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "hl": "en",
                "gl": "us"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check if we need to fetch AI Overview separately
            if 'ai_overview' in results and 'page_token' in results['ai_overview']:
                # Fetch dedicated AI Overview
                ai_results = self.get_ai_overview(results['ai_overview']['page_token'])
                if ai_results:
                    results['ai_overview'] = ai_results.get('ai_overview', {})
            
            return results
            
        except Exception as e:
            st.error(f"Web search failed: {str(e)}")
            return {}
    
    def get_ai_overview(self, page_token: str) -> Dict:
        """Get dedicated AI Overview results"""
        try:
            params = {
                "engine": "google_ai_overview",
                "page_token": page_token,
                "api_key": self.api_key
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return results
            
        except Exception as e:
            st.error(f"AI Overview fetch failed: {str(e)}")
            return {}

def display_ai_overview(ai_overview: Dict):
    """Display AI Overview in a formatted way"""
    if not ai_overview or 'error' in ai_overview:
        return
    
    with st.expander("🤖 AI Summary", expanded=True):
        if 'text_blocks' in ai_overview:
            for block in ai_overview['text_blocks']:
                if block['type'] == 'heading':
                    st.markdown(f"### {block.get('snippet', '')}")
                elif block['type'] == 'paragraph':
                    st.markdown(block.get('snippet', ''))
                elif block['type'] == 'list' and 'list' in block:
                    for item in block['list']:
                        if 'title' in item:
                            st.markdown(f"- **{item['title']}**: {item.get('snippet', '')}")
                        else:
                            st.markdown(f"- {item.get('snippet', '')}")
                elif block['type'] == 'comparison' and 'comparison' in block:
                    # Display comparison table
                    if 'product_labels' in block:
                        st.markdown(f"**Comparing**: {', '.join(block['product_labels'])}")
                    comparison_data = []
                    for comp in block['comparison']:
                        row = {'Feature': comp['feature']}
                        for i, value in enumerate(comp['values']):
                            row[f'Value {i+1}'] = value
                        comparison_data.append(row)
                    if comparison_data:
                        st.dataframe(comparison_data)
        
        # Show references
        if 'references' in ai_overview:
            with st.expander("📚 References"):
                for ref in ai_overview['references']:
                    st.markdown(f"- [{ref['title']}]({ref['link']}) - {ref.get('source', '')}")
                    st.caption(ref.get('snippet', ''))

def display_scholar_results(results: Dict):
    """Display Google Scholar results"""
    if 'organic_results' not in results:
        st.warning("No academic papers found.")
        return
    
    st.markdown("### 📚 Academic Papers")
    
    for i, paper in enumerate(results['organic_results']):
        with st.expander(f"📄 {paper.get('title', 'Untitled')}", expanded=(i < 3)):
            # Paper details
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Title**: {paper.get('title', 'N/A')}")
                
                # Authors
                if 'publication_info' in paper and 'authors' in paper['publication_info']:
                    authors = ", ".join([author['name'] for author in paper['publication_info']['authors']])
                    st.markdown(f"**Authors**: {authors}")
                
                # Publication info
                pub_info = paper.get('publication_info', {}).get('summary', '')
                if pub_info:
                    st.markdown(f"**Published**: {pub_info}")
                
                # Abstract/Snippet
                snippet = paper.get('snippet', '')
                if snippet:
                    st.markdown(f"**Abstract**: {snippet}")
                
                # Link
                if 'link' in paper:
                    st.markdown(f"🔗 [View Paper]({paper['link']})")
                
                # PDF link if available
                if 'resources' in paper:
                    for resource in paper['resources']:
                        if 'file_format' in resource and resource['file_format'] == 'PDF':
                            st.markdown(f"📑 [Download PDF]({resource['link']})")
            
            with col2:
                # Citations
                cited_by = paper.get('inline_links', {}).get('cited_by', {})
                if cited_by and 'total' in cited_by:
                    st.metric("Citations", cited_by['total'])
                
                # Related articles
                related = paper.get('inline_links', {}).get('related_pages_link', '')
                if related:
                    st.markdown(f"[Related Articles]({related})")

def display_image_results(results: Dict):
    """Display Google Images results"""
    if 'images_results' not in results:
        st.warning("No images found.")
        return
    
    st.markdown("### 🖼️ Visual Results")
    
    # Create columns for image grid
    cols = st.columns(4)
    
    for i, img in enumerate(results['images_results'][:20]):  # Limit to 20 images
        col_idx = i % 4
        
        with cols[col_idx]:
            # Display thumbnail
            if 'thumbnail' in img:
                st.image(img['thumbnail'], caption=img.get('title', '')[:50] + '...', use_container_width=True)
                
                # Show source
                if 'source' in img:
                    st.caption(f"Source: {img['source']}")
                
                # Link to original
                if 'link' in img:
                    st.markdown(f"[View Original]({img['link']})")
                
                st.markdown("---")

def display_web_results(results: Dict):
    """Display web search results"""
    if 'organic_results' not in results:
        st.warning("No web results found.")
        return
    
    st.markdown("### 🌐 Web Results")
    
    for result in results['organic_results'][:10]:
        with st.container():
            st.markdown(f"#### [{result.get('title', 'Untitled')}]({result.get('link', '#')})")
            
            # Show source
            if 'source' in result:
                st.caption(f"Source: {result['source']}")
            
            # Show snippet
            if 'snippet' in result:
                st.markdown(result['snippet'])
            
            # Show date if available
            if 'date' in result:
                st.caption(f"Published: {result['date']}")
            
            st.markdown("---")

def educational_search_page():
    """Main educational search page"""
    st.title("🧠 Educational Search")
    st.markdown("Search for academic papers, visual content, and educational resources about exoplanets and astronomy.")
    
    # Initialize searcher
    searcher = EducationalSearch()
    
    # Search input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter a topic or question:",
            placeholder="e.g., 'what is a hot Jupiter?' or 'TESS exoplanet discoveries 2024'",
            help="Search for educational content about exoplanets"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Google Scholar", "Google Search", "Images"],
            help="Choose the type of search"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        num_results = st.slider("Number of results", 5, 50, 10)
        
        # Add search filters based on type
        if search_type == "Google Scholar":
            col1, col2 = st.columns(2)
            with col1:
                year_start = st.number_input("Year from:", 2000, 2025, 2020)
            with col2:
                year_end = st.number_input("Year to:", 2000, 2025, 2025)
            
            # Modify query with year filter
            if st.checkbox("Apply year filter"):
                query = f"{query} after:{year_start} before:{year_end}"
        
        elif search_type == "Images":
            img_size = st.selectbox("Image size", ["any", "large", "medium", "small"])
            img_type = st.selectbox("Image type", ["any", "photo", "clipart", "line drawing"])
            
            if img_size != "any":
                query += f" imagesize:{img_size}"
            if img_type != "any":
                query += f" imagetype:{img_type}"
    
    # Search button
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner(f"Searching {search_type}..."):
            # Store search in session state
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            search_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'search_type': search_type,
                'num_results': num_results
            }
            
            # Perform search based on type
            if search_type == "Google Scholar":
                results = searcher.search_google_scholar(query, num_results)
                
                if results:
                    st.success(f"Found {len(results.get('organic_results', []))} academic papers")
                    display_scholar_results(results)
                    
                    # Export option
                    if st.button("📥 Export Scholar Results as JSON"):
                        json_str = json.dumps(results.get('organic_results', []), indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"scholar_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            elif search_type == "Images":
                results = searcher.search_google_images(query, num_results)
                
                if results:
                    st.success(f"Found {len(results.get('images_results', []))} images")
                    display_image_results(results)
                    
                    # Export option
                    if st.button("📥 Export Image URLs as JSON"):
                        image_data = [
                            {
                                'title': img.get('title', ''),
                                'link': img.get('link', ''),
                                'thumbnail': img.get('thumbnail', ''),
                                'source': img.get('source', '')
                            }
                            for img in results.get('images_results', [])
                        ]
                        json_str = json.dumps(image_data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"image_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            else:  # Google Search
                results = searcher.search_google_with_ai(query, num_results)
                
                if results:
                    # Show AI Overview if available
                    if 'ai_overview' in results and results['ai_overview']:
                        display_ai_overview(results['ai_overview'])
                    
                    # Show regular results
                    display_web_results(results)
                    
                    # Export option
                    if st.button("📥 Export Web Results as JSON"):
                        export_data = {
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'ai_overview': results.get('ai_overview', {}),
                            'results': results.get('organic_results', [])
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"web_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            # Add to search history
            search_record['results_count'] = len(results.get('organic_results', results.get('images_results', [])))
            st.session_state.search_history.append(search_record)
    
    # Search history
    with st.expander("📜 Search History"):
        if 'search_history' in st.session_state and st.session_state.search_history:
            history_df = pd.DataFrame(st.session_state.search_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            st.dataframe(history_df[['timestamp', 'query', 'search_type', 'results_count']], use_container_width=True)
        else:
            st.info("No search history yet.")
    
    # Educational resources section
    st.markdown("---")
    st.markdown("### 📚 Quick Educational Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌟 Exoplanet Basics**")
        if st.button("What are exoplanets?", use_container_width=True):
            st.session_state.preset_query = "what are exoplanets definition types"
            st.rerun()
        if st.button("Transit method explained", use_container_width=True):
            st.session_state.preset_query = "transit photometry method exoplanet detection"
            st.rerun()
    
    with col2:
        st.markdown("**🔭 Recent Discoveries**")
        if st.button("JWST exoplanet findings", use_container_width=True):
            st.session_state.preset_query = "JWST James Webb exoplanet discoveries 2024"
            st.rerun()
        if st.button("Habitable zone planets", use_container_width=True):
            st.session_state.preset_query = "habitable zone exoplanets potentially habitable"
            st.rerun()
    
    with col3:
        st.markdown("**📊 Data & Methods**")
        if st.button("Machine learning in astronomy", use_container_width=True):
            st.session_state.preset_query = "machine learning exoplanet detection classification"
            st.rerun()
        if st.button("Radial velocity method", use_container_width=True):
            st.session_state.preset_query = "radial velocity method exoplanet detection doppler"
            st.rerun()
    
    # Handle preset queries
    if 'preset_query' in st.session_state:
        st.info(f"💡 Try searching for: '{st.session_state.preset_query}'")
        del st.session_state.preset_query

if __name__ == "__main__":
    educational_search_page()