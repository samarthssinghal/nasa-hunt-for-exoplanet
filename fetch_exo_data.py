#!/usr/bin/env python3
"""
Serp API Integration for Latest Exoplanet Data Fetching
Dynamically pulls latest exoplanet datasets from NASA/CSA archives via structured search
"""

import os
import requests
import pandas as pd
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('SERPAPI_KEY')
SEARCH_ENGINE = os.getenv('SEARCH_ENGINE', 'google')
SEARCH_DELAY = int(os.getenv('SEARCH_DELAY', 1))
DATA_DIR = os.getenv('DATA_DIR', 'data')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class ExoplanetDataFetcher:
    """Fetches latest exoplanet data using Serp API"""
    
    def __init__(self):
        if not API_KEY or API_KEY == 'your_api_key_here':
            raise ValueError("Please set SERPAPI_KEY in .env file. Get your key from https://serpapi.com/")
        
        self.search_queries = {
            'TESS_TOI': {
                'query': '"TESS Objects of Interest" latest 2025 filetype:csv site:exoplanetarchive.ipac.caltech.edu',
                'filename': 'tess_toi_latest.csv'
            },
            'Kepler_KOI': {
                'query': '"Kepler Objects of Interest" latest 2025 filetype:csv site:nasa.gov OR site:exoplanetarchive.ipac.caltech.edu',
                'filename': 'kepler_koi_latest.csv'
            },
            'K2_Candidates': {
                'query': '"K2 exoplanet candidates" CSV archive disposition 2025',
                'filename': 'k2_candidates_latest.csv'
            },
            'JWST_Confirmations': {
                'query': '"JWST exoplanet confirmations" table CSV 2025',
                'filename': 'jwst_confirmations_latest.csv'
            },
            'NEOSSat_CSA': {
                'query': '"NEOSSat exoplanet data" CSV "Canadian Space Agency" 2025',
                'filename': 'neossat_csa_latest.csv'
            },
            'General_Updates': {
                'query': '"latest exoplanet catalog" NASA 2025 confirmed planets CSV',
                'filename': 'general_exo_updates.csv'
            }
        }
    
    def search_serp(self, query, num_results=5):
        """Search Serp API for exoplanet data links"""
        try:
            params = {
                "engine": SEARCH_ENGINE,
                "q": query,
                "api_key": API_KEY,
                "num": num_results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if 'error' in results:
                logger.error(f"Serp API error: {results['error']}")
                return []
            
            return results.get('organic_results', [])
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return []
    
    def is_valid_csv_url(self, url):
        """Check if URL points to a valid CSV file"""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                (url.endswith('.csv') or 'csv' in url.lower() or 'download' in url.lower())
            )
        except:
            return False
    
    def download_csv(self, url, filename):
        """Download CSV from URL and return as DataFrame"""
        try:
            if not self.is_valid_csv_url(url):
                logger.warning(f"Skipping non-CSV URL: {url}")
                return None
            
            logger.info(f"Attempting to download: {url}")
            
            # Set headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Try to read as CSV
            try:
                df = pd.read_csv(url)
                if len(df) > 0:
                    filepath = os.path.join(DATA_DIR, filename)
                    df.to_csv(filepath, index=False)
                    logger.info(f"Successfully downloaded {filename}: {len(df)} rows, {len(df.columns)} columns")
                    return df
                else:
                    logger.warning(f"Empty dataset from {url}")
                    return None
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty CSV from {url}")
                return None
            except Exception as csv_error:
                logger.error(f"CSV parsing error for {url}: {str(csv_error)}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {str(e)}")
            return None
    
    def fetch_latest_data(self, mission_key, max_attempts=3):
        """Fetch latest dataset for a specific mission"""
        if mission_key not in self.search_queries:
            logger.error(f"Unknown mission: {mission_key}")
            return pd.DataFrame()
        
        mission_config = self.search_queries[mission_key]
        query = mission_config['query']
        filename = mission_config['filename']
        
        logger.info(f"Searching for {mission_key} data...")
        logger.info(f"Query: {query}")
        
        results = self.search_serp(query)
        
        if not results:
            logger.warning(f"No search results for {mission_key}")
            return pd.DataFrame()
        
        # Try downloading from top results
        for i, result in enumerate(results[:max_attempts]):
            link = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            logger.info(f"Trying result {i+1}: {title}")
            logger.info(f"URL: {link}")
            
            # Check if this looks like a data table/dataset
            data_indicators = ['table', 'dataset', 'data', 'csv', 'archive', 'catalog']
            if any(indicator in snippet.lower() or indicator in title.lower() for indicator in data_indicators):
                df = self.download_csv(link, filename)
                if df is not None and len(df) > 0:
                    logger.info(f"✅ Successfully fetched {mission_key} data: {len(df)} rows")
                    return df
            else:
                logger.info(f"Skipping non-data result: {title}")
            
            time.sleep(SEARCH_DELAY)
        
        logger.warning(f"❌ No valid CSV found for {mission_key}")
        return pd.DataFrame()
    
    def fetch_all_missions(self):
        """Fetch data for all configured missions"""
        results = {}
        total_rows = 0
        
        logger.info("🚀 Starting exoplanet data fetch for all missions...")
        
        for mission_key in self.search_queries.keys():
            logger.info(f"\n--- Fetching {mission_key} ---")
            df = self.fetch_latest_data(mission_key)
            results[mission_key] = df
            
            if len(df) > 0:
                total_rows += len(df)
                logger.info(f"✅ {mission_key}: {len(df)} rows")
            else:
                logger.warning(f"❌ {mission_key}: No data")
            
            time.sleep(SEARCH_DELAY)
        
        logger.info(f"\n🎯 Fetch complete! Total rows across all missions: {total_rows}")
        return results
    
    def save_fetch_summary(self, results):
        """Save summary of fetch operation"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'missions': {}
        }
        
        for mission, df in results.items():
            summary['missions'][mission] = {
                'rows': len(df),
                'columns': len(df.columns) if len(df) > 0 else 0,
                'success': len(df) > 0,
                'filename': self.search_queries[mission]['filename'] if len(df) > 0 else None
            }
        
        summary_path = os.path.join(DATA_DIR, 'fetch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Fetch summary saved to {summary_path}")


def main():
    """Main execution function"""
    try:
        fetcher = ExoplanetDataFetcher()
        
        # Option 1: Fetch specific mission
        # df = fetcher.fetch_latest_data('TESS_TOI')
        
        # Option 2: Fetch all missions (recommended for hackathon)
        results = fetcher.fetch_all_missions()
        fetcher.save_fetch_summary(results)
        
        # Display results summary
        print("\n" + "="*60)
        print("🌟 EXOPLANET DATA FETCH SUMMARY")
        print("="*60)
        
        successful_missions = []
        for mission, df in results.items():
            status = "✅ SUCCESS" if len(df) > 0 else "❌ FAILED"
            rows = len(df) if len(df) > 0 else 0
            print(f"{mission:20} | {status:10} | {rows:6} rows")
            
            if len(df) > 0:
                successful_missions.append(mission)
        
        print("="*60)
        print(f"📈 Successful missions: {len(successful_missions)}/{len(results)}")
        print(f"📁 Data saved to: {DATA_DIR}/")
        
        if successful_missions:
            print(f"🎯 Ready for ML pipeline integration!")
            print(f"💡 Next steps:")
            print(f"   1. Run your data preprocessing: python data_preprocessing_enhanced.py")
            print(f"   2. Integrate with existing pipeline")
            print(f"   3. Add 'Refresh Data' button to Streamlit app")
        else:
            print(f"⚠️  No data fetched. Check your API key and network connection.")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print(f"💡 Check your .env file and API key configuration")


if __name__ == "__main__":
    main()