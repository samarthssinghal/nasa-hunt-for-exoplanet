# 🔍 Serp API Integration for NASA Exoplanet Data

Dynamic data fetching system that pulls the latest exoplanet datasets from NASA archives using Serp API.

## 🚀 Quick Setup for Developers

### 1. Get Your API Key (Free)
1. Sign up at [serpapi.com](https://serpapi.com/users/sign_up)
2. Get 100 free searches/month (perfect for prototyping)
3. Copy your API key from the dashboard

### 2. Environment Setup
```bash
# Clone and navigate to project
git clone <your-repo-url>
cd nasa_exoplanets

# Copy environment template
cp .env.template .env

# Edit .env file and replace 'your_api_key_here' with your actual key
# Example: SERPAPI_KEY=abc123def456...

# Install dependencies
pip install -r requirements.txt
```

### 3. Test Your Setup
```bash
# Quick API test
python test_serp_api.py

# Full data fetch
python fetch_exo_data.py
```

## 📊 What It Does

### Data Sources Searched:
- **TESS TOI**: Latest TESS Objects of Interest
- **Kepler KOI**: Kepler Objects of Interest  
- **K2 Candidates**: K2 mission exoplanet candidates
- **JWST Confirmations**: Latest JWST exoplanet data
- **NEOSSat CSA**: Canadian Space Agency data
- **General Updates**: Latest NASA exoplanet catalogs

### Output:
- Downloads CSV files to `data/` directory
- Creates `fetch_summary.json` with operation details
- Logs all activity with timestamps

## 🔧 Integration with ML Pipeline

### Option 1: Manual Refresh
```python
from fetch_exo_data import ExoplanetDataFetcher

fetcher = ExoplanetDataFetcher()
tess_data = fetcher.fetch_latest_data('TESS_TOI')
```

### Option 2: Scheduled Updates
```bash
# Add to crontab for weekly updates
0 0 * * 0 cd /path/to/project && python fetch_exo_data.py
```

### Option 3: Streamlit Integration
```python
import streamlit as st
from fetch_exo_data import ExoplanetDataFetcher

if st.button("🔄 Refresh Latest Data"):
    fetcher = ExoplanetDataFetcher()
    results = fetcher.fetch_all_missions()
    st.success(f"Updated data for {len([r for r in results.values() if len(r) > 0])} missions")
```

## 💡 Usage Tips

### For Hackathons:
- Free tier gives 100 searches (plenty for prototyping)
- Takes ~2-4 minutes to fetch all missions
- Results cached in `data/` for offline use

### For Production:
- Upgrade to paid plan for more searches
- Implement caching to avoid duplicate API calls
- Set up monitoring for data freshness

## 🛠 Troubleshooting

### No Data Found?
- Try broader search terms
- Check if target sites have updated their structure
- Some datasets may not be in CSV format

### API Errors?
- Verify your API key in `.env`
- Check remaining quota at serpapi.com
- Ensure stable internet connection

### Rate Limiting?
- Increase `SEARCH_DELAY` in `.env`
- Reduce number of search results

## 📁 File Structure
```
nasa_exoplanets/
├── .env                    # Your API key (not in git)
├── .env.template          # Template for other developers
├── fetch_exo_data.py      # Main fetching script
├── test_serp_api.py       # Quick API test
├── requirements.txt       # Dependencies
└── data/                  # Downloaded datasets
    ├── fetch_summary.json # Operation summary
    └── *.csv              # Downloaded data
```

## 🎯 Next Steps
1. **Test the setup**: Run `python test_serp_api.py`
2. **Fetch some data**: Run `python fetch_exo_data.py`
3. **Integrate with your ML pipeline**: Import and use the fetcher
4. **Add to Streamlit**: Create refresh button for live updates

---
*Built for NASA Space Apps Challenge - Making exoplanet research dynamic and current! 🌟*