# 🌟 Exoplanet Data Hunter - Streamlit Guide

Interactive web interface for testing Serp API functionality and searching for the latest exoplanet datasets.

## 🚀 Quick Start

### Option 1: Simple Launch
```bash
python launch_app.py
```

### Option 2: Direct Streamlit
```bash
streamlit run streamlit_serp_test.py
```

### Option 3: Custom Port
```bash
streamlit run streamlit_serp_test.py --server.port 8502
```

## 🎯 Features

### 📊 Predefined Mission Types
The app includes optimized search configurations for:

1. **Kepler Objects of Interest (KOI)**
   - Keywords: "Kepler Objects of Interest", "KOI", "kepler disposition"
   - Sites: exoplanetarchive.ipac.caltech.edu, nasa.gov
   - Focus: Confirmed exoplanets, planetary candidates, false positives

2. **TESS Objects of Interest (TOI)**
   - Keywords: "TESS Objects of Interest", "TOI", "TFOPWG Disposition"
   - Sites: exoplanetarchive.ipac.caltech.edu, tess.mit.edu
   - Focus: Latest TESS mission discoveries and classifications

3. **K2 Planets and Candidates**
   - Keywords: "K2 exoplanet candidates", "Archive Disposition"
   - Sites: exoplanetarchive.ipac.caltech.edu, nasa.gov
   - Focus: K2 mission data with Archive Disposition classifications

4. **JWST Exoplanet Data**
   - Keywords: "JWST exoplanet", "James Webb atmospheric", "JWST transit"
   - Sites: mast.stsci.edu, jwst.nasa.gov, exoplanetarchive.ipac.caltech.edu
   - Focus: Latest JWST atmospheric and transit observations

5. **NEOSSat Astronomy Data**
   - Keywords: "NEOSSat exoplanet", "Canadian Space Agency exoplanet"
   - Sites: donnees-data.asc-csa.gc.ca, asc-csa.gc.ca
   - Focus: Canadian Space Agency's NEOSSat astronomical data

6. **Latest Confirmed Exoplanets**
   - Keywords: "latest confirmed exoplanets", "new exoplanet discoveries"
   - Sites: exoplanetarchive.ipac.caltech.edu, nasa.gov
   - Focus: Most recent confirmed discoveries and updates

### 🔍 Search Capabilities

#### Smart Query Building
- Automatically combines mission-specific keywords
- Adds temporal constraints (2024-2025, 2023-2024, etc.)
- Targets official NASA/CSA data repositories
- Prefers structured data formats (CSV, TXT, FITS)

#### Custom Search Options
- **Additional Keywords**: Add custom terms like "atmospheric composition", "transit photometry"
- **Time Range**: Focus on specific years or "latest" data
- **Result Count**: Control number of search results (5-20)

#### Result Categorization
Search results are automatically categorized into:
- 📁 **Data Files**: Direct CSV/TXT downloads
- 📄 **Research Papers**: Academic articles and preprints
- 🏛️ **Official Sites**: NASA, CSA, and observatory pages
- 🔗 **Other Results**: Additional relevant content

### 📥 Data Download & Preview

#### Automatic Download
- Click "📥 Try Download" on data file results
- Automatically attempts to parse as CSV
- Shows data preview with first 10 rows
- Saves to local `data/` directory

#### Data Statistics
For successfully downloaded datasets:
- Total rows and columns
- Number of numeric columns
- Missing value counts
- Basic data quality metrics

### 📜 Search History
- Tracks last 20 searches with timestamps
- Shows mission type, query, and result counts
- Includes custom keywords used
- Stored in `data/search_history.json`

### 🧪 API Testing
- **Test API Connection**: Quick connectivity check
- **File Counter**: Shows downloaded datasets
- **Data Directory**: Browse local data files

## 💡 Usage Tips

### For Research & Discovery
1. **Start Broad**: Use predefined mission types first
2. **Add Specifics**: Include custom keywords for targeted searches
3. **Check Recent**: Use "2024-2025" timeframe for latest data
4. **Download & Preview**: Test data quality before ML integration

### For ML Pipeline Integration
1. **Test Downloads**: Verify data format and columns match your model
2. **Monitor Updates**: Regular searches can catch new dataset releases
3. **Automate**: Use search history to track what's already been processed
4. **Validate**: Check for new columns or classification schemes

### Search Strategy Examples

#### Finding Latest TESS Data
```
Mission Type: TESS Objects of Interest (TOI)
Custom Keywords: "confirmed planets, 2025"
Time Range: 2024-2025
```

#### Atmospheric Composition Data
```
Mission Type: JWST Exoplanet Data
Custom Keywords: "atmospheric composition, spectroscopy, transmission"
Time Range: latest
```

#### Updated Classifications
```
Mission Type: Latest Confirmed Exoplanets
Custom Keywords: "reclassified, updated disposition"
Time Range: 2024-2025
```

## 🔧 Integration with ML Pipeline

### Direct Import
```python
# In your ML script
import pandas as pd
from streamlit_serp_test import StreamlitSerpTester

# Initialize and search
tester = StreamlitSerpTester()
results = tester.search_serp("latest TESS TOI 2025", 10)

# Process results for data files
for result in results:
    if 'csv' in result['link']:
        df = tester.attempt_download(result['link'], 'new_data.csv')
        if df is not None:
            # Integrate with your preprocessing pipeline
            processed_df = your_preprocessing_function(df)
```

### Streamlit Integration
```python
# Add to existing Streamlit app
if st.button("🔄 Refresh Training Data"):
    tester = StreamlitSerpTester()
    new_data = tester.search_serp("latest exoplanet confirmations", 5)
    st.success(f"Found {len(new_data)} potential new datasets")
```

## 🛠 Troubleshooting

### No Results Found
- **Check API Key**: Verify .env configuration
- **Broaden Search**: Try fewer, more general keywords
- **Check Sites**: Some datasets may have moved locations
- **Try Different Timeframes**: Recent data might not be indexed yet

### Download Failures
- **File Format**: Not all results are downloadable CSVs
- **Access Restrictions**: Some sites require authentication
- **Rate Limiting**: Wait between download attempts
- **File Size**: Very large datasets might timeout

### Performance Issues
- **Reduce Results**: Lower the number of search results
- **Increase Delays**: Add longer pauses between API calls
- **Check Quota**: Monitor Serp API usage limits
- **Clear Cache**: Remove old files from data/ directory

## 📁 File Structure After Use
```
nasa_exoplanets/
├── streamlit_serp_test.py      # Main Streamlit app
├── launch_app.py               # Easy launch script
├── data/
│   ├── search_history.json     # Recent searches
│   ├── *.csv                   # Downloaded datasets
│   └── fetch_summary.json      # Download summaries
└── .env                        # Your API configuration
```

## 🎯 Next Steps
1. **Explore Mission Types**: Try each predefined search configuration
2. **Test Downloads**: Verify data quality and format compatibility
3. **Integrate Results**: Use discovered datasets in your ML pipeline
4. **Schedule Updates**: Set up regular searches for new data releases
5. **Share Findings**: Document useful search patterns for your team

---
*Built for NASA Space Apps Challenge - Dynamic exoplanet research! 🚀*