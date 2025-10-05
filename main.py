# exoplanet_streamlit_app.py
"""
Enhanced Streamlit app for Exoplanet Classification Pipeline

This version adds an **Inference** panel where users can:
- Select a saved model and run predictions on new data
- Provide a single observation via manual form (numeric inputs) and get prediction + probabilities
- Upload a CSV for batch inference — either a raw mission CSV (KOI/TOI/K2) which will be mapped, cleaned, and engineered automatically, or a prepared features CSV with matching feature columns
- Download prediction results as CSV

Run:
    streamlit run exoplanet_streamlit_app.py

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib
Optional: lightgbm, shap
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, os, pickle, json
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Optional imports
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ----------------- Utility functions -----------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, low_memory=False)

SCHEMA_MAP = {
    'orbital_period': {'koi':'koi_period','toi':'pl_orbper','k2':'pl_orbper'},
    'transit_duration': {'koi':'koi_duration','toi':'pl_trandurh','k2':'pl_trandur'},
    'transit_depth': {'koi':'koi_depth','toi':'pl_trandep','k2':'pl_trandep'},
    'planet_radius': {'koi':'koi_prad','toi':'pl_rade','k2':'pl_rade'},
    'radius_ratio': {'koi':'koi_ror','toi':None,'k2':'pl_ratror'},
    'stellar_teff': {'koi':'koi_steff','toi':'st_teff','k2':'st_teff'},
    'stellar_radius': {'koi':'koi_srad','toi':'st_rad','k2':'st_rad'},
    'stellar_mass': {'koi':'koi_smass','toi':None,'k2':'st_mass'},
    'insolation_flux': {'koi':'koi_insol','toi':'pl_insol','k2':'pl_insol'},
    'teq': {'koi':'koi_teq','toi':'pl_eqt','k2':'pl_eqt'},
    'label': {'koi':'koi_disposition','toi':'tfopwg_disp','k2':'disposition'}
}

def standardize_df(df, mission_code):
    out = {}
    for std_col, mapping in SCHEMA_MAP.items():
        src = mapping.get(mission_code)
        if src and src in df.columns:
            out[std_col] = df[src]
        else:
            out[std_col] = pd.Series([None]*len(df))
    res = pd.DataFrame(out)
    res['mission'] = mission_code
    return res

def build_unified(dfs):
    parts = []
    for mission_code, df in dfs.items():
        parts.append(standardize_df(df, mission_code))
    unified = pd.concat(parts, ignore_index=True)
    unified['mission'] = unified['mission'].map({'koi':'Kepler','toi':'TESS','k2':'K2'})
    return unified

def normalize_label(x):
    if pd.isna(x): return None
    txt = str(x).strip().upper()
    if txt in ('CONFIRMED','CP','KP'): return 'Confirmed'
    if txt in ('CANDIDATE','PC'): return 'Candidate'
    if txt in ('FALSE POSITIVE', 'FP', 'APC', 'FA', 'REFUTED'): return 'False Positive'
    return txt.title()

def clean_and_align(unified):
    df = unified.copy()
    df['label'] = df['label'].apply(normalize_label)
    def depth_to_ppm(row):
        v = row['transit_depth']
        try:
            vv = float(v)
        except:
            return np.nan
        if row['mission']=='K2':
            return vv * 10000.0
        return vv
    df['transit_depth_ppm'] = df.apply(depth_to_ppm, axis=1)
    num_cols = ['orbital_period','transit_duration','transit_depth_ppm','planet_radius',
                'radius_ratio','stellar_teff','stellar_radius','stellar_mass','insolation_flux','teq']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df.groupby('mission')[c].transform(lambda g: g.fillna(g.median()))
        df[c] = df[c].fillna(df[c].median())
    return df

def engineer_features(df):
    df = df.copy()
    earth_per_sun = 695700.0/6371.0
    df['radius_ratio_calc'] = np.nan
    mask_need = df['radius_ratio'].isna() & df['planet_radius'].notna() & df['stellar_radius'].notna()
    df.loc[mask_need, 'radius_ratio_calc'] = df.loc[mask_need, 'planet_radius'] / (df.loc[mask_need, 'stellar_radius'] * earth_per_sun)
    df['radius_ratio_final'] = df['radius_ratio']
    df.loc[df['radius_ratio_final'].isna(), 'radius_ratio_final'] = df.loc[df['radius_ratio_final'].isna(), 'radius_ratio_calc']
    df['transit_depth_frac'] = df['transit_depth_ppm'] / 1e6
    df['expected_depth_frac'] = df['radius_ratio_final']**2
    valid = (df['expected_depth_frac']>0) & df['transit_depth_frac'].notna()
    df['depth_ratio'] = np.nan
    df.loc[valid, 'depth_ratio'] = df.loc[valid,'transit_depth_frac'] / df.loc[valid,'expected_depth_frac']
    df['depth_diff'] = df['transit_depth_frac'] - df['expected_depth_frac']
    df['snr_proxy'] = np.nan
    mask = df['transit_duration'].notna() & (df['transit_duration']>0) & df['transit_depth_ppm'].notna()
    df.loc[mask, 'snr_proxy'] = df.loc[mask,'transit_depth_ppm'] / np.sqrt(df.loc[mask,'transit_duration'])
    for c in ['orbital_period','planet_radius','transit_depth_ppm','insolation_flux','teq']:
        df[f'log1p_{c}'] = np.log1p(df[c].clip(lower=0).fillna(0))
    df['habitable_zone_flag'] = df['insolation_flux'].apply(lambda x: 1 if pd.notna(x) and (0.25 <= x <= 2.0) else 0)
    eng_feats = ['radius_ratio_final','radius_ratio_calc','transit_depth_frac','expected_depth_frac',
                 'depth_ratio','depth_diff','snr_proxy','log1p_orbital_period','log1p_planet_radius',
                 'log1p_transit_depth_ppm','log1p_insolation_flux','log1p_teq','habitable_zone_flag']
    for c in eng_feats:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df.groupby('mission')[c].transform(lambda g: g.fillna(g.median()))
        df[c] = df[c].fillna(df[c].median())
    return df

def prepare_training(df, features):
    labeled = df[df['label'].notna()].copy()
    for c in features:
        labeled[c] = pd.to_numeric(labeled[c], errors='coerce')
        labeled[c] = labeled.groupby('mission')[c].transform(lambda g: g.fillna(g.median()))
        labeled[c] = labeled[c].fillna(labeled[c].median())
    X = labeled[features].values
    le = LabelEncoder()
    y = le.fit_transform(labeled['label'].astype(str).values)
    return X, y, le, labeled

def train_model(X_train, y_train, model_name, params):
    if model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=int(params.get('n_estimators',200)),
                                     max_depth=None if params.get('max_depth') is None else int(params.get('max_depth')),
                                     class_weight='balanced', random_state=42, n_jobs=1)
    elif model_name == 'ExtraTrees':
        clf = ExtraTreesClassifier(n_estimators=int(params.get('n_estimators',200)), class_weight='balanced', random_state=42, n_jobs=1)
    elif model_name == 'HistGradientBoosting':
        clf = HistGradientBoostingClassifier(max_iter=int(params.get('n_estimators',200)), random_state=42)
    elif model_name == 'LightGBM' and LGB_AVAILABLE:
        clf = lgb.LGBMClassifier(n_estimators=int(params.get('n_estimators',200)), learning_rate=float(params.get('learning_rate',0.1)), class_weight='balanced', random_state=42)
    else:
        raise ValueError('Unsupported model or LightGBM not available')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X, y, label_encoder):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, target_names=label_encoder.classes_)
    conf = confusion_matrix(y, preds)
    return acc, report, conf

# Visualization helpers (same as before)
def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6,3))
    sns.countplot(data=df, x='label', order=df['label'].value_counts().index, ax=ax)
    ax.set_title('Class distribution (labeled rows)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_mission_breakdown(df):
    fig, ax = plt.subplots(figsize=(6,3))
    sns.countplot(data=df, x='mission', hue='label', ax=ax)
    ax.set_title('Mission breakdown by label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

from sklearn.preprocessing import label_binarize

def plot_roc_pr(clf, X, y, le):
    classes = le.classes_
    y_bin = label_binarize(y, classes=range(len(classes)))
    if hasattr(clf, 'predict_proba'):
        prob = clf.predict_proba(X)
    else:
        if hasattr(clf, 'decision_function'):
            prob = clf.decision_function(X)
            exp = np.exp(prob - np.max(prob, axis=1, keepdims=True))
            prob = exp / exp.sum(axis=1, keepdims=True)
        else:
            return None, None
    roc_fig, axr = plt.subplots(figsize=(6,4))
    pr_fig, axp = plt.subplots(figsize=(6,4))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        roc_auc = auc(fpr, tpr)
        axr.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
        precision, recall, _ = precision_recall_curve(y_bin[:, i], prob[:, i])
        ap = average_precision_score(y_bin[:, i], prob[:, i])
        axp.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
    axr.plot([0,1],[0,1],'k--',alpha=0.3)
    axr.set_xlabel('FPR')
    axr.set_ylabel('TPR')
    axr.set_title('ROC curves (one-vs-rest)')
    axr.legend()
    axp.set_xlabel('Recall')
    axp.set_ylabel('Precision')
    axp.set_title('Precision-Recall curves')
    axp.legend()
    plt.tight_layout()
    return roc_fig, pr_fig

def plot_calibration(clf, X, y, le, n_bins=10):
    classes = le.classes_
    if not hasattr(clf, 'predict_proba'):
        return None
    prob = clf.predict_proba(X)
    fig, ax = plt.subplots(1, len(classes), figsize=(4*len(classes),3))
    if len(classes) == 1:
        ax = [ax]
    for i, cls in enumerate(classes):
        true, pred = calibration_curve((y==i).astype(int), prob[:, i], n_bins=n_bins)
        ax[i].plot(pred, true, marker='o')
        ax[i].plot([0,1],[0,1],'k--',alpha=0.3)
        ax[i].set_title(f'Calib: {cls}')
        ax[i].set_xlabel('Mean predicted prob')
        ax[i].set_ylabel('Fraction positive')
    plt.tight_layout()
    return fig

def plot_learning_curve(clf, X, y, title='Learning Curve'):
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=3, n_jobs=1, train_sizes=np.linspace(0.1,1.0,5))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(train_sizes, train_mean, 'o-', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', label='Cross-val score')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_pca_tsne(df, features, label_col='label', sample_size=1000):
    X = df[features].values
    labels = df[label_col].astype(str).values
    if len(df) > sample_size:
        idx = np.random.choice(range(len(df)), sample_size, replace=False)
        Xs = X[idx]
        labs = labels[idx]
    else:
        Xs = X
        labs = labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(Xs)
    fig1, ax1 = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labs, ax=ax1, legend='full', s=20)
    ax1.set_title('PCA 2D')
    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labs, ax=ax2, legend='full', s=20)
    ax2.set_title('t-SNE 2D')
    plt.tight_layout()
    return fig1, fig2

# Final features list used for models and inference
FINAL_FEATURES = ['orbital_period','transit_duration','transit_depth_ppm','planet_radius',
                  'radius_ratio_final','stellar_teff','stellar_radius','stellar_mass','insolation_flux','teq',
                  'depth_ratio','depth_diff','snr_proxy','log1p_orbital_period','log1p_planet_radius','log1p_transit_depth_ppm','habitable_zone_flag']

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title='Exoplanet Classifier', layout='wide')
st.title('Exoplanet Classification Lab — Visual Explorer & Inference')

# Sidebar: Data ingestion
st.sidebar.header('Data Ingestion')
upload_koi = st.sidebar.file_uploader('Upload KOI CSV (Kepler)', type=['csv'])
upload_toi = st.sidebar.file_uploader('Upload TOI CSV (TESS)', type=['csv'])
upload_k2  = st.sidebar.file_uploader('Upload K2 CSV', type=['csv'])
use_cleaned = st.sidebar.file_uploader('Or upload a unified cleaned CSV (optional)', type=['csv'])

if 'state' not in st.session_state:
    st.session_state['state'] = {'raw_dfs':{}, 'unified':None, 'processed':None, 'model_history':[]}

uploaded_map = {}
if upload_koi is not None:
    try:
        dfk = load_csv(upload_koi)
        uploaded_map['koi'] = dfk
        st.sidebar.success('Loaded KOI')
    except Exception as e:
        st.sidebar.error(f'KOI load failed: {e}')
if upload_toi is not None:
    try:
        dft = load_csv(upload_toi)
        uploaded_map['toi'] = dft
        st.sidebar.success('Loaded TOI')
    except Exception as e:
        st.sidebar.error(f'TOI load failed: {e}')
if upload_k2 is not None:
    try:
        dfk2 = load_csv(upload_k2)
        uploaded_map['k2'] = dfk2
        st.sidebar.success('Loaded K2')
    except Exception as e:
        st.sidebar.error(f'K2 load failed: {e}')
if use_cleaned is not None:
    try:
        df_cleaned = load_csv(use_cleaned)
        st.session_state['state']['processed'] = df_cleaned
        st.sidebar.success('Loaded cleaned unified CSV')
    except Exception as e:
        st.sidebar.error(f'Cleaned CSV load failed: {e}')

# Controls: Build/Clean/Reset
st.header('Pipeline Controls')
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button('Build unified dataset from uploaded files'):
        if not uploaded_map:
            st.warning('No mission CSVs uploaded. Upload at least one (KOI/TOI/K2).')
        else:
            unified = build_unified(uploaded_map)
            st.session_state['state']['unified'] = unified
            st.success('Unified dataset built — preview below')
with col2:
    if st.button('Clean & feature-engineer'):
        if st.session_state['state']['unified'] is None and st.session_state['state']['processed'] is None:
            st.warning('No unified data available. Upload files or a cleaned CSV.')
        else:
            base_df = st.session_state['state']['unified'] if st.session_state['state']['unified'] is not None else st.session_state['state']['processed']
            proc = clean_and_align(base_df)
            proc = engineer_features(proc)
            st.session_state['state']['processed'] = proc
            st.success('Data cleaned & features engineered — preview below')
with col3:
    if st.button('Reset session'):
        st.session_state.clear()
        st.experimental_rerun()

# Previews
st.subheader('Data Preview')
if st.session_state['state']['unified'] is not None:
    st.write('Unified (raw) — first 5 rows')
    st.dataframe(st.session_state['state']['unified'].head())
if st.session_state['state']['processed'] is not None:
    st.write('Processed (cleaned + engineered) — first 10 rows')
    st.dataframe(st.session_state['state']['processed'].head(10))

# Visualizations: class distribution & mission breakdown
st.subheader('Dataset Visualizations')
if st.session_state['state']['processed'] is not None:
    proc = st.session_state['state']['processed']
    fig1 = plot_class_distribution(proc)
    st.pyplot(fig1)
    fig2 = plot_mission_breakdown(proc)
    st.pyplot(fig2)

# Model training controls
st.sidebar.header('Training / Model')
model_choice = st.sidebar.selectbox('Model', ['RandomForest','ExtraTrees','HistGradientBoosting'] + (['LightGBM'] if LGB_AVAILABLE else []))
n_estimators = st.sidebar.slider('n_estimators', 50, 1000, 200, step=50)
max_depth = st.sidebar.slider('max_depth (RF/ET)', 1, 50, 12)
learning_rate = st.sidebar.slider('learning_rate (LGB)', 0.01, 0.5, 0.1)
test_size = st.sidebar.slider('test_size (fraction)', 0.1, 0.5, 0.2)

# Train
st.header('Train Model & Visualize')
if st.button('Train model on processed data'):
    if st.session_state['state']['processed'] is None:
        st.warning('No processed data. Please Build unified and Clean & feature-engineer first.')
    else:
        proc = st.session_state['state']['processed']
        X, y, le, labeled_df = prepare_training(proc, FINAL_FEATURES)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        try:
            clf = train_model(X_train_s, y_train, model_choice, params)
        except Exception as e:
            st.error(f'Model training failed: {e}')
            raise
        acc, report, conf = evaluate_model(clf, X_test_s, y_test, le)
        st.success(f'Training completed. Test accuracy: {acc:.4f}')
        st.text('Classification report:\n')
        st.text(report)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(conf, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
        if hasattr(clf, 'feature_importances_'):
            imp = clf.feature_importances_
            feat_imp = pd.DataFrame({'feature': FINAL_FEATURES, 'importance': imp}).sort_values('importance', ascending=False)
            st.subheader('Feature importances')
            st.dataframe(feat_imp.head(20))
            if st.button('Plot feature importances'):
                figfi, axfi = plt.subplots(figsize=(6,6))
                sns.barplot(x='importance', y='feature', data=feat_imp.head(20), ax=axfi)
                axfi.set_title('Top 20 feature importances')
                st.pyplot(figfi)
        model_obj = {'model': clf, 'label_encoder': le, 'scaler': scaler, 'features': FINAL_FEATURES, 'metrics': {'accuracy': acc}}
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        model_id = f"model_{model_choice}_{len(st.session_state['state']['model_history'])+1}"
        p = models_dir / f"{model_id}.pkl"
        joblib.dump(model_obj, p)
        st.session_state['state']['model_history'].append({'id': model_id, 'path': str(p), 'accuracy': acc})
        st.success(f'Model saved to {p}')
        if hasattr(clf, 'predict_proba'):
            roc_fig, pr_fig = plot_roc_pr(clf, X_test_s, y_test, le)
            if roc_fig is not None:
                st.subheader('ROC curves (one-vs-rest)')
                st.pyplot(roc_fig)
            if pr_fig is not None:
                st.subheader('Precision-Recall curves')
                st.pyplot(pr_fig)
            calib_fig = plot_calibration(clf, X_test_s, y_test, le)
            if calib_fig is not None:
                st.subheader('Calibration plots')
                st.pyplot(calib_fig)
        lc_fig = plot_learning_curve(clf, X, y, title=f'Learning curve ({model_choice})')
        st.subheader('Learning curve')
        st.pyplot(lc_fig)
        try:
            pca_fig, tsne_fig = plot_pca_tsne(labeled_df, FINAL_FEATURES, sample_size=1000)
            st.subheader('PCA 2D')
            st.pyplot(pca_fig)
            st.subheader('t-SNE 2D (sample)')
            st.pyplot(tsne_fig)
        except Exception as e:
            st.warning(f'PCA/t-SNE failed or is slow on your environment: {e}')

# Model management & thresholding
st.sidebar.header('Model Management')
if st.session_state['state']['model_history']:
    st.sidebar.write('Saved models:')
    for i, mh in enumerate(st.session_state['state']['model_history']):
        st.sidebar.write(f"{i+1}. {mh['id']} — acc: {mh['accuracy']:.3f}")
    sel = st.sidebar.selectbox('Load model index', options=[None]+list(range(len(st.session_state['state']['model_history']))))
    if sel is not None:
        mh = st.session_state['state']['model_history'][sel]
        loadp = Path(mh['path'])
        if loadp.exists():
            loaded = joblib.load(loadp)
            st.sidebar.success(f"Loaded {mh['id']}")
            if st.sidebar.button('Download model'):
                with open(loadp, 'rb') as f:
                    st.sidebar.download_button('Download .pkl', f, file_name=loadp.name)
            if 'model' in loaded:
                clf = loaded['model']
                le = loaded['label_encoder']
                features = loaded['features']
                if hasattr(clf, 'predict_proba'):
                    target_class = st.sidebar.selectbox('Threshold target class', options=list(le.classes_))
                    thresh = st.sidebar.slider('Probability threshold for target class', 0.0, 1.0, 0.5)
                    if st.sidebar.button('Apply threshold to full labeled dataset'):
                        proc = st.session_state['state']['processed']
                        X_all, y_all, le2, labeled_all = prepare_training(proc, features)
                        X_all_s = loaded['scaler'].transform(X_all)
                        probs = clf.predict_proba(X_all_s)
                        target_idx = list(le.classes_).index(target_class)
                        preds_by_thresh = np.argmax(probs, axis=1)
                        override = probs[:, target_idx] > thresh
                        preds = preds_by_thresh.copy()
                        preds[override] = target_idx
                        acc_t = accuracy_score(y_all, preds)
                        st.sidebar.write(f'Accuracy with thresholding on full labeled set: {acc_t:.4f}')

# Ingest new data and append
st.header('Ingest new data (append)')
new_file = st.file_uploader('Upload a new mission CSV to ingest', type=['csv'], key='ingest')
if new_file is not None:
    try:
        newdf = pd.read_csv(new_file, low_memory=False)
        st.write('Uploaded — preview')
        st.dataframe(newdf.head())
        mission_code = st.selectbox('Select mission mapping for this file', options=['koi','toi','k2'])
        if st.button('Map & append to processed dataset'):
            std = standardize_df(newdf, mission_code)
            base = st.session_state['state']['unified'] if st.session_state['state']['unified'] is not None else pd.DataFrame()
            appended = pd.concat([base, std], ignore_index=True)
            proc = clean_and_align(appended)
            proc = engineer_features(proc)
            st.session_state['state']['processed'] = proc
            st.success('New data appended and processed. You can retrain the model using Train model button.')
    except Exception as e:
        st.error(f'Failed to ingest file: {e}')

# ----------------- NEW: Inference Panel -----------------
st.header('Inference — Single or Batch')
st.write('Use a saved model to make predictions. You can enter a single example manually, or upload a CSV for batch inference.')

# Select model
model_idx = None
if st.session_state['state']['model_history']:
    model_idx = st.selectbox('Select saved model to use for inference', options=list(range(len(st.session_state['state']['model_history']))), format_func=lambda i: st.session_state['state']['model_history'][i]['id'])

if model_idx is None:
    st.info('Train and save a model first, then it will appear here for inference.')
else:
    model_meta = st.session_state['state']['model_history'][model_idx]
    model_obj = joblib.load(Path(model_meta['path']))
    clf = model_obj['model']
    scaler = model_obj['scaler']
    label_enc = model_obj['label_encoder']
    features = model_obj['features']

    # Inference method tabs
    tab1, tab2 = st.tabs(['Manual input','Batch CSV'])

    # Manual input form
    with tab1:
        st.subheader('Manual single-example prediction')
        with st.form('manual_form'):
            inputs = {}
            # populate defaults from processed medians if available
            proc = st.session_state['state']['processed']
            for f in features:
                default = 0.0
                try:
                    if proc is not None and f in proc.columns:
                        default = float(proc[f].median())
                except Exception:
                    default = 0.0
                inputs[f] = st.number_input(f, value=float(default), format="%.6f")
            submitted = st.form_submit_button('Predict single example')
        if submitted:
            X_new = np.array([[inputs[f] for f in features]])
            X_new_s = scaler.transform(X_new)
            pred_idx = clf.predict(X_new_s)[0]
            pred_label = label_enc.inverse_transform([pred_idx])[0]
            probs = clf.predict_proba(X_new_s)[0] if hasattr(clf,'predict_proba') else None
            st.write('**Prediction:**', pred_label)
            if probs is not None:
                proba_df = pd.DataFrame({'class': label_enc.classes_, 'probability': probs})
                st.dataframe(proba_df)
                # Download
                csv_buf = proba_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download probabilities CSV', csv_buf, file_name='prediction_probabilities.csv')

    # Batch CSV inference
    with tab2:
        st.subheader('Batch inference from CSV')
        batch_file = st.file_uploader('Upload CSV for prediction (raw mission CSV or features CSV)', type=['csv'], key='batch_inf')
        if batch_file is not None:
            try:
                df_batch = pd.read_csv(batch_file, low_memory=False)
                st.write('Uploaded batch — preview')
                st.dataframe(df_batch.head())
                infer_mode = st.selectbox('Is this a raw mission CSV or a processed/features CSV?', options=['raw_mission','features_csv'])
                if infer_mode == 'raw_mission':
                    mission_code = st.selectbox('Select mission for this raw file', options=['koi','toi','k2'])
                if st.button('Run batch inference'):
                    if infer_mode == 'raw_mission':
                        std = standardize_df(df_batch, mission_code)
                        std = clean_and_align(std)
                        std = engineer_features(std)
                        X_batch = std[features].values
                    else:
                        # Expect the file to contain the model feature columns
                        missing = [f for f in features if f not in df_batch.columns]
                        if missing:
                            st.error(f'Missing feature columns in uploaded CSV: {missing}')
                            st.stop()
                        X_batch = df_batch[features].values
                    X_batch_s = scaler.transform(X_batch)
                    preds = clf.predict(X_batch_s)
                    labels = label_enc.inverse_transform(preds)
                    result_df = pd.DataFrame({'prediction': labels})
                    if hasattr(clf,'predict_proba'):
                        probs = clf.predict_proba(X_batch_s)
                        for i, cls in enumerate(label_enc.classes_):
                            result_df[f'prob_{cls}'] = probs[:, i]
                    # Append original identifiers if present
                    out = pd.concat([df_batch.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
                    st.write('Prediction results (first 20 rows)')
                    st.dataframe(out.head(20))
                    csv_bytes = out.to_csv(index=False).encode('utf-8')
                    st.download_button('Download predictions CSV', csv_bytes, file_name='batch_predictions.csv')
            except Exception as e:
                st.error(f'Batch inference failed: {e}')

# Full-dataset evaluation
st.header('Model Evaluation (full labeled dataset)')
if st.button('Evaluate latest saved model on full labeled dataset'):
    if not st.session_state['state']['model_history']:
        st.warning('No saved models. Train and save a model first.')
    elif st.session_state['state']['processed'] is None:
        st.warning('No processed dataset available.')
    else:
        last = st.session_state['state']['model_history'][-1]
        obj = joblib.load(last['path'])
        model = obj['model']
        scaler = obj['scaler']
        features = obj['features']
        proc = st.session_state['state']['processed']
        X, y, le, labeled_df = prepare_training(proc, features)
        X_s = scaler.transform(X)
        acc, report, conf = evaluate_model(model, X_s, y, le)
        st.success(f'Accuracy on full labeled dataset: {acc:.4f}')
        st.text(report)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(conf, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
        if hasattr(model, 'predict_proba'):
            rfig, pfig = plot_roc_pr(model, X_s, y, le)
            st.subheader('ROC (full dataset)')
            st.pyplot(rfig)
            st.subheader('Precision-Recall (full dataset)')
            st.pyplot(pfig)

st.markdown('---')
st.caption('This interactive UI is a visualization-rich starter toolbox. For production: add user auth, persistent DB for data and models, input validation, and logging.')

# End of file
