
# ==============================================================================
# SIGAP SAMPAH JOGJA - ADVANCED VERSION
# Sistem Informasi Geografis Analisis Potensi Sampah Kota Yogyakarta
# ==============================================================================

import os
import tempfile
import zipfile
import warnings
import subprocess
import time
import threading
import json
import datetime
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysal.lib import weights
from pysal.explore import esda
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

FOLDER_PATH = 'data/'
SHAPE_ADMIN = 'ADMINISTRASI_AR_KECAMATAN.shp'
EXCEL_PENDUDUK = 'Proyeksi Penduduk Kota Yogyakarta menurut Kecamatan dan Jenis Kelamin, 2015 – 2025, 2025.xlsx'
KMZ_FILENAME = 'BANK SAMPAH.kmz'
METRIC_CRS_EPSG = 32749
GEOGRAPHIC_CRS_EPSG = 4326

# Color schemes
LISA_COLORS = {
    'High-High': '#d73027',
    'Low-Low': '#313695',
    'Low-High': '#fee08b',
    'High-Low': '#abd9e9',
    'Not Significant': '#f7f7f7'
}

PRIORITY_COLORS = {
    'Sangat Tinggi': '#8B0000',
    'Tinggi': '#FF4500',
    'Sedang': '#FFD700',
    'Rendah': '#32CD32',
    'Sangat Rendah': '#4169E1'
}

HOTSPOT_COLORS = {
    'Hot Spot (99%)': '#8B0000',
    'Hot Spot (95%)': '#DC143C',
    'Hot Spot (90%)': '#FF6347',
    'Cold Spot (90%)': '#4169E1',
    'Cold Spot (95%)': '#0000CD',
    'Cold Spot (99%)': '#000080',
    'Not Significant': '#D3D3D3'
}

GEARY_COLORS = {
    'Strong Clustering': '#8B0000',
    'Moderate Clustering': '#FF4500',
    'Moderate Dispersion': '#32CD32',
    'Strong Dispersion': '#4169E1',
    'Random': '#D3D3D3'
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def read_kmz_or_kml(path_kmz: str) -> Optional[gpd.GeoDataFrame]:
    """Enhanced KMZ/KML reader with better error handling."""
    if not path_kmz or not os.path.exists(path_kmz):
        st.warning(f"File spasial tidak ditemukan: {path_kmz}")
        return None

    try:
        if path_kmz.lower().endswith('.kmz'):
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(path_kmz, 'r') as zf:
                    kml_files = [s for s in zf.namelist() if s.lower().endswith('.kml')]
                    if not kml_files:
                        st.error("Tidak ada file KML dalam arsip KMZ")
                        return None
                    zf.extractall(tmp)
                    kml_path = os.path.join(tmp, kml_files[0])
                    return gpd.read_file(kml_path, driver='KML')
        else:
            return gpd.read_file(path_kmz)
    except Exception as e:
        st.error(f"Error membaca file spasial: {e}")
        return None

def calculate_accessibility_index(gdf_kecamatan: gpd.GeoDataFrame,
                                gdf_bank_sampah: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate accessibility index to waste banks."""
    if gdf_bank_sampah.empty:
        gdf_kecamatan['Accessibility_Index'] = 0
        return gdf_kecamatan

    # Convert to metric CRS for distance calculation
    gdf_kec_metric = gdf_kecamatan.to_crs(epsg=METRIC_CRS_EPSG)
    gdf_bank_metric = gdf_bank_sampah.to_crs(epsg=METRIC_CRS_EPSG)

    accessibility_scores = []

    for idx, kecamatan in gdf_kec_metric.iterrows():
        centroid = kecamatan.geometry.centroid

        if gdf_bank_metric.empty:
            accessibility_scores.append(0)
            continue

        # Calculate distances to all waste banks
        distances = []
        for _, bank in gdf_bank_metric.iterrows():
            dist = centroid.distance(bank.geometry) / 1000  # Convert to km
            distances.append(dist)

        if distances:
            # Accessibility score based on nearest bank and density
            min_distance = min(distances)
            nearby_banks = len([d for d in distances if d <= 2])  # Within 2km

            # Higher score for closer banks and more nearby options
            accessibility_score = (1 / (1 + min_distance)) * (1 + nearby_banks * 0.1)
            accessibility_scores.append(accessibility_score)
        else:
            accessibility_scores.append(0)

    gdf_kecamatan['Accessibility_Index'] = accessibility_scores
    return gdf_kecamatan

def perform_cluster_analysis(gdf: gpd.GeoDataFrame,
                           features: List[str],
                           n_clusters: int = 4) -> gpd.GeoDataFrame:
    """Perform K-means clustering analysis."""
    data = gdf[features].fillna(0).replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    gdf['Cluster'] = clusters
    gdf['Cluster_Label'] = gdf['Cluster'].map({
        0: 'Cluster A', 1: 'Cluster B',
        2: 'Cluster C', 3: 'Cluster D'
    })

    return gdf, kmeans, scaler

def calculate_waste_generation_estimate(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Estimate daily waste generation per district."""
    # Average waste generation per person per day (kg) - Indonesian standard
    waste_per_person_kg = 0.7

    # Different factors for different land uses
    commercial_factor = 1.5  # Commercial areas generate more waste
    density_factor_threshold = 5000  # people per km2

    waste_estimates = []

    for idx, row in gdf.iterrows():
        base_waste = row['JML_PENDUDUK'] * waste_per_person_kg

        # Apply commercial factor
        commercial_boost = (row['Jml_Niaga'] / 100) * commercial_factor

        # Apply density factor
        if row['Kepadatan_Penduduk'] > density_factor_threshold:
            density_boost = 1.2
        else:
            density_boost = 1.0

        total_waste = base_waste * (1 + commercial_boost) * density_boost
        waste_estimates.append(total_waste)

    gdf['Estimated_Daily_Waste_Kg'] = waste_estimates
    gdf['Estimated_Daily_Waste_Ton'] = gdf['Estimated_Daily_Waste_Kg'] / 1000

    return gdf

# ==============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ==============================================================================

@st.cache_data
def run_comprehensive_analysis():
    """Enhanced analysis pipeline with additional features."""

    # Load administrative data
    shp_admin_path = os.path.join(FOLDER_PATH, SHAPE_ADMIN)
    gdf_admin_raw = gpd.read_file(shp_admin_path)
    gdf_admin = gdf_admin_raw[['WADMKC', 'geometry']].to_crs(epsg=METRIC_CRS_EPSG)

    # Filter for Yogyakarta districts
    kecamatan_kota_yogya = [
        'Mergangsan', 'Mantrijeron', 'Kraton', 'Wirobrajan', 'Gedongtengen',
        'Ngampilan', 'Danurejan', 'Gondomanan', 'Pakualaman', 'Jetis',
        'Tegalrejo', 'Gondokusuman', 'Umbulharjo', 'Kotagede'
    ]

    gdf_admin = gdf_admin[gdf_admin['WADMKC'].isin(kecamatan_kota_yogya)].reset_index(drop=True)
    gdf_admin['luas_km2'] = gdf_admin.geometry.area / 1_000_000

    # Load population data
    excel_path = os.path.join(FOLDER_PATH, EXCEL_PENDUDUK)
    df_penduduk_raw = pd.read_excel(excel_path, skiprows=3)
    df_penduduk = df_penduduk_raw.iloc[:, [0, -1]].copy()
    df_penduduk.columns = ['WADMKC', 'JML_PENDUDUK']
    df_penduduk['WADMKC_UPPER'] = df_penduduk['WADMKC'].astype(str).str.strip().str.upper()
    gdf_admin['WADMKC_UPPER'] = gdf_admin['WADMKC'].astype(str).str.strip().str.upper()
    gdf_admin = gdf_admin.merge(df_penduduk[['WADMKC_UPPER', 'JML_PENDUDUK']], on='WADMKC_UPPER', how='left')
    gdf_admin.drop(columns=['WADMKC_UPPER'], inplace=True)
    gdf_admin['JML_PENDUDUK'] = gdf_admin['JML_PENDUDUK'].fillna(0)

    # Enhanced feature engineering
    data_sources = {
        'niaga': 'NIAGA_PT_25K.shp',
        'pemukiman': 'PERMUKIMAN_AR_25K.shp',
        'jalan': 'JALAN_LN_25K.shp',
        'pendidikan': 'PENDIDIKAN_PT_25K.shp',  # Tambahan
        'pemerintahan': 'PEMERINTAHAN_PT_25K.shp',  # Tambahan
        'rumahsakit': 'RUMAHSAKIT_PT_25K.shp'  # Tambahan
    }

    gdfs = {}
    for name, fname in data_sources.items():
        path = os.path.join(FOLDER_PATH, fname)
        if os.path.exists(path):
            gdfs[name] = gpd.read_file(path).to_crs(epsg=METRIC_CRS_EPSG)
        else:
            gdfs[name] = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=f"EPSG:{METRIC_CRS_EPSG}")

    # Calculate enhanced metrics
    gdf_admin['Kepadatan_Penduduk'] = (gdf_admin['JML_PENDUDUK'] / gdf_admin['luas_km2']).fillna(0)

    # Commercial areas analysis
    if not gdfs['niaga'].empty:
        joined_niaga = gpd.sjoin(gdfs['niaga'], gdf_admin[['WADMKC', 'geometry']], how="inner", predicate="within")
        counts_niaga = joined_niaga.groupby('WADMKC').size().rename('Jml_Niaga')
        gdf_admin = gdf_admin.merge(counts_niaga, on='WADMKC', how='left').fillna({'Jml_Niaga': 0})
        gdf_admin['Kepadatan_Niaga'] = gdf_admin['Jml_Niaga'] / gdf_admin['luas_km2']
    else:
        gdf_admin['Jml_Niaga'] = 0
        gdf_admin['Kepadatan_Niaga'] = 0

    # Pendidikan analysis (TAMBAHAN)
    if not gdfs['pendidikan'].empty:
        joined_pendidikan = gpd.sjoin(gdfs['pendidikan'], gdf_admin[['WADMKC', 'geometry']], how="inner", predicate="within")
        counts_pendidikan = joined_pendidikan.groupby('WADMKC').size().rename('Jml_Pendidikan')
        gdf_admin = gdf_admin.merge(counts_pendidikan, on='WADMKC', how='left').fillna({'Jml_Pendidikan': 0})
    else:
        gdf_admin['Jml_Pendidikan'] = 0

    # Pemerintahan analysis (TAMBAHAN)
    if not gdfs['pemerintahan'].empty:
        joined_pemerintahan = gpd.sjoin(gdfs['pemerintahan'], gdf_admin[['WADMKC', 'geometry']], how="inner", predicate="within")
        counts_pemerintahan = joined_pemerintahan.groupby('WADMKC').size().rename('Jml_Pemerintahan')
        gdf_admin = gdf_admin.merge(counts_pemerintahan, on='WADMKC', how='left').fillna({'Jml_Pemerintahan': 0})
    else:
        gdf_admin['Jml_Pemerintahan'] = 0

    # Rumah Sakit analysis (TAMBAHAN)
    if not gdfs['rumahsakit'].empty:
        joined_rumahsakit = gpd.sjoin(gdfs['rumahsakit'], gdf_admin[['WADMKC', 'geometry']], how="inner", predicate="within")
        counts_rumahsakit = joined_rumahsakit.groupby('WADMKC').size().rename('Jml_RumahSakit')
        gdf_admin = gdf_admin.merge(counts_rumahsakit, on='WADMKC', how='left').fillna({'Jml_RumahSakit': 0})
    else:
        gdf_admin['Jml_RumahSakit'] = 0

    # Residential areas analysis
    if not gdfs['pemukiman'].empty:
        overlay_pemukiman = gpd.overlay(gdf_admin[['WADMKC', 'geometry']], gdfs['pemukiman'][['geometry']], how='intersection')
        overlay_pemukiman['area_pemukiman_m2'] = overlay_pemukiman.geometry.area
        area_pemukiman = overlay_pemukiman.groupby('WADMKC')['area_pemukiman_m2'].sum()
        gdf_admin = gdf_admin.merge(area_pemukiman, on='WADMKC', how='left').fillna({'area_pemukiman_m2': 0})
        gdf_admin['Persen_Pemukiman'] = (gdf_admin['area_pemukiman_m2'] / (gdf_admin['luas_km2'] * 1_000_000)) * 100
    else:
        gdf_admin['area_pemukiman_m2'] = 0
        gdf_admin['Persen_Pemukiman'] = 0

    # Road network analysis
    if not gdfs['jalan'].empty:
        jalan_overlay = gpd.overlay(gdfs['jalan'][['geometry']], gdf_admin[['WADMKC', 'geometry']], how='intersection')
        jalan_overlay['panjang_m'] = jalan_overlay.geometry.length
        panjang_jalan = jalan_overlay.groupby('WADMKC')['panjang_m'].sum() / 1000.0
        gdf_admin = gdf_admin.merge(panjang_jalan.rename('Total_Panjang_Jalan_km'), on='WADMKC', how='left')
        gdf_admin['Total_Panjang_Jalan_km'] = gdf_admin['Total_Panjang_Jalan_km'].fillna(0)
        gdf_admin['Kepadatan_Jalan'] = gdf_admin['Total_Panjang_Jalan_km'] / gdf_admin['luas_km2']
    else:
        gdf_admin['Total_Panjang_Jalan_km'] = 0
        gdf_admin['Kepadatan_Jalan'] = 0

    # Load waste bank data
    file_kmz_path = os.path.join(FOLDER_PATH, KMZ_FILENAME)
    gdf_bank_sampah_raw = read_kmz_or_kml(file_kmz_path)
    gdf_bank_sampah = gpd.GeoDataFrame()

    if gdf_bank_sampah_raw is not None and not gdf_bank_sampah_raw.empty:
        gdf_bank_sampah = gdf_bank_sampah_raw.to_crs(epsg=GEOGRAPHIC_CRS_EPSG)
        if gdf_bank_sampah.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).any():
            gdf_metric = gdf_bank_sampah.to_crs(epsg=METRIC_CRS_EPSG)
            gdf_bank_sampah['geometry'] = gdf_metric.geometry.centroid.to_crs(epsg=GEOGRAPHIC_CRS_EPSG)

        gdf_bank_sampah_clean = gdf_bank_sampah[['Name', 'Description', 'geometry']].copy()
        gdf_bank_sampah_clean.rename(columns={'Name': 'nama', 'Description': 'deskripsi'}, inplace=True)
        gdf_bank_sampah_clean['latitude'] = gdf_bank_sampah_clean.geometry.y
        gdf_bank_sampah_clean['longitude'] = gdf_bank_sampah_clean.geometry.x
        gdf_bank_sampah = gdf_bank_sampah_clean

    # Calculate accessibility index
    gdf_admin = calculate_accessibility_index(gdf_admin, gdf_bank_sampah)

    # Calculate waste generation estimates
    gdf_admin = calculate_waste_generation_estimate(gdf_admin)

    # Create enhanced waste potential index - BAGIAN YANG DIGANTI DENGAN PEMBOBOTAN BARU
    faktor_potensi = [
        'Kepadatan_Penduduk', 'Jml_Niaga', 'Persen_Pemukiman',
        'Kepadatan_Jalan', 'Jml_Pendidikan', 'Jml_Pemerintahan', 'Jml_RumahSakit'
    ]

    # Pastikan kolom ada (isi 0 jika tidak ada)
    for f in faktor_potensi:
        if f not in gdf_admin.columns:
            gdf_admin[f] = 0

    data_indeks = gdf_admin[faktor_potensi].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_indeks), columns=faktor_potensi, index=gdf_admin.index)

    # PEMBOBOTAN BARU sesuai kode singkat Anda
    bobot = {
        'Kepadatan_Penduduk': 0.30,
        'Jml_Niaga': 0.25,
        'Persen_Pemukiman': 0.15,
        'Kepadatan_Jalan': 0.10,
        'Jml_RumahSakit': 0.10,
        'Jml_Pendidikan': 0.05,
        'Jml_Pemerintahan': 0.05
    }

    gdf_admin['Indeks_Potensi_Sampah'] = (df_scaled * pd.Series(bobot)).sum(axis=1)
    # Create priority categories
    gdf_admin['Priority_Category'] = pd.cut(
        gdf_admin['Indeks_Potensi_Sampah'],
        bins=5,
        labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    )

    # ===== ENHANCED SPATIAL ANALYSIS =====
    # Spatial autocorrelation analysis (LISA) - FIXED VERSION
    if len(gdf_admin) > 3:  # Need minimum districts for spatial analysis
        try:
            wq = weights.Queen.from_dataframe(gdf_admin)
            wq.transform = 'r'
            y = gdf_admin['Indeks_Potensi_Sampah'].values

            # Global Moran's I
            moran_global = esda.moran.Moran(y, wq)

            # Local Moran's I (LISA)
            lisa = esda.moran.Moran_Local(y, wq)

            # Initialize LISA clusters
            gdf_admin['lisa_cluster_id'] = lisa.q
            gdf_admin['lisa_significant'] = lisa.p_sim < 0.1
            gdf_admin['lisa_cluster'] = 'Not Significant'

            # Assign LISA cluster labels based on significance
            mask_sig = gdf_admin['lisa_significant']

            # High-High: high value surrounded by high values
            gdf_admin.loc[mask_sig & (gdf_admin['lisa_cluster_id'] == 1), 'lisa_cluster'] = 'High-High'

            # Low-Low: low value surrounded by low values
            gdf_admin.loc[mask_sig & (gdf_admin['lisa_cluster_id'] == 2), 'lisa_cluster'] = 'Low-Low'

            # Low-High: low value surrounded by high values
            gdf_admin.loc[mask_sig & (gdf_admin['lisa_cluster_id'] == 3), 'lisa_cluster'] = 'Low-High'

            # High-Low: high value surrounded by low values
            gdf_admin.loc[mask_sig & (gdf_admin['lisa_cluster_id'] == 4), 'lisa_cluster'] = 'High-Low'

            spatial_stats = {
                'moran_i': moran_global.I,
                'p_value': moran_global.p_norm,
                'expected_i': moran_global.EI,
                'variance': moran_global.VI_norm,
                'z_score': moran_global.z_norm,
                'lisa_clusters': gdf_admin['lisa_cluster'].value_counts().to_dict()
            }

            print(f"Global Moran's I: {moran_global.I:.4f}")
            print(f"P-value: {moran_global.p_norm:.4f}")
            print(f"LISA clusters: {spatial_stats['lisa_clusters']}")

        except Exception as e:
            print(f"Error in LISA analysis: {e}")
            gdf_admin['lisa_cluster'] = 'Not Significant'
            gdf_admin['lisa_significant'] = False
            spatial_stats = None
    else:
        gdf_admin['lisa_cluster'] = 'Not Significant'
        gdf_admin['lisa_significant'] = False
        spatial_stats = None

    # Additional Analysis 1: Getis-Ord Gi* (Hot Spot Analysis)
    try:
        if len(gdf_admin) > 3 and spatial_stats is not None:
            from pysal.explore.esda import G_Local

            # Calculate Getis-Ord Gi* statistic
            gi_star = G_Local(y, wq)

            # Classify hot spots and cold spots
            gdf_admin['gi_star_z'] = gi_star.Zs
            gdf_admin['gi_star_p'] = gi_star.p_sim

            # Create hot spot categories
            conditions = [
                (gdf_admin['gi_star_p'] < 0.01) & (gdf_admin['gi_star_z'] > 2.58),   # 99% confidence hot spot
                (gdf_admin['gi_star_p'] < 0.05) & (gdf_admin['gi_star_z'] > 1.96),   # 95% confidence hot spot
                (gdf_admin['gi_star_p'] < 0.1) & (gdf_admin['gi_star_z'] > 1.65),    # 90% confidence hot spot
                (gdf_admin['gi_star_p'] < 0.1) & (gdf_admin['gi_star_z'] < -1.65),   # 90% confidence cold spot
                (gdf_admin['gi_star_p'] < 0.05) & (gdf_admin['gi_star_z'] < -1.96),  # 95% confidence cold spot
                (gdf_admin['gi_star_p'] < 0.01) & (gdf_admin['gi_star_z'] < -2.58),  # 99% confidence cold spot
            ]

            choices = ['Hot Spot (99%)', 'Hot Spot (95%)', 'Hot Spot (90%)',
                      'Cold Spot (90%)', 'Cold Spot (95%)', 'Cold Spot (99%)']

            gdf_admin['hotspot_category'] = np.select(conditions, choices, default='Not Significant')

            print(f"Hot Spot Analysis completed: {gdf_admin['hotspot_category'].value_counts().to_dict()}")

    except Exception as e:
        print(f"Error in Getis-Ord Gi* analysis: {e}")
        gdf_admin['hotspot_category'] = 'Not Significant'
        gdf_admin['gi_star_z'] = 0
        gdf_admin['gi_star_p'] = 1

    # Additional Analysis 2: Geary's C Local Analysis
    try:
        if len(gdf_admin) > 3 and spatial_stats is not None:
            from pysal.explore.esda import Geary_Local

            # Calculate Local Geary's C
            geary_local = Geary_Local(y, wq)

            gdf_admin['geary_c_local'] = geary_local.localG
            gdf_admin['geary_c_p'] = geary_local.p_sim
            gdf_admin['geary_c_significant'] = geary_local.p_sim < 0.1

            # Classify based on Geary's C
            # Low values indicate positive spatial autocorrelation (similar values clustered)
            # High values indicate negative spatial autocorrelation (dissimilar values clustered)
            conditions_geary = [
                (gdf_admin['geary_c_significant']) & (gdf_admin['geary_c_local'] < 0.5),   # Strong positive autocorr
                (gdf_admin['geary_c_significant']) & (gdf_admin['geary_c_local'] < 1.0),   # Moderate positive autocorr
                (gdf_admin['geary_c_significant']) & (gdf_admin['geary_c_local'] > 1.5),   # Strong negative autocorr
                (gdf_admin['geary_c_significant']) & (gdf_admin['geary_c_local'] > 1.0),   # Moderate negative autocorr
            ]

            choices_geary = ['Strong Clustering', 'Moderate Clustering',
                            'Strong Dispersion', 'Moderate Dispersion']

            gdf_admin['geary_cluster'] = np.select(conditions_geary, choices_geary, default='Random')

            # Update spatial_stats with additional metrics
            if spatial_stats:
                spatial_stats.update({
                    'hotspot_distribution': gdf_admin['hotspot_category'].value_counts().to_dict(),
                    'geary_distribution': gdf_admin['geary_cluster'].value_counts().to_dict(),
                    'avg_geary_c': gdf_admin['geary_c_local'].mean()
                })

            print(f"Geary's C Analysis completed: {gdf_admin['geary_cluster'].value_counts().to_dict()}")

    except Exception as e:
        print(f"Error in Local Geary's C analysis: {e}")
        gdf_admin['geary_cluster'] = 'Random'
        gdf_admin['geary_c_local'] = 1.0
        gdf_admin['geary_c_p'] = 1.0

    # Perform cluster analysis
    cluster_features = ['Kepadatan_Penduduk', 'Kepadatan_Niaga', 'Persen_Pemukiman', 'Indeks_Potensi_Sampah']
    gdf_admin, kmeans_model, cluster_scaler = perform_cluster_analysis(gdf_admin, cluster_features)

    return (gdf_admin.to_crs(epsg=GEOGRAPHIC_CRS_EPSG), gdf_bank_sampah, spatial_stats,
            kmeans_model, cluster_scaler, faktor_potensi, bobot)

# ==============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ==============================================================================

def create_advanced_dashboard_charts(gdf_kecamatan: gpd.GeoDataFrame):
    """Create comprehensive dashboard charts."""

    # 1. Correlation Heatmap
    fig_corr = plt.figure(figsize=(10, 8))
    numeric_cols = ['Kepadatan_Penduduk', 'Jml_Niaga', 'Persen_Pemukiman',
                   'Kepadatan_Jalan', 'Indeks_Potensi_Sampah', 'Estimated_Daily_Waste_Ton']
    corr_matrix = gdf_kecamatan[numeric_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Korelasi'})
    plt.title('Matriks Korelasi Faktor-Faktor Potensi Sampah', fontsize=14, fontweight='bold')
    plt.tight_layout()

    st.pyplot(fig_corr)

    # 2. Multi-factor Analysis with Plotly
    fig_scatter = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Kepadatan Penduduk vs Potensi Sampah',
                       'Jumlah Niaga vs Potensi Sampah',
                       'Persentase Pemukiman vs Potensi Sampah',
                       'Estimasi Produksi Sampah (Ton/Hari)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Scatter plots
    fig_scatter.add_trace(
        go.Scatter(x=gdf_kecamatan['Kepadatan_Penduduk'],
                  y=gdf_kecamatan['Indeks_Potensi_Sampah'],
                  mode='markers',
                  text=gdf_kecamatan['WADMKC'],
                  marker=dict(size=10, color=gdf_kecamatan['Estimated_Daily_Waste_Ton'],
                            colorscale='Viridis', showscale=False)),
        row=1, col=1
    )

    fig_scatter.add_trace(
        go.Scatter(x=gdf_kecamatan['Jml_Niaga'],
                  y=gdf_kecamatan['Indeks_Potensi_Sampah'],
                  mode='markers',
                  text=gdf_kecamatan['WADMKC'],
                  marker=dict(size=10, color='orange')),
        row=1, col=2
    )

    fig_scatter.add_trace(
        go.Scatter(x=gdf_kecamatan['Persen_Pemukiman'],
                  y=gdf_kecamatan['Indeks_Potensi_Sampah'],
                  mode='markers',
                  text=gdf_kecamatan['WADMKC'],
                  marker=dict(size=10, color='green')),
        row=2, col=1
    )

    fig_scatter.add_trace(
        go.Bar(x=gdf_kecamatan['WADMKC'],
               y=gdf_kecamatan['Estimated_Daily_Waste_Ton'],
               marker=dict(color=gdf_kecamatan['Indeks_Potensi_Sampah'],
                          colorscale='Reds', showscale=True)),
        row=2, col=2
    )

    fig_scatter.update_layout(height=800, showlegend=False,
                             title_text="Analisis Multi-Faktor Potensi Sampah")
    st.plotly_chart(fig_scatter, use_container_width=True)

def create_interactive_map(gdf_kecamatan: gpd.GeoDataFrame,
                          gdf_bank_sampah: gpd.GeoDataFrame,
                          map_type: str = "lisa") -> folium.Map:
    """Create enhanced interactive map with multiple layer options."""

    center_lat = gdf_kecamatan.geometry.centroid.y.mean()
    center_lon = gdf_kecamatan.geometry.centroid.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None
    )

    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)

    if map_type == "lisa":
        # LISA cluster map
        for idx, row in gdf_kecamatan.iterrows():
            color = LISA_COLORS.get(row['lisa_cluster'], '#gray')

            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(
                    f"""
                    <b>{row['WADMKC']}</b><br>
                    LISA Cluster: {row['lisa_cluster']}<br>
                    Signifikan: {'Ya' if row.get('lisa_significant', False) else 'Tidak'}<br>
                    Indeks Potensi: {row['Indeks_Potensi_Sampah']:.3f}<br>
                    Populasi: {row['JML_PENDUDUK']:,.0f}<br>
                    Est. Sampah: {row['Estimated_Daily_Waste_Ton']:.1f} ton/hari
                    """, max_width=300
                ),
                tooltip=f"{row['WADMKC']} - {row['lisa_cluster']}"
            ).add_to(m)

    elif map_type == "priority":
        # Priority level map
        for idx, row in gdf_kecamatan.iterrows():
            color = PRIORITY_COLORS.get(str(row['Priority_Category']), '#gray')

            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(
                    f"""
                    <b>{row['WADMKC']}</b><br>
                    Prioritas: {row['Priority_Category']}<br>
                    Indeks Potensi: {row['Indeks_Potensi_Sampah']:.3f}<br>
                    Kepadatan Penduduk: {row['Kepadatan_Penduduk']:.0f}/km²<br>
                    Jumlah Niaga: {row['Jml_Niaga']:.0f}<br>
                    Est. Sampah: {row['Estimated_Daily_Waste_Ton']:.1f} ton/hari
                    """, max_width=300
                ),
                tooltip=f"{row['WADMKC']} - {row['Priority_Category']}"
            ).add_to(m)

    elif map_type == "heatmap":
        # Waste generation heatmap
        heat_data = []
        for idx, row in gdf_kecamatan.iterrows():
            centroid = row.geometry.centroid
            heat_data.append([centroid.y, centroid.x, row['Estimated_Daily_Waste_Ton']])

        plugins.HeatMap(heat_data, name='Heat Map').add_to(m)

        # Add district boundaries
        folium.GeoJson(
            gdf_kecamatan.to_json(),
            style_function=lambda feature: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)

    elif map_type == "hotspot":
        # Hot Spot map
        for idx, row in gdf_kecamatan.iterrows():
            color = HOTSPOT_COLORS.get(row.get('hotspot_category', 'Not Significant'), '#gray')

            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(
                    f"""
                    <b>{row['WADMKC']}</b><br>
                    Hot Spot: {row.get('hotspot_category', 'Not Significant')}<br>
                    Gi* Z-Score: {row.get('gi_star_z', 0):.3f}<br>
                    P-Value: {row.get('gi_star_p', 1):.4f}<br>
                    Indeks Potensi: {row['Indeks_Potensi_Sampah']:.3f}
                    """, max_width=300
                ),
                tooltip=f"{row['WADMKC']} - {row.get('hotspot_category', 'Not Significant')}"
            ).add_to(m)

    elif map_type == "geary":
        # Geary's C map
        for idx, row in gdf_kecamatan.iterrows():
            color = GEARY_COLORS.get(row.get('geary_cluster', 'Random'), '#gray')

            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(
                    f"""
                    <b>{row['WADMKC']}</b><br>
                    Geary Pattern: {row.get('geary_cluster', 'Random')}<br>
                    Geary's C: {row.get('geary_c_local', 1):.3f}<br>
                    P-Value: {row.get('geary_c_p', 1):.4f}<br>
                    Indeks Potensi: {row['Indeks_Potensi_Sampah']:.3f}
                    """, max_width=300
                ),
                tooltip=f"{row['WADMKC']} - {row.get('geary_cluster', 'Random')}"
            ).add_to(m)

    # Add waste banks
    if not gdf_bank_sampah.empty:
        bank_cluster = plugins.MarkerCluster(name='Bank Sampah').add_to(m)

        for idx, bank in gdf_bank_sampah.iterrows():
            folium.Marker(
                location=[bank['latitude'], bank['longitude']],
                popup=folium.Popup(
                    f"""
                    <b>{bank['nama']}</b><br>
                    {bank.get('deskripsi', 'Tidak ada deskripsi')}
                    """, max_width=250
                ),
                tooltip=bank['nama'],
                icon=folium.Icon(color='green', icon='recycle', prefix='fa')
            ).add_to(bank_cluster)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

# ==============================================================================
# STREAMLIT PAGES
# ==============================================================================

def show_enhanced_dashboard(gdf_kecamatan: gpd.GeoDataFrame,
                          gdf_bank_sampah: gpd.GeoDataFrame,
                          spatial_stats: Dict):
    """Enhanced dashboard with comprehensive analytics."""

    st.title("Dashboard Analisis Komprehensif")
    st.markdown("---")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    hotspot_count = gdf_kecamatan[gdf_kecamatan['lisa_cluster'] == 'High-High'].shape[0]
    total_waste = gdf_kecamatan['Estimated_Daily_Waste_Ton'].sum()
    avg_index = gdf_kecamatan['Indeks_Potensi_Sampah'].mean()
    high_priority = gdf_kecamatan[gdf_kecamatan['Priority_Category'].isin(['Tinggi', 'Sangat Tinggi'])].shape[0]

    with col1:
        st.metric("Kecamatan Hotspot", f"{hotspot_count}")
    with col2:
        st.metric("Rata-rata Indeks", f"{avg_index:.3f}")
    with col3:
        st.metric("Prioritas Tinggi", f"{high_priority}")
    with col4:
        st.metric("Total Sampah/Hari", f"{total_waste:.1f} ton")

    # Enhanced Spatial Statistics Section
    if spatial_stats:
        st.markdown("### Analisis Autokorelasi Spasial Komprehensif")

        # Global Moran's I Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Moran's I", f"{spatial_stats['moran_i']:.4f}")
        with col2:
            st.metric("P-Value", f"{spatial_stats['p_value']:.4f}")
        with col3:
            st.metric("Z-Score", f"{spatial_stats.get('z_score', 0):.3f}")
        with col4:
            interpretation = "Signifikan" if spatial_stats['p_value'] < 0.1 else "Tidak Signifikan"
            st.metric("Status", interpretation)

        # Interpretation
        if spatial_stats['p_value'] < 0.1:
            if spatial_stats['moran_i'] > 0:
                st.success("Terdapat clustering spasial positif - daerah dengan nilai serupa cenderung bertetangga")
            else:
                st.warning("Terdapat clustering spasial negatif - daerah dengan nilai berbeda cenderung bertetangga")
        else:
            st.info("Tidak ada pola spasial yang signifikan terdeteksi")

        # LISA Distribution
        st.markdown("#### Distribusi Klaster LISA")
        col1, col2 = st.columns(2)

        with col1:
            lisa_counts = gdf_kecamatan['lisa_cluster'].value_counts()
            fig_lisa = px.pie(
                values=lisa_counts.values,
                names=lisa_counts.index,
                title="Distribusi Klaster LISA",
                color_discrete_map=LISA_COLORS
            )
            st.plotly_chart(fig_lisa, use_container_width=True)

        with col2:
            lisa_table = pd.DataFrame({
                'Kluster LISA': lisa_counts.index,
                'Jumlah Kecamatan': lisa_counts.values,
                'Persentase': (lisa_counts.values / lisa_counts.sum() * 100).round(1)
            })
            st.dataframe(lisa_table, use_container_width=True)

        # Additional Spatial Analyses Display
        if 'hotspot_distribution' in spatial_stats:
            st.markdown("#### Analisis Hot Spot (Getis-Ord Gi*)")
            col1, col2 = st.columns(2)

            with col1:
                hotspot_counts = pd.Series(spatial_stats['hotspot_distribution'])
                hotspot_counts = hotspot_counts[hotspot_counts > 0]  # Only show non-zero counts

                if len(hotspot_counts) > 0:
                    fig_hotspot = px.bar(
                        x=hotspot_counts.index,
                        y=hotspot_counts.values,
                        title="Distribusi Hot Spot Analysis",
                        color=hotspot_counts.values,
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig_hotspot.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_hotspot, use_container_width=True)
                else:
                    st.info("Tidak ada hot spot atau cold spot yang signifikan terdeteksi")

            with col2:
                # Show hot spot details
                hotspot_areas = gdf_kecamatan[gdf_kecamatan['hotspot_category'] != 'Not Significant']
                if not hotspot_areas.empty:
                    st.markdown("**Area dengan Hot/Cold Spot Signifikan:**")
                    for _, row in hotspot_areas.iterrows():
                        category = row['hotspot_category']
                        z_score = row.get('gi_star_z', 0)
                        if 'Hot' in category:
                            st.markdown(f"**{row['WADMKC']}**: {category} (Z-score: {z_score:.2f})")
                        elif 'Cold' in category:
                            st.markdown(f"**{row['WADMKC']}**: {category} (Z-score: {z_score:.2f})")
                else:
                    st.info("Tidak ada area dengan hot/cold spot yang signifikan")

        if 'geary_distribution' in spatial_stats:
            st.markdown("#### Analisis Geary's C Lokal")
            col1, col2 = st.columns(2)

            with col1:
                geary_counts = pd.Series(spatial_stats['geary_distribution'])
                geary_counts = geary_counts[geary_counts > 0]

                if len(geary_counts) > 0:
                    fig_geary = px.bar(
                        x=geary_counts.index,
                        y=geary_counts.values,
                        title="Distribusi Geary's C Analysis",
                        color=geary_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    fig_geary.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_geary, use_container_width=True)
                else:
                    st.info("Semua area memiliki pola spasial random berdasarkan Geary's C")

            with col2:
                avg_geary = spatial_stats.get('avg_geary_c', 1.0)
                st.metric("Rata-rata Geary's C", f"{avg_geary:.3f}")

                if avg_geary < 1.0:
                    st.success("Indikasi clustering positif (nilai serupa bertetangga)")
                elif avg_geary > 1.0:
                    st.warning("Indikasi clustering negatif (nilai berbeda bertetangga)")
                else:
                    st.info("Pola spasial random")

                # Show clustering areas
                cluster_areas = gdf_kecamatan[gdf_kecamatan['geary_cluster'] != 'Random']
                if not cluster_areas.empty:
                    st.markdown("**Area dengan Clustering Signifikan:**")
                    for _, row in cluster_areas.iterrows():
                        pattern = row['geary_cluster']
                        if 'Clustering' in pattern:
                            st.markdown(f"**{row['WADMKC']}**: {pattern}")
                        elif 'Dispersion' in pattern:
                            st.markdown(f"**{row['WADMKC']}**: {pattern}")

    st.markdown("---")

    # Interactive Map Section
    st.subheader("Peta Interaktif Multi-Layer")

    map_option = st.selectbox(
        "Pilih jenis peta:",
        ["lisa", "priority", "heatmap", "hotspot", "geary"],
        format_func=lambda x: {
            "lisa": "Analisis LISA (Klaster Spasial)",
            "priority": "Tingkat Prioritas",
            "heatmap": "Heatmap Produksi Sampah",
            "hotspot": "Hot Spot Analysis (Gi*)",
            "geary": "Geary's C Clustering"
        }[x]
    )

    interactive_map = create_interactive_map(gdf_kecamatan, gdf_bank_sampah, map_option)
    st_folium(interactive_map, width='100%', height=600)

    st.markdown("---")

    # Charts Section
    st.subheader("Analisis Visual Komprehensif")

    tab1, tab2, tab3 = st.tabs(["Analisis Korelasi", "Distribusi & Ranking", "Analisis Kluster"])

    with tab1:
        st.markdown("#### Matriks Korelasi & Scatter Plot Multi-Faktor")
        create_advanced_dashboard_charts(gdf_kecamatan)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Priority distribution pie chart
            priority_counts = gdf_kecamatan['Priority_Category'].value_counts()
            fig_pie = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Distribusi Tingkat Prioritas Kecamatan",
                color_discrete_map=PRIORITY_COLORS
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Top waste producers
            top_waste = gdf_kecamatan.nlargest(10, 'Estimated_Daily_Waste_Ton')
            fig_bar = px.bar(
                top_waste,
                x='WADMKC',
                y='Estimated_Daily_Waste_Ton',
                color='Indeks_Potensi_Sampah',
                title="Top 10 Penghasil Sampah (Ton/Hari)",
                color_continuous_scale='Reds'
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Ranking table
        st.markdown("#### Peringkat Kecamatan Berdasarkan Indeks Potensi")
        ranking_df = gdf_kecamatan[['WADMKC', 'Indeks_Potensi_Sampah', 'Priority_Category',
                                  'Estimated_Daily_Waste_Ton', 'lisa_cluster']].copy()
        ranking_df = ranking_df.sort_values('Indeks_Potensi_Sampah', ascending=False).reset_index(drop=True)
        ranking_df.index += 1
        st.dataframe(ranking_df, use_container_width=True)

    with tab3:
        st.markdown("#### Analisis Kluster K-Means")

        # Cluster distribution
        cluster_counts = gdf_kecamatan['Cluster_Label'].value_counts()
        fig_cluster = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Distribusi Kluster Kecamatan"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Cluster characteristics
        st.markdown("#### Karakteristik Setiap Kluster")
        cluster_summary = gdf_kecamatan.groupby('Cluster_Label')[
            ['Kepadatan_Penduduk', 'Jml_Niaga', 'Persen_Pemukiman', 'Indeks_Potensi_Sampah']
        ].mean().round(3)
        st.dataframe(cluster_summary, use_container_width=True)

def show_waste_bank_finder(gdf_bank_sampah: gpd.GeoDataFrame, gdf_kecamatan: gpd.GeoDataFrame):
    """Enhanced waste bank finder with routing and analysis."""

    st.title("Bank Sampah Finder Plus")
    st.markdown("Temukan bank sampah terdekat dengan fitur pencarian dan analisis aksesibilitas")
    st.markdown("---")

    if gdf_bank_sampah.empty:
        st.warning("Data bank sampah tidak tersedia")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Panel Pencarian")

        # Search by district
        selected_district = st.selectbox(
            "Pilih Kecamatan:",
            ["Semua"] + list(gdf_kecamatan['WADMKC'].sort_values())
        )

        # Search by name
        search_term = st.text_input("Cari nama bank sampah:", "")

        # Filter options
        show_nearest = st.checkbox("Tampilkan hanya 5 terdekat", value=False)

        # Apply filters
        filtered_banks = gdf_bank_sampah.copy()

        if search_term:
            filtered_banks = filtered_banks[
                filtered_banks['nama'].str.contains(search_term, case=False, na=False)
            ]

        st.markdown(f"**Ditemukan: {len(filtered_banks)} bank sampah**")

        # Bank list
        if not filtered_banks.empty:
            for idx, bank in filtered_banks.iterrows():
                with st.expander(f"🏦 {bank['nama']}", expanded=False):
                    st.write(f"**Deskripsi:** {bank.get('deskripsi', 'N/A')}")
                    st.write(f"**Koordinat:** {bank['latitude']:.6f}, {bank['longitude']:.6f}")

                    if st.button(f"📍 Fokus ke {bank['nama']}", key=f"focus_{idx}"):
                        st.session_state['map_center'] = [bank['latitude'], bank['longitude']]

    with col2:
        st.subheader("🗺️ Peta Bank Sampah")

        # Create map
        center = st.session_state.get('map_center', [-7.7956, 110.3695])
        m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

        # Add district boundaries
        folium.GeoJson(
            gdf_kecamatan.to_json(),
            style_function=lambda feature: {
                'fillColor': 'lightblue',
                'color': 'navy',
                'weight': 2,
                'fillOpacity': 0.1
            },
            popup=folium.GeoJsonPopup(fields=['WADMKC'])
        ).add_to(m)

        # Add waste banks
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
                 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

        for idx, bank in filtered_banks.iterrows():
            color = colors[idx % len(colors)]

            folium.Marker(
                location=[bank['latitude'], bank['longitude']],
                popup=folium.Popup(
                    f"""
                    <div style="width: 300px">
                        <h4>{bank['nama']}</h4>
                        <p><strong>Deskripsi:</strong><br>{bank.get('deskripsi', 'Tidak ada deskripsi')}</p>
                        <p><strong>Koordinat:</strong><br>{bank['latitude']:.6f}, {bank['longitude']:.6f}</p>
                    </div>
                    """, max_width=350
                ),
                tooltip=f"🏦 {bank['nama']}",
                icon=folium.Icon(color=color, icon='university', prefix='fa')
            ).add_to(m)

        st_folium(m, width='100%', height=500)

    st.markdown("---")

    # Accessibility Analysis
    st.subheader("📊 Analisis Aksesibilitas Bank Sampah")

    col1, col2 = st.columns(2)

    with col1:
        # Accessibility by district
        accessibility_df = gdf_kecamatan[['WADMKC', 'Accessibility_Index']].copy()
        accessibility_df = accessibility_df.sort_values('Accessibility_Index', ascending=False)

        fig_acc = px.bar(
            accessibility_df,
            x='WADMKC',
            y='Accessibility_Index',
            title="Indeks Aksesibilitas Bank Sampah per Kecamatan",
            color='Accessibility_Index',
            color_continuous_scale='Greens'
        )
        fig_acc.update_xaxes(tickangle=45)
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        # Gap analysis
        st.markdown("#### 🎯 Analisis Gap Layanan")

        low_access = gdf_kecamatan[gdf_kecamatan['Accessibility_Index'] < 0.5]
        high_potential = gdf_kecamatan[gdf_kecamatan['Indeks_Potensi_Sampah'] > 0.5]

        gap_areas = pd.merge(low_access[['WADMKC', 'Accessibility_Index']],
                            high_potential[['WADMKC', 'Indeks_Potensi_Sampah']],
                            on='WADMKC', how='inner')

        if not gap_areas.empty:
            st.warning(f"⚠️ {len(gap_areas)} kecamatan memiliki potensi tinggi namun aksesibilitas rendah:")
            for _, area in gap_areas.iterrows():
                st.write(f"• **{area['WADMKC']}** - Potensi: {area['Indeks_Potensi_Sampah']:.3f}, Aksesibilitas: {area['Accessibility_Index']:.3f}")
        else:
            st.success("✅ Tidak ada gap layanan yang signifikan terdeteksi")

def show_prediction_analysis(gdf_kecamatan: gpd.GeoDataFrame):
    """Advanced prediction and scenario analysis."""

    st.title("🔮 Analisis Prediksi & Skenario")
    st.markdown("Model prediksi dan analisis skenario untuk perencanaan strategis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Proyeksi Sampah", "🎛️ Analisis Skenario", "🎯 Rekomendasi"])

    with tab1:
        st.subheader("📊 Proyeksi Produksi Sampah 5 Tahun ke Depan")

        # Growth rate input
        col1, col2 = st.columns(2)
        with col1:
            pop_growth = st.slider("Tingkat Pertumbuhan Penduduk (%/tahun)", 0.0, 5.0, 1.5, 0.1)
        with col2:
            commercial_growth = st.slider("Tingkat Pertumbuhan Komersial (%/tahun)", 0.0, 10.0, 3.0, 0.1)

        # Calculate projections
        years = range(2025, 2031)
        projection_data = []

        for year in years:
            year_offset = year - 2025
            pop_multiplier = (1 + pop_growth/100) ** year_offset
            comm_multiplier = (1 + commercial_growth/100) ** year_offset

            for _, row in gdf_kecamatan.iterrows():
                projected_waste = (row['Estimated_Daily_Waste_Ton'] * pop_multiplier *
                                 (1 + (row['Jml_Niaga'] / 1000) * (comm_multiplier - 1)))

                projection_data.append({
                    'Year': year,
                    'Kecamatan': row['WADMKC'],
                    'Projected_Waste_Ton': projected_waste,
                    'Current_Waste_Ton': row['Estimated_Daily_Waste_Ton']
                })

        projection_df = pd.DataFrame(projection_data)

        # Total projection chart
        total_by_year = projection_df.groupby('Year')['Projected_Waste_Ton'].sum()

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=total_by_year.index,
            y=total_by_year.values,
            mode='lines+markers',
            name='Proyeksi Total Sampah',
            line=dict(color='red', width=3)
        ))

        fig_proj.update_layout(
            title="Proyeksi Total Produksi Sampah Kota Yogyakarta",
            xaxis_title="Tahun",
            yaxis_title="Produksi Sampah (Ton/Hari)",
            template="plotly_white"
        )

        st.plotly_chart(fig_proj, use_container_width=True)

        # Top contributors in 2030
        proj_2030 = projection_df[projection_df['Year'] == 2030].nlargest(10, 'Projected_Waste_Ton')

        fig_top = px.bar(
            proj_2030,
            x='Kecamatan',
            y='Projected_Waste_Ton',
            title="Top 10 Penghasil Sampah Proyeksi 2030",
            color='Projected_Waste_Ton',
            color_continuous_scale='Reds'
        )
        fig_top.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top, use_container_width=True)

    with tab2:
        st.subheader("🎛️ Analisis Skenario Kebijakan")

        scenario = st.selectbox(
            "Pilih Skenario:",
            ["baseline", "waste_reduction", "bank_expansion", "comprehensive"],
            format_func=lambda x: {
                "baseline": "📊 Baseline - Kondisi Saat Ini",
                "waste_reduction": "♻️ Pengurangan Sampah 20%",
                "bank_expansion": "🏦 Ekspansi Bank Sampah",
                "comprehensive": "🎯 Strategi Komprehensif"
            }[x]
        )

        scenario_results = gdf_kecamatan.copy()

        if scenario == "waste_reduction":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton'] * 0.8
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.9
            impact_desc = "Pengurangan sampah 20% melalui program 3R intensif"

        elif scenario == "bank_expansion":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton']
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.95
            scenario_results['Accessibility_Index'] = np.minimum(scenario_results['Accessibility_Index'] * 2, 1.0)
            impact_desc = "Penambahan bank sampah meningkatkan aksesibilitas 2x lipat"

        elif scenario == "comprehensive":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton'] * 0.7
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.8
            scenario_results['Accessibility_Index'] = np.minimum(scenario_results['Accessibility_Index'] * 2.5, 1.0)
            impact_desc = "Kombinasi pengurangan sampah 30% + ekspansi bank sampah"

        else:  # baseline
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton']
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah']
            impact_desc = "Kondisi saat ini tanpa intervensi"

        st.info(f"**Deskripsi Skenario:** {impact_desc}")

        # Comparison metrics
        col1, col2, col3 = st.columns(3)

        current_total = gdf_kecamatan['Estimated_Daily_Waste_Ton'].sum()
        scenario_total = scenario_results['Scenario_Waste'].sum()
        reduction = ((current_total - scenario_total) / current_total) * 100

        with col1:
            st.metric("Total Sampah Saat Ini", f"{current_total:.1f} ton/hari")
        with col2:
            st.metric("Total Sampah Skenario", f"{scenario_total:.1f} ton/hari")
        with col3:
            st.metric("Pengurangan", f"{reduction:.1f}%", f"{current_total - scenario_total:.1f} ton/hari")

        # Before/after comparison
        comparison_data = []
        for _, row in scenario_results.iterrows():
            comparison_data.extend([
                {'Kecamatan': row['WADMKC'], 'Type': 'Saat Ini', 'Value': row['Estimated_Daily_Waste_Ton']},
                {'Kecamatan': row['WADMKC'], 'Type': 'Skenario', 'Value': row['Scenario_Waste']}
            ])

        comparison_df = pd.DataFrame(comparison_data)

        fig_comparison = px.bar(
            comparison_df,
            x='Kecamatan',
            y='Value',
            color='Type',
            barmode='group',
            title=f"Perbandingan Produksi Sampah: Saat Ini vs {scenario.replace('_', ' ').title()}",
            color_discrete_map={'Saat Ini': 'red', 'Skenario': 'green'}
        )
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
    with tab2:
        st.subheader("🎛️ Analisis Skenario Kebijakan")

        scenario = st.selectbox(
            "Pilih Skenario:",
            ["baseline", "waste_reduction", "bank_expansion", "comprehensive"],
            format_func=lambda x: {
                "baseline": "📊 Baseline - Kondisi Saat Ini",
                "waste_reduction": "♻️ Pengurangan Sampah 20%",
                "bank_expansion": "🏦 Ekspansi Bank Sampah",
                "comprehensive": "🎯 Strategi Komprehensif"
            }[x]
        )

        scenario_results = gdf_kecamatan.copy()

        if scenario == "waste_reduction":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton'] * 0.8
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.9
            impact_desc = "Pengurangan sampah 20% melalui program 3R intensif"

        elif scenario == "bank_expansion":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton']
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.95
            scenario_results['Accessibility_Index'] = np.minimum(scenario_results['Accessibility_Index'] * 2, 1.0)
            impact_desc = "Penambahan bank sampah meningkatkan aksesibilitas 2x lipat"

        elif scenario == "comprehensive":
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton'] * 0.7
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah'] * 0.8
            scenario_results['Accessibility_Index'] = np.minimum(scenario_results['Accessibility_Index'] * 2.5, 1.0)
            impact_desc = "Kombinasi pengurangan sampah 30% + ekspansi bank sampah"

        else:  # baseline
            scenario_results['Scenario_Waste'] = scenario_results['Estimated_Daily_Waste_Ton']
            scenario_results['Scenario_Index'] = scenario_results['Indeks_Potensi_Sampah']
            impact_desc = "Kondisi saat ini tanpa intervensi"

        st.info(f"**Deskripsi Skenario:** {impact_desc}")

        # Comparison metrics
        col1, col2, col3 = st.columns(3)

        current_total = gdf_kecamatan['Estimated_Daily_Waste_Ton'].sum()
        scenario_total = scenario_results['Scenario_Waste'].sum()
        reduction = ((current_total - scenario_total) / current_total) * 100

        with col1:
            st.metric("Total Sampah Saat Ini", f"{current_total:.1f} ton/hari")
        with col2:
            st.metric("Total Sampah Skenario", f"{scenario_total:.1f} ton/hari")
        with col3:
            st.metric("Pengurangan", f"{reduction:.1f}%", f"{current_total - scenario_total:.1f} ton/hari")

        # Before/after comparison
        comparison_data = []
        for _, row in scenario_results.iterrows():
            comparison_data.extend([
                {'Kecamatan': row['WADMKC'], 'Type': 'Saat Ini', 'Value': row['Estimated_Daily_Waste_Ton']},
                {'Kecamatan': row['WADMKC'], 'Type': 'Skenario', 'Value': row['Scenario_Waste']}
            ])

        comparison_df = pd.DataFrame(comparison_data)

        fig_comparison = px.bar(
            comparison_df,
            x='Kecamatan',
            y='Value',
            color='Type',
            barmode='group',
            title=f"Perbandingan Produksi Sampah: Saat Ini vs {scenario.replace('_', ' ').title()}",
            color_discrete_map={'Saat Ini': 'red', 'Skenario': 'green'}
        )
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)

    with tab3:
        st.subheader("🎯 Rekomendasi Strategis")

        # Priority recommendations based on analysis
        high_priority = gdf_kecamatan[gdf_kecamatan['Priority_Category'].isin(['Tinggi', 'Sangat Tinggi'])]
        low_access = gdf_kecamatan[gdf_kecamatan['Accessibility_Index'] < 0.3]
        hotspots = gdf_kecamatan[gdf_kecamatan['lisa_cluster'] == 'High-High']

        st.markdown("#### 🚨 Rekomendasi Prioritas Tinggi")

        if not hotspots.empty:
            st.error("**Intervensi Segera - Hotspot Spasial:**")
            for _, area in hotspots.iterrows():
                st.write(f"• **{area['WADMKC']}** - Fokus pengurangan sampah dan peningkatan pengelolaan")

        if not low_access.empty:
            st.warning("**Peningkatan Aksesibilitas - Area dengan Akses Rendah:**")
            for _, area in low_access.iterrows():
                st.write(f"• **{area['WADMKC']}** - Pertimbangkan penambahan bank sampah atau fasilitas pengumpulan")

        st.markdown("#### 💡 Strategi Jangka Menengah")

        recommendations = [
            "🏗️ **Infrastruktur:** Bangun 3-5 bank sampah baru di area dengan aksesibilitas rendah",
            "📚 **Edukasi:** Program sosialisasi 3R intensif di kecamatan prioritas tinggi",
            "🤝 **Kemitraan:** Libatkan sektor swasta dalam pengelolaan sampah komersial",
            "📱 **Teknologi:** Implementasi sistem monitoring real-time produksi sampah",
            "🎯 **Targeting:** Fokus program pengurangan sampah di cluster High-High"
        ]

        for rec in recommendations:
            st.markdown(rec)

        st.markdown("#### 📊 Indikator Keberhasilan")

        indicators = pd.DataFrame({
            'Indikator': [
                'Pengurangan Total Sampah',
                'Peningkatan Partisipasi Bank Sampah',
                'Pemerataan Aksesibilitas',
                'Efisiensi Pengumpulan',
                'Tingkat Daur Ulang'
            ],
            'Target 1 Tahun': ['10%', '25%', '15%', '20%', '15%'],
            'Target 3 Tahun': ['25%', '50%', '40%', '35%', '30%'],
            'Metrik': [
                'Ton/hari',
                'Jumlah peserta aktif',
                'Indeks aksesibilitas rata-rata',
                'Waktu pengumpulan rata-rata',
                'Persentase sampah terdaur ulang'
            ]
        })

        st.dataframe(indicators, use_container_width=True)

def show_data_export_tools(gdf_kecamatan: gpd.GeoDataFrame, gdf_bank_sampah: gpd.GeoDataFrame):
    """Data export and reporting tools."""

    st.title("📄 Tools Export & Pelaporan")
    st.markdown("Ekspor data dan buat laporan untuk keperluan presentasi dan dokumentasi")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Export Data", "📋 Generator Laporan"])

    with tab1:
        st.subheader("📁 Export Data Analisis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Kecamatan (Excel/CSV)**")

            export_columns = st.multiselect(
                "Pilih kolom untuk export:",
                gdf_kecamatan.columns.tolist(),
                default=['WADMKC', 'Indeks_Potensi_Sampah', 'Priority_Category',
                        'Estimated_Daily_Waste_Ton', 'lisa_cluster']
            )

            if st.button("📥 Download Excel"):
                import io
                output = io.BytesIO()
                gdf_kecamatan[export_columns].to_excel(output, index=False, engine='openpyxl')  # ✅ Benar
                output.seek(0)
                st.download_button(
                    label="💾 Download Data Kecamatan.xlsx",
                    data=output,
                    file_name=f"analisis_kecamatan_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

            if st.button("📥 Download CSV"):
                csv_data = gdf_kecamatan[export_columns].to_csv(index=False)
                st.download_button(
                    label="💾 Download Data Kecamatan.csv",
                    data=csv_data,
                    file_name=f"analisis_kecamatan_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with col2:
            st.markdown("**Data Bank Sampah (Excel/CSV)**")

            if not gdf_bank_sampah.empty:
                if st.button("📥 Download Bank Sampah Excel"):
                    import io # Pastikan io diimpor jika belum
                    bank_output = io.BytesIO() # 1. Buat buffer
                    gdf_bank_sampah.to_excel(bank_output, index=False, engine='openpyxl')
                    bank_output.seek(0) 

                    st.download_button(
                        label="💾 Download Data Bank Sampah.xlsx",
                        data=bank_output,
                        file_name=f"bank_sampah_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )

                if st.button("📥 Download Bank Sampah CSV"):
                    bank_csv = gdf_bank_sampah.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Data Bank Sampah.csv",
                        data=bank_csv,
                        file_name=f"bank_sampah_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Data bank sampah tidak tersedia")

        st.markdown("---")
        st.subheader("🗺️ Export Peta & Visualisasi")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗺️ Generate Static Map"):
                fig, ax = plt.subplots(figsize=(12, 10))

                gdf_kecamatan['color'] = gdf_kecamatan['lisa_cluster'].map(LISA_COLORS)
                gdf_kecamatan.plot(color=gdf_kecamatan['color'], ax=ax, edgecolor='black', linewidth=0.7)

                # Add labels
                gdf_kecamatan.apply(
                    lambda x: ax.annotate(
                        text=x['WADMKC'],
                        xy=x.geometry.centroid.coords[0],
                        ha='center', fontsize=8,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')]
                    ), axis=1
                )

                ax.set_title("Peta Klaster LISA - Potensi Timbulan Sampah", fontsize=16, fontweight='bold')
                ax.set_axis_off()

                plt.tight_layout()

                # Save to bytes
                import io
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)

                st.download_button(
                    label="💾 Download Peta (PNG)",
                    data=img_buffer.getvalue(),
                    file_name=f"peta_lisa_{datetime.datetime.now().strftime('%Y%m%d')}.png",
                    mime="image/png"
                )

                plt.close()

    with tab2:
        st.subheader("📋 Generator Laporan Otomatis")

        # Report configuration
        col1, col2 = st.columns(2)

        with col1:
            report_type = st.selectbox(
                "Jenis Laporan:",
                ["executive_summary", "technical_detail", "policy_brief"],
                format_func=lambda x: {
                    "executive_summary": "📊 Executive Summary",
                    "technical_detail": "🔬 Technical Detail Report",
                    "policy_brief": "📋 Policy Brief"
                }[x]
            )

        with col2:
            include_maps = st.checkbox("Sertakan Peta", value=True)
            include_charts = st.checkbox("Sertakan Grafik", value=True)
            include_recommendations = st.checkbox("Sertakan Rekomendasi", value=True)

        if st.button("📝 Generate Laporan"):
            # Create report content
            report_content = generate_report(
                gdf_kecamatan, gdf_bank_sampah,
                report_type, include_maps, include_charts, include_recommendations
            )

            st.markdown("### 📄 Preview Laporan")
            st.markdown(report_content)

            # Download button for report
            st.download_button(
                label="💾 Download Laporan (Markdown)",
                data=report_content,
                file_name=f"laporan_{report_type}_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

def generate_report(gdf_kecamatan: gpd.GeoDataFrame,
                   gdf_bank_sampah: gpd.GeoDataFrame,
                   report_type: str,
                   include_maps: bool,
                   include_charts: bool,
                   include_recommendations: bool) -> str:
    """Generate automated report based on analysis."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    if report_type == "executive_summary":
        report = f"""
# LAPORAN EXECUTIVE SUMMARY
## Analisis Potensi Timbulan Sampah Kota Yogyakarta

**Tanggal:** {current_date}
**Sistem:** SIGAP (Sistem Informasi Geografis Analisis Potensi Sampah)

---

### 🎯 RINGKASAN EKSEKUTIF

#### Temuan Utama:
- **Total Kecamatan Dianalisis:** {len(gdf_kecamatan)} kecamatan
- **Estimasi Total Sampah Harian:** {gdf_kecamatan['Estimated_Daily_Waste_Ton'].sum():.1f} ton/hari
- **Kecamatan Prioritas Tinggi:** {len(gdf_kecamatan[gdf_kecamatan['Priority_Category'].isin(['Tinggi', 'Sangat Tinggi'])])} kecamatan
- **Hotspot Spasial (High-High):** {len(gdf_kecamatan[gdf_kecamatan['lisa_cluster'] == 'High-High'])} kecamatan

#### Kecamatan dengan Potensi Tertinggi:
"""

        top_5 = gdf_kecamatan.nlargest(5, 'Indeks_Potensi_Sampah')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            report += f"{i}. **{row['WADMKC']}** - Indeks: {row['Indeks_Potensi_Sampah']:.3f}\n"

        if include_recommendations:
            report += """
#### 🎯 Rekomendasi Kunci:
1. **Prioritas Segera:** Fokus intervensi pada kecamatan hotspot
2. **Infrastruktur:** Tambah bank sampah di area akses rendah
3. **Program 3R:** Intensifikasi di kecamatan prioritas tinggi
4. **Monitoring:** Implementasi sistem pemantauan real-time
"""

    return report

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="SIGAP Sampah Jogja Advanced",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2d5a87 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f4e79;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px 8px 0px 0px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏙️ SIGAP Sampah Jogja - Advanced Analytics</h1>
        <p>Sistem Informasi Geografis Analisis Potensi Sampah Kota Yogyakarta</p>
        <p><em>Powered by Advanced Geospatial Analytics & Machine Learning</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(180deg, #1f4e79 0%, #2d5a87 100%);
                padding: 1rem; border-radius: 10px; color: white; text-align: center;">
        <h3>🎯 SIGAP Analytics</h3>
        <p>Advanced Geospatial Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Navigation
    page_options = {
        "🎯 Dashboard Analitik": "dashboard",
        "🗺️ Peta Interaktif": "maps",
        "🏦 Bank Sampah Finder": "bank_finder",
        "🔮 Prediksi & Skenario": "prediction",
        "📄 Export & Laporan": "export"
    }

    selected_page = st.sidebar.selectbox(
        "🧭 Navigasi Halaman:",
        list(page_options.keys())
    )

    # Advanced settings
    st.sidebar.markdown("### ⚙️ Pengaturan Lanjutan")

    analysis_mode = st.sidebar.selectbox(
        "Mode Analisis:",
        ["Standard", "Advanced", "Research"],
        help="Standard: Analisis dasar, Advanced: Dengan prediksi, Research: Semua fitur"
    )

    cache_data = st.sidebar.checkbox("Gunakan Cache Data", value=True,
                                   help="Mempercepat loading dengan menyimpan hasil analisis")

    st.sidebar.markdown("---")

    # System info
    st.sidebar.markdown("### ℹ️ Info Sistem")
    st.sidebar.info(f"""
    **Versi:** 2.0 Advanced
    **Mode:** {analysis_mode}
    **Cache:** {'Aktif' if cache_data else 'Nonaktif'}
    **Data Update:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
    """)

    # Load data
    try:
        with st.spinner("🔄 Memuat dan menganalisis data spasial..."):
            if cache_data:
                data_result = run_comprehensive_analysis()
            else:
                st.cache_data.clear()
                data_result = run_comprehensive_analysis()

        (gdf_kecamatan, gdf_bank_sampah, spatial_stats,
         kmeans_model, cluster_scaler, faktor_potensi, bobot) = data_result

        # Store in session state
        st.session_state['gdf_kecamatan'] = gdf_kecamatan
        st.session_state['gdf_bank_sampah'] = gdf_bank_sampah
        st.session_state['spatial_stats'] = spatial_stats

        # Page routing
        page_key = page_options[selected_page]

        if page_key == "dashboard":
            show_enhanced_dashboard(gdf_kecamatan, gdf_bank_sampah, spatial_stats)
        elif page_key == "maps":
            st.title("🗺️ Peta Interaktif Multi-Layer")

            col1, col2 = st.columns([1, 3])
            with col1:
                map_type = st.selectbox(
                    "Pilih Jenis Peta:",
                    ["lisa", "priority", "heatmap", "hotspot", "geary"],
                    format_func=lambda x: {
                        "lisa": "🔍 Analisis LISA",
                        "priority": "🎯 Tingkat Prioritas",
                        "heatmap": "🌡️ Heatmap Sampah",
                        "hotspot": "🌡️ Hot Spot Analysis",
                        "geary": "🔄 Geary's C"
                    }[x]
                )

                show_legend = st.checkbox("Tampilkan Legenda", value=True)

                if map_type == "lisa":
                    st.markdown("**Keterangan LISA:**")
                    st.markdown("- 🔴 **High-High:** Hotspot")
                    st.markdown("- 🔵 **Low-Low:** Coldspot")
                    st.markdown("- 🟡 **Low-High:** Outlier Rendah")
                    st.markdown("- 🟦 **High-Low:** Outlier Tinggi")
                elif map_type == "priority":
                    st.markdown("**Tingkat Prioritas:**")
                    st.markdown("- 🟥 **Sangat Tinggi**")
                    st.markdown("- 🟧 **Tinggi**")
                    st.markdown("- 🟨 **Sedang**")
                    st.markdown("- 🟩 **Rendah**")
                    st.markdown("- 🟦 **Sangat Rendah**")

            with col2:
                interactive_map = create_interactive_map(gdf_kecamatan, gdf_bank_sampah, map_type)
                st_folium(interactive_map, width='100%', height=600)

        elif page_key == "bank_finder":
            show_waste_bank_finder(gdf_bank_sampah, gdf_kecamatan)
        elif page_key == "prediction":
            show_prediction_analysis(gdf_kecamatan)
        elif page_key == "export":
            show_data_export_tools(gdf_kecamatan, gdf_bank_sampah)

    except FileNotFoundError as e:
        st.error(f"❌ File data tidak ditemukan: {e}")
        st.info("💡 Pastikan folder '/content/proyek_bank_sampah/data/' berisi semua file yang diperlukan")
    except Exception as e:
        st.error(f"❌ Error dalam analisis: {e}")
        st.info("🔧 Coba restart aplikasi atau periksa integritas data")

# ==============================================================================
# NGROK DEPLOYMENT FUNCTIONS
# ==============================================================================



# ==============================================================================
# ENTRY POINT
# ==============================================================================

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Check if data folder exists
    if not os.path.exists(FOLDER_PATH):
        st.error(f"""
        ❌ **Folder data tidak ditemukan!**

        Pastikan folder `{FOLDER_PATH}` ada di repositori dan berisi semua file yang diperlukan.
        """)
    else:

        main()




