"""
═══════════════════════════════════════════════════════════════════════════════
STEP 3: WIDE FEATURES EXTRACTION - PTB-XL Wide+Deep Pipeline
═══════════════════════════════════════════════════════════════════════════════
Extraction features cliniques avec NeuroKit2 (Wide Branch - Clinical):
  - Lead II par défaut (dérivation standard)
  - R-peaks detection
  - Heart Rate & HRV (time domain)
  - Intervalles P-QRS-T (si détection OK)
  - Entropies simples
  - Qualité signal (RPeaks_ok, NaN_ratio)
  
+ Wide Branch - Metadata (CSV):
  - age, sex, height, weight
  - device, site, nurse
  - baseline_drift, static_noise, extra_beats, etc.
  
Sortie: W_features.csv (~27 clinical + ~15 metadata = ~42 features)
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("STEP 3: WIDE FEATURES EXTRACTION (Clinical + Metadata)")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DATASET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Chargement dataset...")

df = pd.read_csv('ptbxl_with_cleaned_signals.csv', index_col='ecg_id')
print(f"  ✓ {len(df):,} enregistrements chargés")
print(f"  ✓ Signaux disponibles: {df['X_clean_available'].sum():,}")

# Filtrer seulement ECG avec signaux nettoyés
df = df[df['X_clean_available'] == True].copy()
print(f"  ✓ {len(df):,} ECG à traiter")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACTION FEATURES CLINIQUES (NeuroKit2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Extraction features cliniques (Lead II)...")

def extract_clinical_features(X_clean, sampling_rate=100):
    """
    Extrait ~27 features cliniques depuis Lead II avec NeuroKit2
    
    Args:
        X_clean: array (12, 1000)
        sampling_rate: 100 Hz
    
    Returns:
        dict avec features cliniques
    """
    features = {}
    
    # Lead II (index 1)
    lead_ii = X_clean[1, :]
    
    # ─────────────────────────────────────────────────────────────────────────
    # A. R-PEAKS DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Détection R-peaks (méthode neurokit)
        _, rpeaks_dict = nk.ecg_peaks(lead_ii, sampling_rate=sampling_rate, method='neurokit')
        rpeaks = rpeaks_dict['ECG_R_Peaks']
        
        features['n_rpeaks'] = len(rpeaks)
        features['rpeaks_ok'] = 1 if len(rpeaks) >= 5 else 0  # au moins 5 battements
        
        # ─────────────────────────────────────────────────────────────────────
        # B. HEART RATE (HR)
        # ─────────────────────────────────────────────────────────────────────
        if len(rpeaks) >= 2:
            # HR moyen
            rr_intervals = np.diff(rpeaks) / sampling_rate  # en secondes
            hr_values = 60.0 / rr_intervals  # bpm
            
            features['hr_mean'] = np.mean(hr_values)
            features['hr_std'] = np.std(hr_values)
            features['hr_min'] = np.min(hr_values)
            features['hr_max'] = np.max(hr_values)
        else:
            features['hr_mean'] = np.nan
            features['hr_std'] = np.nan
            features['hr_min'] = np.nan
            features['hr_max'] = np.nan
        
        # ─────────────────────────────────────────────────────────────────────
        # C. HRV TIME DOMAIN
        # ─────────────────────────────────────────────────────────────────────
        if len(rpeaks) >= 5:
            try:
                # HRV features (time domain)
                hrv = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
                
                # Extraire features principales
                features['hrv_rmssd'] = hrv['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv.columns else np.nan
                features['hrv_sdnn'] = hrv['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv.columns else np.nan
                features['hrv_sdsd'] = hrv['HRV_SDSD'].values[0] if 'HRV_SDSD' in hrv.columns else np.nan
                features['hrv_nn50'] = hrv['HRV_NN50'].values[0] if 'HRV_NN50' in hrv.columns else np.nan
                features['hrv_pnn50'] = hrv['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv.columns else np.nan
                
            except:
                features['hrv_rmssd'] = np.nan
                features['hrv_sdnn'] = np.nan
                features['hrv_sdsd'] = np.nan
                features['hrv_nn50'] = np.nan
                features['hrv_pnn50'] = np.nan
        else:
            features['hrv_rmssd'] = np.nan
            features['hrv_sdnn'] = np.nan
            features['hrv_sdsd'] = np.nan
            features['hrv_nn50'] = np.nan
            features['hrv_pnn50'] = np.nan
        
        # ─────────────────────────────────────────────────────────────────────
        # D. INTERVALLES ECG (P-QRS-T)
        # ─────────────────────────────────────────────────────────────────────
        try:
            # Délinéation complète (coûteux, peut échouer)
            _, waves = nk.ecg_delineate(lead_ii, rpeaks, sampling_rate=sampling_rate, method='dwt')
            
            # Extraire intervalles si disponibles
            p_peaks = waves.get('ECG_P_Peaks', [])
            q_peaks = waves.get('ECG_Q_Peaks', [])
            s_peaks = waves.get('ECG_S_Peaks', [])
            t_peaks = waves.get('ECG_T_Peaks', [])
            
            # PR interval (P à R)
            if len(p_peaks) > 0 and len(rpeaks) > 0:
                pr_intervals = []
                for p_idx in p_peaks:
                    if not np.isnan(p_idx):
                        # Trouver R-peak suivant
                        next_r = rpeaks[rpeaks > p_idx]
                        if len(next_r) > 0:
                            pr_intervals.append((next_r[0] - p_idx) / sampling_rate * 1000)  # ms
                features['pr_interval_mean'] = np.mean(pr_intervals) if len(pr_intervals) > 0 else np.nan
            else:
                features['pr_interval_mean'] = np.nan
            
            # QRS duration (Q à S)
            if len(q_peaks) > 0 and len(s_peaks) > 0:
                qrs_durations = []
                for q_idx, s_idx in zip(q_peaks, s_peaks):
                    if not np.isnan(q_idx) and not np.isnan(s_idx):
                        qrs_durations.append((s_idx - q_idx) / sampling_rate * 1000)  # ms
                features['qrs_duration_mean'] = np.mean(qrs_durations) if len(qrs_durations) > 0 else np.nan
            else:
                features['qrs_duration_mean'] = np.nan
            
            # QT interval (Q à T)
            if len(q_peaks) > 0 and len(t_peaks) > 0:
                qt_intervals = []
                for q_idx, t_idx in zip(q_peaks, t_peaks):
                    if not np.isnan(q_idx) and not np.isnan(t_idx):
                        qt_intervals.append((t_idx - q_idx) / sampling_rate * 1000)  # ms
                features['qt_interval_mean'] = np.mean(qt_intervals) if len(qt_intervals) > 0 else np.nan
            else:
                features['qt_interval_mean'] = np.nan
            
            features['delineation_ok'] = 1
            
        except:
            features['pr_interval_mean'] = np.nan
            features['qrs_duration_mean'] = np.nan
            features['qt_interval_mean'] = np.nan
            features['delineation_ok'] = 0
        
    except:
        # Si R-peaks échoue complètement
        features['n_rpeaks'] = 0
        features['rpeaks_ok'] = 0
        features['hr_mean'] = np.nan
        features['hr_std'] = np.nan
        features['hr_min'] = np.nan
        features['hr_max'] = np.nan
        features['hrv_rmssd'] = np.nan
        features['hrv_sdnn'] = np.nan
        features['hrv_sdsd'] = np.nan
        features['hrv_nn50'] = np.nan
        features['hrv_pnn50'] = np.nan
        features['pr_interval_mean'] = np.nan
        features['qrs_duration_mean'] = np.nan
        features['qt_interval_mean'] = np.nan
        features['delineation_ok'] = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # E. ENTROPIES SIMPLES
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Sample Entropy
        features['sample_entropy'] = nk.entropy_sample(lead_ii, delay=1, dimension=2)
    except:
        features['sample_entropy'] = np.nan
    
    try:
        # Approximate Entropy
        features['approx_entropy'] = nk.entropy_approximate(lead_ii, delay=1, dimension=2)
    except:
        features['approx_entropy'] = np.nan
    
    # ─────────────────────────────────────────────────────────────────────────
    # F. QUALITÉ SIGNAL
    # ─────────────────────────────────────────────────────────────────────────
    # NaN ratio sur tous les leads
    features['nan_ratio'] = np.isnan(X_clean).sum() / X_clean.size
    
    # Amplitude moyenne (Lead II)
    features['amplitude_mean'] = np.nanmean(np.abs(lead_ii))
    features['amplitude_std'] = np.nanstd(lead_ii)
    
    return features

# Traiter tous les ECG
clinical_features_list = []

for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting"):
    # Charger signal nettoyé
    signal_path = row['X_clean_path']
    
    try:
        data = np.load(signal_path)
        X_clean = data['signal']  # (12, 1000)
        
        # Extraire features
        features = extract_clinical_features(X_clean, sampling_rate=100)
        features['ecg_id'] = ecg_id
        
        clinical_features_list.append(features)
        
    except Exception as e:
        # ECG en erreur: features NaN
        features = {
            'ecg_id': ecg_id,
            'n_rpeaks': 0,
            'rpeaks_ok': 0,
            'hr_mean': np.nan,
            'hr_std': np.nan,
            'hr_min': np.nan,
            'hr_max': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_sdnn': np.nan,
            'hrv_sdsd': np.nan,
            'hrv_nn50': np.nan,
            'hrv_pnn50': np.nan,
            'pr_interval_mean': np.nan,
            'qrs_duration_mean': np.nan,
            'qt_interval_mean': np.nan,
            'delineation_ok': 0,
            'sample_entropy': np.nan,
            'approx_entropy': np.nan,
            'nan_ratio': 1.0,
            'amplitude_mean': np.nan,
            'amplitude_std': np.nan
        }
        clinical_features_list.append(features)

# Créer DataFrame
df_clinical = pd.DataFrame(clinical_features_list)
df_clinical.set_index('ecg_id', inplace=True)

print(f"\n  ✓ Features cliniques extraites: {len(df_clinical.columns)} colonnes")
print(f"    Exemples: {list(df_clinical.columns[:5])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACTION FEATURES METADATA (CSV)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Extraction features metadata (CSV)...")

metadata_cols = [
    'age', 'sex', 'height', 'weight',
    'nurse', 'site', 'device',
    'baseline_drift', 'static_noise', 'burst_noise',
    'electrodes_problems', 'extra_beats', 'pacemaker',
    'heart_axis', 'strat_fold'
]

# Vérifier colonnes disponibles
metadata_cols_existing = [c for c in metadata_cols if c in df.columns]
df_metadata = df[metadata_cols_existing].copy()

print(f"  ✓ Features metadata extraites: {len(df_metadata.columns)} colonnes")
print(f"    Exemples: {list(df_metadata.columns[:5])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FUSION CLINICAL + METADATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Fusion features Wide (Clinical + Metadata)...")

# Joindre clinical + metadata
df_wide = df_clinical.join(df_metadata, how='inner')

print(f"  ✓ Features Wide total: {len(df_wide.columns)} colonnes")
print(f"    • Clinical : {len(df_clinical.columns)}")
print(f"    • Metadata : {len(df_metadata.columns)}")

# Sauvegarder
output_path = 'ptbxl_wide_features.csv'
df_wide.to_csv(output_path)
print(f"  ✓ Sauvegardé: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STATISTIQUES FINALES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STATISTIQUES FINALES")
print("=" * 100)

print(f"\n📊 WIDE FEATURES:")
print(f"  • Total ECG               : {len(df_wide):,}")
print(f"  • Features clinical (NK2) : {len(df_clinical.columns)}")
print(f"  • Features metadata (CSV) : {len(df_metadata.columns)}")
print(f"  • Total features Wide     : {len(df_wide.columns)}")

print(f"\n🔍 QUALITÉ EXTRACTION:")
rpeaks_success_rate = (df_wide['rpeaks_ok'].sum() / len(df_wide)) * 100
print(f"  • R-peaks détectés        : {rpeaks_success_rate:.1f}%")

delineation_success_rate = (df_wide['delineation_ok'].sum() / len(df_wide)) * 100
print(f"  • Délinéation réussie     : {delineation_success_rate:.1f}%")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"  • {output_path}")

print(f"\n✅ STEP 3 TERMINÉ")
print(f"   Prochaine étape: step4_wide_preprocessing.py")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# APERÇU FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📋 APERÇU FEATURES WIDE:")
print(df_wide.head())
print(f"\n{df_wide.describe()}")
