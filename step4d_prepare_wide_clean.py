"""
STEP 4D: PRÉPARATION FEATURES WIDE PROPRES
===========================================
Crée 2 versions de features Wide:
1. Wide Pure (32 Excel uniquement)
2. Wide Hybride (32 Excel + 26 NeuroKit2)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('wide_features_clean')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*85)
print("STEP 4D: PRÉPARATION FEATURES WIDE PROPRES")
print("="*85)

# ============================================================================
# 1. CHARGEMENT DONNÉES
# ============================================================================

print("\n[1/4] Chargement des données...")

# Charger Excel
df_train_excel = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Train')
df_val_excel = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Val')
df_test_excel = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Test')

# Charger NeuroKit2
df_nk2 = pd.read_csv('ptbxl_wide_features.csv', index_col='ecg_id')

# Identifier signaux disponibles
SIGNALS_DIR = Path('cleaned_signals_100hz')
available_signals = []
for npz_file in SIGNALS_DIR.glob('X_clean_*.npz'):
    ecg_id = int(npz_file.stem.replace('X_clean_', ''))
    available_signals.append(ecg_id)
available_signals = set(available_signals)

print(f"  ✓ {len(available_signals)} signaux disponibles")

# Filtrer par signaux disponibles
df_train_excel = df_train_excel[df_train_excel['ecg_id'].isin(available_signals)]
df_val_excel = df_val_excel[df_val_excel['ecg_id'].isin(available_signals)]
df_test_excel = df_test_excel[df_test_excel['ecg_id'].isin(available_signals)]

# ============================================================================
# 2. SÉLECTION COLONNES EXCEL (32 features)
# ============================================================================

print("\n[2/4] Sélection features Excel...")

# Colonnes à exclure
EXCLUDE_COLS = [
    'ecg_id', 'patient_id', 'filename_lr', 'filename_hr',
    'report', 'validated_by', 'nurse', 'site', 'device',
    'recording_date', 'strat_fold'
] + [f'scp_{x}' for x in ['SR', 'NORM', 'ABQRS', 'IMI', 'ASMI', 'LVH', 'NDT', 'LAFB', 
                          'AFIB', 'ISC_', 'PVC', 'IRBBB', 'STD_', 'VCLVH', 'STACH', 
                          'IVCD', '1AVB', 'SARRH', 'NST_', 'ISCAL', 'SBRAD', 'CRBBB', 
                          'QWAVE', 'CLBBB', 'ILMI', 'LOWT', 'LAO/LAE', 'NT_', 'PAC', 'AMI']] \
  + [f'scp_superclass_{x}' for x in ['NORM', 'MI', 'STTC', 'CD', 'HYP']]

# Features Excel
excel_cols = [col for col in df_train_excel.columns if col not in EXCLUDE_COLS]
print(f"  ✓ {len(excel_cols)} features Excel")

# Features NeuroKit2 (exclure strat_fold)
nk2_cols = [col for col in df_nk2.columns if col != 'strat_fold']
print(f"  ✓ {len(nk2_cols)} features NeuroKit2")

# ============================================================================
# 3. CRÉATION DES 2 VERSIONS
# ============================================================================

print("\n[3/4] Création des 2 versions...")

# Extraire features Excel
W_excel_train = df_train_excel[excel_cols].values
W_excel_val = df_val_excel[excel_cols].values
W_excel_test = df_test_excel[excel_cols].values

# Extraire features NeuroKit2
ids_train = df_train_excel['ecg_id'].values
ids_val = df_val_excel['ecg_id'].values
ids_test = df_test_excel['ecg_id'].values

W_nk2_train = df_nk2.loc[ids_train, nk2_cols].values
W_nk2_val = df_nk2.loc[ids_val, nk2_cols].values
W_nk2_test = df_nk2.loc[ids_test, nk2_cols].values

# Filenames
filenames_train = df_train_excel['filename_lr'].values
filenames_val = df_val_excel['filename_lr'].values
filenames_test = df_test_excel['filename_lr'].values

# --- VERSION 1: PURE (32 Excel) ---
print("\n  Version 1: Wide Pure (32 Excel)")
W_pure_train = W_excel_train
W_pure_val = W_excel_val
W_pure_test = W_excel_test

# Preprocessing
preprocessor_pure = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

W_pure_train_scaled = preprocessor_pure.fit_transform(W_pure_train)
W_pure_val_scaled = preprocessor_pure.transform(W_pure_val)
W_pure_test_scaled = preprocessor_pure.transform(W_pure_test)

print(f"    ✓ Train: {W_pure_train_scaled.shape}")
print(f"    ✓ Val  : {W_pure_val_scaled.shape}")
print(f"    ✓ Test : {W_pure_test_scaled.shape}")

# --- VERSION 2: HYBRIDE (32 Excel + 26 NeuroKit2) ---
print("\n  Version 2: Wide Hybride (32 Excel + 26 NeuroKit2)")
W_hybrid_train = np.hstack([W_excel_train, W_nk2_train])
W_hybrid_val = np.hstack([W_excel_val, W_nk2_val])
W_hybrid_test = np.hstack([W_excel_test, W_nk2_test])

# Preprocessing
preprocessor_hybrid = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

W_hybrid_train_scaled = preprocessor_hybrid.fit_transform(W_hybrid_train)
W_hybrid_val_scaled = preprocessor_hybrid.transform(W_hybrid_val)
W_hybrid_test_scaled = preprocessor_hybrid.transform(W_hybrid_test)

print(f"    ✓ Train: {W_hybrid_train_scaled.shape}")
print(f"    ✓ Val  : {W_hybrid_val_scaled.shape}")
print(f"    ✓ Test : {W_hybrid_test_scaled.shape}")

# ============================================================================
# 4. SAUVEGARDE
# ============================================================================

print("\n[4/4] Sauvegarde...")

# Version Pure
np.savez_compressed(
    OUTPUT_DIR / 'W_pure_train.npz',
    W=W_pure_train_scaled,
    ecg_ids=ids_train,
    filenames=filenames_train
)
np.savez_compressed(
    OUTPUT_DIR / 'W_pure_val.npz',
    W=W_pure_val_scaled,
    ecg_ids=ids_val,
    filenames=filenames_val
)
np.savez_compressed(
    OUTPUT_DIR / 'W_pure_test.npz',
    W=W_pure_test_scaled,
    ecg_ids=ids_test,
    filenames=filenames_test
)

# Version Hybride
np.savez_compressed(
    OUTPUT_DIR / 'W_hybrid_train.npz',
    W=W_hybrid_train_scaled,
    ecg_ids=ids_train,
    filenames=filenames_train
)
np.savez_compressed(
    OUTPUT_DIR / 'W_hybrid_val.npz',
    W=W_hybrid_val_scaled,
    ecg_ids=ids_val,
    filenames=filenames_val
)
np.savez_compressed(
    OUTPUT_DIR / 'W_hybrid_test.npz',
    W=W_hybrid_test_scaled,
    ecg_ids=ids_test,
    filenames=filenames_test
)

print(f"\n💾 Sauvegardé:")
print(f"  • {OUTPUT_DIR}/W_pure_*.npz (32 features)")
print(f"  • {OUTPUT_DIR}/W_hybrid_*.npz (58 features)")

print("\n" + "="*85)
print("RÉSUMÉ")
print("="*85)
print(f"\n✅ VERSION 1 - Wide Pure:")
print(f"  • 32 features Excel uniquement")
print(f"  • {W_pure_train_scaled.shape[0]} Train / {W_pure_val_scaled.shape[0]} Val / {W_pure_test_scaled.shape[0]} Test")

print(f"\n✅ VERSION 2 - Wide Hybride:")
print(f"  • 32 Excel + 26 NeuroKit2 = 58 features")
print(f"  • {W_hybrid_train_scaled.shape[0]} Train / {W_hybrid_val_scaled.shape[0]} Val / {W_hybrid_test_scaled.shape[0]} Test")

print("\n✅ STEP 4D TERMINÉ")
print("="*85)
