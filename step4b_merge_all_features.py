"""
=================================================================
STEP 4B: FUSION DES 3 TYPES DE FEATURES
=================================================================

Fusionne les 3 types de features pour créer le dataset final:
  1. Features Excel (68) - démographiques, qualité, temporelles
  2. Features Deep (64) - CNN + Transformer latent representations
  3. Features NeuroKit2 (25) - HR, HRV, intervals, entropy

TOTAL: 157 FEATURES

OUTPUT:
  • all_features/W_all_train.npz (17,182 × 157)
  • all_features/W_all_val.npz (2,137 × 157)
  • all_features/W_all_test.npz (2,162 × 157)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

print("=" * 100)
print("STEP 4B: FUSION DES 3 TYPES DE FEATURES (157 TOTAL)")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

EXCEL_FILE = 'PTB_XL_ML_Features_WITH_FILENAMES.xlsx'
DEEP_DIR = Path('deep_features')
NEUROKIT_FILE = 'ptbxl_wide_features.csv'
OUTPUT_DIR = Path('all_features')
OUTPUT_DIR.mkdir(exist_ok=True)

# Colonnes à exclure d'Excel (déjà dans labels ou non-features)
EXCLUDE_EXCEL_COLS = [
    'ecg_id', 'patient_id', 'filename_lr', 'filename_hr',
    'report', 'validated_by', 'nurse', 'site', 'device',
    'recording_date', 'strat_fold',
    # Exclure les labels SCP (déjà dans labels)
] + [f'scp_{x}' for x in ['SR', 'NORM', 'ABQRS', 'IMI', 'ASMI', 'LVH', 'NDT', 'LAFB', 
                           'AFIB', 'ISC_', 'PVC', 'IRBBB', 'STD_', 'VCLVH', 'STACH', 
                           'IVCD', '1AVB', 'SARRH', 'NST_', 'ISCAL', 'SBRAD', 'CRBBB', 
                           'QWAVE', 'CLBBB', 'ILMI', 'LOWT', 'LAO/LAE', 'NT_', 'PAC', 'AMI']] \
  + [f'scp_superclass_{x}' for x in ['NORM', 'MI', 'STTC', 'CD', 'HYP']]

print(f"\n⚙️  CONFIGURATION:")
print(f"  • Output: {OUTPUT_DIR}/")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[1/5] Chargement des 3 types de features...")

# ─────────────────────────────────────────────────────────────────────────
# 0. IDENTIFIER SIGNAUX NETTOYÉS DISPONIBLES
# ─────────────────────────────────────────────────────────────────────────
print(f"\n  Identification des signaux nettoyés...")
SIGNALS_DIR = Path('cleaned_signals_100hz')
available_signals = []
for npz_file in SIGNALS_DIR.glob('X_clean_*.npz'):
    ecg_id = int(npz_file.stem.replace('X_clean_', ''))
    available_signals.append(ecg_id)

available_signals = set(available_signals)
print(f"    ✓ {len(available_signals)} signaux nettoyés disponibles")

# ─────────────────────────────────────────────────────────────────────────
# A. Features Excel (Type 1)
# ─────────────────────────────────────────────────────────────────────────
print(f"\n  Type 1: Features Excel...")
df_train_excel = pd.read_excel(EXCEL_FILE, sheet_name='Train')
df_val_excel = pd.read_excel(EXCEL_FILE, sheet_name='Val')
df_test_excel = pd.read_excel(EXCEL_FILE, sheet_name='Test')

# FILTRER UNIQUEMENT LES ECG AVEC SIGNAUX NETTOYÉS
df_train_excel = df_train_excel[df_train_excel['ecg_id'].isin(available_signals)]
df_val_excel = df_val_excel[df_val_excel['ecg_id'].isin(available_signals)]
df_test_excel = df_test_excel[df_test_excel['ecg_id'].isin(available_signals)]

print(f"    ✓ Train: {len(df_train_excel)} ECG (filtered)")
print(f"    ✓ Val  : {len(df_val_excel)} ECG (filtered)")
print(f"    ✓ Test : {len(df_test_excel)} ECG (filtered)")

# Sélectionner colonnes features
excel_cols = [col for col in df_train_excel.columns if col not in EXCLUDE_EXCEL_COLS]
print(f"    ✓ {len(excel_cols)} colonnes Excel")
print(f"    Exemples: {excel_cols[:5]}")

# ─────────────────────────────────────────────────────────────────────────
# B. Features Deep (Type 2)
# ─────────────────────────────────────────────────────────────────────────
print(f"\n  Type 2: Features Deep (CNN + Transformer)...")
deep_train = np.load(DEEP_DIR / 'deep_features_train.npz')
deep_val = np.load(DEEP_DIR / 'deep_features_val.npz')
deep_test = np.load(DEEP_DIR / 'deep_features_test.npz')

Deep_train = deep_train['features']
Deep_val = deep_val['features']
Deep_test = deep_test['features']

deep_ids_train = deep_train['ecg_ids']
deep_ids_val = deep_val['ecg_ids']
deep_ids_test = deep_test['ecg_ids']

# FILTRER LES DEEP FEATURES POUR UTILISER UNIQUEMENT LES ECG AVEC SIGNAUX
train_mask = np.isin(deep_ids_train, list(available_signals))
val_mask = np.isin(deep_ids_val, list(available_signals))
test_mask = np.isin(deep_ids_test, list(available_signals))

Deep_train = Deep_train[train_mask]
Deep_val = Deep_val[val_mask]
Deep_test = Deep_test[test_mask]

deep_ids_train = deep_ids_train[train_mask]
deep_ids_val = deep_ids_val[val_mask]
deep_ids_test = deep_ids_test[test_mask]

print(f"    ✓ Train: {len(deep_ids_train)} ECG × {Deep_train.shape[1]} features")
print(f"    ✓ Val  : {len(deep_ids_val)} ECG × {Deep_val.shape[1]} features")
print(f"    ✓ Test : {len(deep_ids_test)} ECG × {Deep_test.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────
# C. Features NeuroKit2 (Type 3)
# ─────────────────────────────────────────────────────────────────────────
print(f"\n  Type 3: Features NeuroKit2...")
df_nk2 = pd.read_csv(NEUROKIT_FILE, index_col='ecg_id')

# Colonnes NeuroKit2 (exclure strat_fold)
nk2_cols = [col for col in df_nk2.columns if col != 'strat_fold']
print(f"    ✓ {len(nk2_cols)} features NeuroKit2")
print(f"    Exemples: {nk2_cols[:5]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUSION DES FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[2/5] Fusion des 3 types...")

def merge_all_features(df_excel, Deep_feats, deep_ids, split_name):
    """Fusionne Excel + Deep + NeuroKit2 pour un split"""
    
    # Convertir ecg_id en index si pas déjà fait
    if 'ecg_id' in df_excel.columns:
        df_excel = df_excel.set_index('ecg_id')
    
    # Aligner Deep features sur les ecg_ids
    df_deep = pd.DataFrame(
        Deep_feats,
        index=deep_ids,
        columns=[f'deep_{i:02d}' for i in range(Deep_feats.shape[1])]
    )
    
    # Aligner NeuroKit2 features
    df_nk2_split = df_nk2.loc[deep_ids, nk2_cols]
    
    # Excel features
    df_excel_split = df_excel.loc[deep_ids, excel_cols]
    
    # Fusionner les 3
    df_all = pd.concat([
        df_excel_split,    # Type 1: Excel (68)
        df_deep,           # Type 2: Deep (64)
        df_nk2_split       # Type 3: NeuroKit2 (25)
    ], axis=1)
    
    print(f"    {split_name}: {df_all.shape[0]} ECG × {df_all.shape[1]} features")
    print(f"      • Excel: {len(excel_cols)}")
    print(f"      • Deep: {Deep_feats.shape[1]}")
    print(f"      • NeuroKit2: {len(nk2_cols)}")
    
    return df_all

# Fusion
df_all_train = merge_all_features(df_train_excel, Deep_train, deep_ids_train, "Train")
df_all_val = merge_all_features(df_val_excel, Deep_val, deep_ids_val, "Val")
df_all_test = merge_all_features(df_test_excel, Deep_test, deep_ids_test, "Test")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/5] Preprocessing (imputation + scaling)...")

# Identifier colonnes numériques (toutes sauf catégorielles)
num_cols = df_all_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"  ✓ {len(num_cols)} colonnes numériques")

# Pipeline: Imputation + Scaling
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# FIT sur Train uniquement
W_all_train = preprocessor.fit_transform(df_all_train[num_cols])
W_all_val = preprocessor.transform(df_all_val[num_cols])
W_all_test = preprocessor.transform(df_all_test[num_cols])

print(f"  ✓ Train: {W_all_train.shape}")
print(f"  ✓ Val  : {W_all_val.shape}")
print(f"  ✓ Test : {W_all_test.shape}")

# Vérifier NaN
print(f"\n  🔍 VÉRIFICATION NaN:")
print(f"    • Train: {np.isnan(W_all_train).sum()} NaN")
print(f"    • Val  : {np.isnan(W_all_val).sum()} NaN")
print(f"    • Test : {np.isnan(W_all_test).sum()} NaN")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/5] Sauvegarde...")

np.savez_compressed(
    OUTPUT_DIR / 'W_all_train.npz',
    W=W_all_train,
    ecg_ids=deep_ids_train
)

np.savez_compressed(
    OUTPUT_DIR / 'W_all_val.npz',
    W=W_all_val,
    ecg_ids=deep_ids_val
)

np.savez_compressed(
    OUTPUT_DIR / 'W_all_test.npz',
    W=W_all_test,
    ecg_ids=deep_ids_test
)

# Sauvegarder preprocessor
with open(OUTPUT_DIR / 'preprocessor_all.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/W_all_train.npz")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/W_all_val.npz")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/W_all_test.npz")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/preprocessor_all.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/5] Création CSV récapitulatif...")

# Sauvegarder aussi en CSV pour inspection
df_all_train.to_csv(OUTPUT_DIR / 'all_features_train.csv')
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/all_features_train.csv")

print("\n" + "=" * 100)
print("STATISTIQUES FINALES")
print("=" * 100)

print(f"\n📊 FEATURES TOTALES FUSIONNÉES:")
print(f"  • Type 1 (Excel)     : {len(excel_cols)} features")
print(f"  • Type 2 (Deep)      : {Deep_train.shape[1]} features")
print(f"  • Type 3 (NeuroKit2) : {len(nk2_cols)} features")
print(f"  • TOTAL              : {W_all_train.shape[1]} features")

print(f"\n📈 SPLITS:")
print(f"  • Train: {W_all_train.shape[0]} ECG × {W_all_train.shape[1]} features")
print(f"  • Val  : {W_all_val.shape[0]} ECG × {W_all_val.shape[1]} features")
print(f"  • Test : {W_all_test.shape[0]} ECG × {W_all_test.shape[1]} features")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"  • {OUTPUT_DIR}/W_all_train.npz")
print(f"  • {OUTPUT_DIR}/W_all_val.npz")
print(f"  • {OUTPUT_DIR}/W_all_test.npz")
print(f"  • {OUTPUT_DIR}/preprocessor_all.pkl")
print(f"  • {OUTPUT_DIR}/all_features_train.csv")

print(f"\n✅ STEP 4B TERMINÉ")
print(f"   Prochaine étape: Modifier step6_training.py pour utiliser {OUTPUT_DIR}/")
print("=" * 100)
