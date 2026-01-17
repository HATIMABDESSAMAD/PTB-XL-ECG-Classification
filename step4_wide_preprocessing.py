"""
=================================================================
STEP 4: PREPROCESSING WIDE FEATURES
=================================================================

Prétraite les features Wide (Clinical + Metadata) avant modélisation.

PIPELINE:
  1. Chargement ptbxl_wide_features.csv
  2. Séparation Train/Val/Test par strat_fold
  3. Identification types features (numériques/catégorielles)
  4. Imputation (SimpleImputer) - FIT sur Train uniquement
  5. Encodage catégorielles (OneHotEncoder) - FIT sur Train uniquement
  6. Scaling (StandardScaler) - FIT sur Train uniquement
  7. Sauvegarde W_train/val/test.npz + wide_preprocessor.pkl

INPUTS:
  • ptbxl_wide_features.csv (28 colonnes: 20 clinical + 8 metadata)

OUTPUTS:
  • preprocessed_wide/W_train.npz
  • preprocessed_wide/W_val.npz
  • preprocessed_wide/W_test.npz
  • wide_preprocessor.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

print("=" * 130)
print("STEP 4: WIDE FEATURES PREPROCESSING (Imputation + Encoding + Scaling)")
print("=" * 130)

# ========================= CONFIG =========================
WIDE_FILE = 'ptbxl_wide_features.csv'
OUTPUT_DIR = Path('preprocessed_wide')
OUTPUT_DIR.mkdir(exist_ok=True)

# Colonnes à exclure du preprocessing (metadata non-features)
EXCLUDE_COLS = [
    'strat_fold', 'filename_lr', 'filename_hr', 'report', 
    'validated_by', 'nurse', 'site', 'device', 
    'recording_date', 'patient_id', 'ecg_id'
]

# ========================= [1/6] CHARGEMENT =========================
print("\n[1/6] Chargement dataset Wide...")
df = pd.read_csv(WIDE_FILE, index_col='ecg_id')
print(f"  ✓ {len(df)} enregistrements chargés")
print(f"  ✓ {len(df.columns)} colonnes totales")

# ========================= [2/6] SPLIT TRAIN/VAL/TEST =========================
print("\n[2/6] Séparation Train/Val/Test...")

# Selon ptb-xl documentation: strat_fold 1-8 pour train, 9 pour val, 10 pour test
df_train = df[df['strat_fold'] <= 8].copy()
df_val = df[df['strat_fold'] == 9].copy()
df_test = df[df['strat_fold'] == 10].copy()

print(f"  ✓ Train: {len(df_train)} ECG")
print(f"  ✓ Val  : {len(df_val)} ECG")
print(f"  ✓ Test : {len(df_test)} ECG")

# ========================= [3/6] IDENTIFICATION FEATURES =========================
print("\n[3/6] Identification types de features...")

# Retirer colonnes à exclure
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

# Supprimer colonnes entièrement NaN
feature_cols = [col for col in feature_cols if not df_train[col].isna().all()]

# Tous numériques (pas de catégorielles dans notre cas)
num_cols = feature_cols

print(f"  ✓ Features numériques: {len(num_cols)} colonnes")
print(f"    Exemples: {num_cols[:5]}")

# ========================= [4/6] CONSTRUCTION PREPROCESSOR =========================
print("\n[4/6] Construction pipeline preprocessing...")

# Pipeline simple: Imputation + Scaling
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols)
    ]
)

print(f"  ✓ Pipeline créé: Imputation(median) + StandardScaler")

# ========================= [5/6] FIT + TRANSFORM =========================
print("\n[5/6] Fit + Transform (FIT sur Train uniquement)...")

# Extraire features pour chaque split
X_train = df_train[num_cols]
X_val = df_val[num_cols] if len(df_val) > 0 else pd.DataFrame(columns=num_cols)
X_test = df_test[num_cols] if len(df_test) > 0 else pd.DataFrame(columns=num_cols)

print(f"  • Train shape avant: {X_train.shape}")
print(f"  • Val shape avant  : {X_val.shape}")
print(f"  • Test shape avant : {X_test.shape}")

# FIT sur Train uniquement
preprocessor.fit(X_train)
print(f"  ✓ Preprocessor fitted sur Train")

# TRANSFORM Train/Val/Test
W_train = preprocessor.transform(X_train)
W_val = preprocessor.transform(X_val) if len(X_val) > 0 else np.array([]).reshape(0, W_train.shape[1])
W_test = preprocessor.transform(X_test) if len(X_test) > 0 else np.array([]).reshape(0, W_train.shape[1])

print(f"  ✓ Train shape après: {W_train.shape}")
print(f"  ✓ Val shape après  : {W_val.shape}")
print(f"  ✓ Test shape après : {W_test.shape}")

# Vérifier NaN
print(f"\n  🔍 VÉRIFICATION NaN:")
print(f"    • Train: {np.isnan(W_train).sum()} NaN")
print(f"    • Val  : {np.isnan(W_val).sum()} NaN")
print(f"    • Test : {np.isnan(W_test).sum()} NaN")

# ========================= [6/6] SAUVEGARDE =========================
print("\n[6/6] Sauvegarde...")

# Sauvegarder preprocessed features
np.savez_compressed(
    OUTPUT_DIR / 'W_train.npz',
    W=W_train,
    ecg_ids=df_train.index.values
)

np.savez_compressed(
    OUTPUT_DIR / 'W_val.npz',
    W=W_val,
    ecg_ids=df_val.index.values
)

np.savez_compressed(
    OUTPUT_DIR / 'W_test.npz',
    W=W_test,
    ecg_ids=df_test.index.values
)

print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_train.npz'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_val.npz'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_test.npz'}")

# Sauvegarder preprocessor
with open('wide_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print(f"  ✓ Sauvegardé: wide_preprocessor.pkl")

# ========================= RAPPORT FINAL =========================
print("\n" + "=" * 130)
print("STATISTIQUES FINALES")
print("=" * 130)

print(f"\n📊 PREPROCESSING WIDE:")
print(f"  • Features numériques : {len(num_cols)}")
print(f"  • Features finales    : {W_train.shape[1]}")

print(f"\n📈 SPLITS:")
print(f"  • Train: {W_train.shape[0]} ECG × {W_train.shape[1]} features")
print(f"  • Val  : {W_val.shape[0]} ECG × {W_val.shape[1]} features")
print(f"  • Test : {W_test.shape[0]} ECG × {W_test.shape[1]} features")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"  • {OUTPUT_DIR}/W_train.npz")
print(f"  • {OUTPUT_DIR}/W_val.npz")
print(f"  • {OUTPUT_DIR}/W_test.npz")
print(f"  • wide_preprocessor.pkl")

print(f"\n✅ STEP 4 TERMINÉ")
print(f"   Prochaine étape: step6_training.py (step5 déjà testé)")
print("=" * 130)
