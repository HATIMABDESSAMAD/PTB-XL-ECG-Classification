"""
STEP 4C: FEATURE ENGINEERING SUR LES 124 FEATURES
====================================================
Nettoyage et amélioration des features avant entraînement:
- Supprimer colonnes inutiles (ID, index, duplicates)
- Créer features dérivées
- Optimiser les features catégorielles
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_DIR = Path('all_features')
OUTPUT_DIR = Path('all_features_engineered')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*85)
print("STEP 4C: FEATURE ENGINEERING")
print("="*85)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Chargement des données...")

# Charger les données
data_train = np.load(INPUT_DIR / 'W_all_train.npz')
data_val = np.load(INPUT_DIR / 'W_all_val.npz')
data_test = np.load(INPUT_DIR / 'W_all_test.npz')

W_train = data_train['W']
W_val = data_val['W']
W_test = data_test['W']

ids_train = data_train['ecg_ids']
ids_val = data_val['ecg_ids']
ids_test = data_test['ecg_ids']

# Charger le CSV pour avoir les noms de colonnes
df_train_original = pd.read_csv(INPUT_DIR / 'all_features_train.csv')

# Charger les filenames depuis Excel
df_excel_train = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Train')
df_excel_val = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Val')
df_excel_test = pd.read_excel('PTB_XL_ML_Features_WITH_FILENAMES.xlsx', sheet_name='Test')

# Créer mapping ecg_id -> filename_lr
filename_map_train = df_excel_train.set_index('ecg_id')['filename_lr'].to_dict()
filename_map_val = df_excel_val.set_index('ecg_id')['filename_lr'].to_dict()
filename_map_test = df_excel_test.set_index('ecg_id')['filename_lr'].to_dict()

# Obtenir filenames pour les IDs
filenames_train = np.array([filename_map_train[ecg_id] for ecg_id in ids_train])
filenames_val = np.array([filename_map_val[ecg_id] for ecg_id in ids_val])
filenames_test = np.array([filename_map_test[ecg_id] for ecg_id in ids_test])

print(f"  ✓ Shape avant: Train {W_train.shape}, Val {W_val.shape}, Test {W_test.shape}")
print(f"  ✓ Colonnes: {len(df_train_original.columns)}")
print(f"  ✓ Filenames mappés: Train {len(filenames_train)}, Val {len(filenames_val)}, Test {len(filenames_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IDENTIFICATION COLONNES À SUPPRIMER
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Feature Engineering...")

# Reconstruire DataFrame pour faciliter le traitement
df_train = pd.DataFrame(W_train, columns=df_train_original.columns)
df_val = pd.DataFrame(W_val, columns=df_train_original.columns)
df_test = pd.DataFrame(W_test, columns=df_train_original.columns)

# Colonnes à supprimer
COLS_TO_DROP = []

# 1. Index inutile
if 'Unnamed: 0' in df_train.columns:
    COLS_TO_DROP.append('Unnamed: 0')

# 2. Identifiants
if 'ecg_id' in df_train.columns:
    COLS_TO_DROP.append('ecg_id')

# 3. Colonnes dupliquées (suffixe .1, .2, etc.)
duplicate_cols = [col for col in df_train.columns if '.1' in col or '.2' in col]
COLS_TO_DROP.extend(duplicate_cols)

# 4. Colonnes avec variance nulle (constantes)
constant_cols = []
for col in df_train.columns:
    if col not in COLS_TO_DROP:
        if df_train[col].std() == 0:
            constant_cols.append(col)
            COLS_TO_DROP.append(col)

# 5. Colonnes avec trop de NaN (>95%)
high_nan_cols = []
for col in df_train.columns:
    if col not in COLS_TO_DROP:
        nan_ratio = df_train[col].isna().sum() / len(df_train)
        if nan_ratio > 0.95:
            high_nan_cols.append(col)
            COLS_TO_DROP.append(col)

print(f"\n  Colonnes à supprimer:")
print(f"    • Index/ID: {[c for c in COLS_TO_DROP if c in ['Unnamed: 0', 'ecg_id']]}")
print(f"    • Duplicates: {len(duplicate_cols)} colonnes")
print(f"    • Constantes: {len(constant_cols)} colonnes")
print(f"    • High NaN (>95%): {len(high_nan_cols)} colonnes")
print(f"    • TOTAL: {len(COLS_TO_DROP)} colonnes à supprimer")

# Supprimer les colonnes
df_train_clean = df_train.drop(columns=COLS_TO_DROP, errors='ignore')
df_val_clean = df_val.drop(columns=COLS_TO_DROP, errors='ignore')
df_test_clean = df_test.drop(columns=COLS_TO_DROP, errors='ignore')

print(f"\n  Shape après suppression: {df_train_clean.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CRÉATION FEATURES DÉRIVÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Création features dérivées...")

def create_derived_features(df):
    """Crée des features dérivées pertinentes"""
    df_new = df.copy()
    
    # 1. BMI categories interaction avec age
    if 'bmi' in df.columns and 'age' in df.columns:
        df_new['bmi_age_risk'] = df['bmi'] * (df['age'] / 100)  # Risque cardiovasculaire
    
    # 2. Ratio taille/poids (alternative au BMI)
    if 'height' in df.columns and 'weight' in df.columns:
        df_new['height_weight_ratio'] = df['height'] / (df['weight'] + 1e-6)
    
    # 3. HRV stress index (SDNN/RMSSD ratio)
    if 'hrv_sdnn' in df.columns and 'hrv_rmssd' in df.columns:
        df_new['hrv_stress_index'] = df['hrv_sdnn'] / (df['hrv_rmssd'] + 1e-6)
    
    # 4. QT corrected (formule de Bazett: QTc = QT / sqrt(RR))
    if 'qt_mean' in df.columns and 'hr_mean' in df.columns:
        rr_interval = 60000 / (df['hr_mean'] + 1e-6)  # ms
        df_new['qtc_bazett'] = df['qt_mean'] / (np.sqrt(rr_interval / 1000) + 1e-6)
    
    # 5. Ratio PR/QRS (conduction auriculo-ventriculaire)
    if 'pr_mean' in df.columns and 'qrs_mean' in df.columns:
        df_new['pr_qrs_ratio'] = df['pr_mean'] / (df['qrs_mean'] + 1e-6)
    
    # 6. Quality composite score
    quality_cols = [c for c in df.columns if 'quality' in c.lower()]
    if len(quality_cols) > 0:
        df_new['quality_composite'] = df[quality_cols].mean(axis=1)
    
    # 7. Age risk score (non-linéaire)
    if 'age' in df.columns:
        df_new['age_risk_score'] = (df['age'] / 100) ** 2  # Risque augmente exponentiellement
    
    # 8. Sex-age interaction
    if 'age' in df.columns and 'sex' in df.columns:
        df_new['sex_age_interaction'] = df['sex'] * (df['age'] / 100)
    
    # 9. Amplitude variability (pour arythmies)
    if 'amplitude_mean' in df.columns and 'amplitude_std' in df.columns:
        df_new['amplitude_cv'] = df['amplitude_std'] / (df['amplitude_mean'] + 1e-6)
    
    # 10. HR variability index
    if 'hr_std' in df.columns and 'hr_mean' in df.columns:
        df_new['hr_variability_index'] = df['hr_std'] / (df['hr_mean'] + 1e-6)
    
    return df_new

df_train_eng = create_derived_features(df_train_clean)
df_val_eng = create_derived_features(df_val_clean)
df_test_eng = create_derived_features(df_test_clean)

new_features = [c for c in df_train_eng.columns if c not in df_train_clean.columns]
print(f"  ✓ Créées: {len(new_features)} nouvelles features")
print(f"    Exemples: {new_features[:5]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREPROCESSING FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/5] Preprocessing final...")

# Identifier colonnes numériques uniquement
num_cols = df_train_eng.select_dtypes(include=[np.number]).columns.tolist()
print(f"  ✓ {len(num_cols)} colonnes numériques")

# Pipeline: Imputation + Scaling
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# FIT sur Train uniquement
W_train_final = preprocessor.fit_transform(df_train_eng[num_cols])
W_val_final = preprocessor.transform(df_val_eng[num_cols])
W_test_final = preprocessor.transform(df_test_eng[num_cols])

print(f"  ✓ Shape final: Train {W_train_final.shape}, Val {W_val_final.shape}, Test {W_test_final.shape}")

# Vérification NaN
print(f"\n  🔍 VÉRIFICATION NaN:")
print(f"    • Train: {np.isnan(W_train_final).sum()} NaN")
print(f"    • Val  : {np.isnan(W_val_final).sum()} NaN")
print(f"    • Test : {np.isnan(W_test_final).sum()} NaN")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Sauvegarde...")

# Sauvegarder NPZ avec filenames
np.savez_compressed(
    OUTPUT_DIR / 'W_engineered_train.npz',
    W=W_train_final,
    ecg_ids=ids_train,
    filenames=filenames_train,
    feature_names=num_cols
)

np.savez_compressed(
    OUTPUT_DIR / 'W_engineered_val.npz',
    W=W_val_final,
    ecg_ids=ids_val,
    filenames=filenames_val,
    feature_names=num_cols
)

np.savez_compressed(
    OUTPUT_DIR / 'W_engineered_test.npz',
    W=W_test_final,
    ecg_ids=ids_test,
    filenames=filenames_test,
    feature_names=num_cols
)

# Sauvegarder preprocessor
import joblib
joblib.dump(preprocessor, OUTPUT_DIR / 'preprocessor_engineered.pkl')

# Sauvegarder CSV pour inspection
pd.DataFrame(
    W_train_final,
    columns=num_cols
).to_csv(OUTPUT_DIR / 'features_engineered_train.csv', index=False)

print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_engineered_train.npz'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_engineered_val.npz'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'W_engineered_test.npz'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'preprocessor_engineered.pkl'}")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR / 'features_engineered_train.csv'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STATISTIQUES FINALES")
print("="*85)

print(f"\n📊 FEATURE ENGINEERING:")
print(f"  • Features avant       : {df_train.shape[1]}")
print(f"  • Supprimées           : {len(COLS_TO_DROP)}")
print(f"  • Créées (dérivées)    : {len(new_features)}")
print(f"  • Features après       : {W_train_final.shape[1]}")
print(f"  • Gain/Perte           : {W_train_final.shape[1] - df_train.shape[1]:+d} features")

print(f"\n📈 SPLITS:")
print(f"  • Train: {W_train_final.shape[0]} ECG × {W_train_final.shape[1]} features")
print(f"  • Val  : {W_val_final.shape[0]} ECG × {W_val_final.shape[1]} features")
print(f"  • Test : {W_test_final.shape[0]} ECG × {W_test_final.shape[1]} features")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"  • {OUTPUT_DIR / 'W_engineered_train.npz'}")
print(f"  • {OUTPUT_DIR / 'W_engineered_val.npz'}")
print(f"  • {OUTPUT_DIR / 'W_engineered_test.npz'}")
print(f"  • {OUTPUT_DIR / 'preprocessor_engineered.pkl'}")
print(f"  • {OUTPUT_DIR / 'features_engineered_train.csv'}")

print(f"\n✅ STEP 4C TERMINÉ")
print(f"   Prochaine étape: step6b_training_all_features.py avec wide_dir='all_features_engineered'")
print("="*85)
