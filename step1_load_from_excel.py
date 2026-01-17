"""
═══════════════════════════════════════════════════════════════════════════════
STEP 1 (ADAPTÉ): CHARGEMENT depuis PTB_XL_ML_Features_WITH_FILENAMES.xlsx
═══════════════════════════════════════════════════════════════════════════════
Charge le fichier Excel préprocessé qui contient déjà:
  - Features tabulaires (age, sex, quality_score, etc.)
  - Labels SCP encodés (scp_NORM, scp_IMI, etc.)
  - Superclasses (scp_superclass_NORM/MI/STTC/CD/HYP)
  - filename_lr et filename_hr pour accès signaux

Sortie: dataset consolidé train/val/test prêt pour le pipeline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=" * 100)
print("STEP 1 (ADAPTÉ): CHARGEMENT depuis Excel")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT FICHIER EXCEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Chargement fichier Excel...")

excel_file = 'PTB_XL_ML_Features_WITH_FILENAMES.xlsx'

# Charger les 3 sheets
df_train = pd.read_excel(excel_file, sheet_name='Train')
df_val = pd.read_excel(excel_file, sheet_name='Val')
df_test = pd.read_excel(excel_file, sheet_name='Test')

# Définir ecg_id comme index
df_train.set_index('ecg_id', inplace=True)
df_val.set_index('ecg_id', inplace=True)
df_test.set_index('ecg_id', inplace=True)

print(f"  ✓ Train: {len(df_train):,} ECG × {len(df_train.columns)} colonnes")
print(f"  ✓ Val  : {len(df_val):,} ECG × {len(df_val.columns)} colonnes")
print(f"  ✓ Test : {len(df_test):,} ECG × {len(df_test.columns)} colonnes")

# Combiner pour analyses globales
df_all = pd.concat([df_train, df_val, df_test])
print(f"  ✓ Total: {len(df_all):,} ECG")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. IDENTIFIER COLONNES SCP et SUPERCLASSES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Identification colonnes labels...")

# Colonnes SCP individuelles
scp_cols = [col for col in df_all.columns if col.startswith('scp_') and not 'superclass' in col]
print(f"  ✓ Codes SCP: {len(scp_cols)} colonnes")
print(f"    Exemples: {scp_cols[:10]}")

# Colonnes superclasses
superclass_cols = [col for col in df_all.columns if col.startswith('scp_superclass_')]
print(f"  ✓ Superclasses: {len(superclass_cols)} colonnes")
print(f"    {superclass_cols}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATISTIQUES LABELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Statistiques labels...")

# Top 10 codes SCP
scp_counts = df_all[scp_cols].sum().sort_values(ascending=False)
print(f"\n  Top 10 codes SCP les plus fréquents:")
for i, (code, count) in enumerate(scp_counts.head(10).items(), 1):
    code_name = code.replace('scp_', '')
    pct = (count / len(df_all)) * 100
    print(f"    {i:2d}. {code_name:10s} : {int(count):5d} ({pct:5.2f}%)")

# Distribution superclasses
print(f"\n  Distribution superclasses:")
for sc_col in superclass_cols:
    sc_name = sc_col.replace('scp_superclass_', '')
    count = df_all[sc_col].sum()
    pct = (count / len(df_all)) * 100
    print(f"    {sc_name:5s} : {int(count):5d} ({pct:5.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CRÉER CONFIGURATION LABELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Création configuration labels...")

# Extraire noms de codes (sans préfixe scp_)
scp_code_names = [col.replace('scp_', '') for col in scp_cols]
superclass_names = [col.replace('scp_superclass_', '') for col in superclass_cols]

label_config = {
    # Colonnes dans Excel
    'scp_cols_excel': scp_cols,
    'superclass_cols_excel': superclass_cols,
    
    # Noms de codes (pour compatibilité)
    'valid_codes': scp_code_names,
    'superclass_names': superclass_names,
    
    # Comptages
    'n_scp_codes': len(scp_cols),
    'n_superclasses': len(superclass_cols),
    'n_total_ecg': len(df_all),
    
    # Splits
    'n_train': len(df_train),
    'n_val': len(df_val),
    'n_test': len(df_test)
}

# Sauvegarder config
with open('label_config_from_excel.json', 'w') as f:
    json.dump(label_config, f, indent=2)

print(f"  ✓ Configuration sauvegardée: label_config_from_excel.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDER DATASET CONSOLIDÉ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Sauvegarde datasets...")

# Ajouter colonne split pour identification
df_train['split'] = 'train'
df_val['split'] = 'val'
df_test['split'] = 'test'

# Combiner
df_consolidated = pd.concat([df_train, df_val, df_test])

# Sauvegarder CSV consolidé
df_consolidated.to_csv('ptbxl_from_excel_consolidated.csv')
print(f"  ✓ ptbxl_from_excel_consolidated.csv ({len(df_consolidated):,} lignes)")

# Sauvegarder aussi les splits séparés pour pipeline
df_train.to_csv('ptbxl_from_excel_train.csv')
df_val.to_csv('ptbxl_from_excel_val.csv')
df_test.to_csv('ptbxl_from_excel_test.csv')

print(f"  ✓ ptbxl_from_excel_train.csv ({len(df_train):,} lignes)")
print(f"  ✓ ptbxl_from_excel_val.csv ({len(df_val):,} lignes)")
print(f"  ✓ ptbxl_from_excel_test.csv ({len(df_test):,} lignes)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VÉRIFIER PRÉSENCE FILENAME_LR/HR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Vérification]")

if 'filename_lr' in df_all.columns and 'filename_hr' in df_all.columns:
    print(f"  ✓ filename_lr et filename_hr présents")
    
    # Vérifier chemins
    example_lr = df_all['filename_lr'].iloc[0]
    example_hr = df_all['filename_hr'].iloc[0]
    
    print(f"\n  Exemples:")
    print(f"    • filename_lr: {example_lr}")
    print(f"    • filename_hr: {example_hr}")
    
    # Vérifier existence fichiers
    from pathlib import Path
    file_lr = Path(f"{example_lr}.dat")
    file_hr = Path(f"{example_hr}.dat")
    
    if file_lr.exists():
        print(f"    ✓ Fichier LR existe")
    else:
        print(f"    ✗ Fichier LR n'existe pas")
    
    if file_hr.exists():
        print(f"    ✓ Fichier HR existe")
    else:
        print(f"    ✗ Fichier HR n'existe pas")
else:
    print(f"  ✗ filename_lr/hr manquants!")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("RÉSUMÉ FINAL")
print("=" * 100)

print(f"\n📊 DONNÉES CHARGÉES:")
print(f"  • Total ECG           : {len(df_all):,}")
print(f"  • Train / Val / Test  : {len(df_train):,} / {len(df_val):,} / {len(df_test):,}")
print(f"  • Features totales    : {len(df_all.columns)}")

print(f"\n🏷️  LABELS:")
print(f"  • Codes SCP          : {len(scp_cols)}")
print(f"  • Superclasses       : {len(superclass_cols)}")

print(f"\n📂 FICHIERS GÉNÉRÉS:")
print(f"  • ptbxl_from_excel_consolidated.csv")
print(f"  • ptbxl_from_excel_train/val/test.csv")
print(f"  • label_config_from_excel.json")

print(f"\n✅ STEP 1 TERMINÉ (depuis Excel)")
print(f"   Prochaine étape: step2_signal_cleaning_adapted.py")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# APERÇU DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📋 APERÇU COLONNES:")
print(f"\nFeatures numériques:")
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if not c.startswith('scp_')]
print(f"  {numeric_cols[:15]}")

print(f"\nFeatures catégorielles:")
cat_cols = df_all.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"  {cat_cols[:10]}")

print(f"\nLabels SCP (premiers 10):")
print(f"  {scp_cols[:10]}")

print(f"\nSuperclasses:")
print(f"  {superclass_cols}")
