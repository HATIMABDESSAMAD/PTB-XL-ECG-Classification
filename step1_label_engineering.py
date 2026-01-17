"""
═══════════════════════════════════════════════════════════════════════════════
STEP 1: LABEL ENGINEERING - PTB-XL Wide+Deep Pipeline
═══════════════════════════════════════════════════════════════════════════════
Transforme scp_codes (dict texte) en colonnes multi-label structurées:
  - y_score__<CODE>  : score original (0-100, 0 si absent)
  - y__<CODE>        : binaire (1 si score > 0, sinon 0)
  - y_SUP__<CLASS>   : 5 superclasses (NORM/MI/STTC/CD/HYP)

Basé sur CinC 2020 + structure PTB-XL officielle
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import json

print("=" * 100)
print("STEP 1: LABEL ENGINEERING - Expansion SCP codes")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Chargement des données...")

# Charger ptbxl_database.csv
df = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
print(f"  ✓ ptbxl_database.csv chargé: {len(df):,} enregistrements")

# Charger scp_statements.csv (mapping CODE → diagnostic_class)
scp_df = pd.read_csv('scp_statements.csv', index_col=0)
print(f"  ✓ scp_statements.csv chargé: {len(scp_df)} codes SCP")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARSING SCP_CODES (dict texte → dict Python)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Parsing scp_codes (texte → dict)...")

def safe_parse_scp_codes(scp_str):
    """Parse scp_codes avec gestion NaN"""
    if pd.isna(scp_str):
        return {}
    try:
        return ast.literal_eval(scp_str)
    except:
        return {}

df['scp_codes_dict'] = df['scp_codes'].apply(safe_parse_scp_codes)
print(f"  ✓ Parsing terminé")

# Exemple
sample_codes = df['scp_codes_dict'].iloc[0]
print(f"  Exemple ECG {df.index[0]}: {sample_codes}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IDENTIFIER TOUS LES CODES UNIQUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Identification des codes SCP uniques...")

all_codes = set()
for codes_dict in df['scp_codes_dict']:
    all_codes.update(codes_dict.keys())

all_codes = sorted(list(all_codes))
print(f"  ✓ {len(all_codes)} codes SCP uniques détectés")
print(f"  Exemples: {all_codes[:10]}")

# Filtrer codes présents dans scp_statements (optionnel: garder seulement les officiels)
valid_codes = [c for c in all_codes if c in scp_df.index]
print(f"  ✓ {len(valid_codes)} codes validés dans scp_statements.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CRÉER COLONNES y_score__<CODE> et y__<CODE>
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Création colonnes multi-label (71 codes)...")

# Colonnes de scores
for code in valid_codes:
    col_score = f'y_score__{code}'
    col_binary = f'y__{code}'
    
    # Score: extraire du dict (0 si absent)
    df[col_score] = df['scp_codes_dict'].apply(lambda d: d.get(code, 0))
    
    # Binaire: 1 si score > 0
    df[col_binary] = (df[col_score] > 0).astype(int)

print(f"  ✓ {len(valid_codes)} × 2 = {len(valid_codes)*2} colonnes créées")
print(f"    • y_score__<CODE> : scores 0-100")
print(f"    • y__<CODE>       : binaire 0/1")

# Statistiques
binary_cols = [f'y__{c}' for c in valid_codes]
prevalences = df[binary_cols].sum().sort_values(ascending=False)
print(f"\n  Top 10 codes les plus fréquents:")
for code in prevalences.head(10).index:
    code_name = code.replace('y__', '')
    count = prevalences[code]
    pct = (count / len(df)) * 100
    description = scp_df.loc[code_name, 'description'] if code_name in scp_df.index else "N/A"
    print(f"    {code_name:5s} : {count:5d} ({pct:5.2f}%) - {description}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CRÉER 5 SUPERCLASSES (NORM/MI/STTC/CD/HYP)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Création 5 superclasses...")

# Définir mapping diagnostic_class → superclass
superclass_mapping = {
    'NORM': 'NORM',   # Normal ECG
    'MI': 'MI',       # Myocardial Infarction
    'STTC': 'STTC',   # ST/T Change
    'CD': 'CD',       # Conduction Disturbance
    'HYP': 'HYP'      # Hypertrophy
}

# Pour chaque superclass, faire OR logique des codes correspondants
for superclass in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
    col_name = f'y_SUP__{superclass}'
    
    # Trouver codes appartenant à cette superclass
    codes_in_class = []
    for code in valid_codes:
        if code in scp_df.index:
            diag_class = scp_df.loc[code, 'diagnostic_class']
            if diag_class == superclass:
                codes_in_class.append(code)
    
    # OR logique: 1 si au moins un code de cette classe est présent
    if len(codes_in_class) > 0:
        binary_cols_class = [f'y__{c}' for c in codes_in_class]
        df[col_name] = df[binary_cols_class].max(axis=1)
    else:
        df[col_name] = 0
    
    count = df[col_name].sum()
    pct = (count / len(df)) * 100
    print(f"  {superclass:5s} : {count:5d} ({pct:5.2f}%) - {len(codes_in_class)} codes")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDER DATASET AVEC LABELS EXPANDED
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Sauvegarde dataset avec labels expandés...")

# Colonnes à garder
metadata_cols = ['patient_id', 'age', 'sex', 'height', 'weight', 
                 'nurse', 'site', 'device', 'recording_date',
                 'report', 'scp_codes', 'heart_axis',
                 'infarction_stadium1', 'infarction_stadium2',
                 'validated_by', 'second_opinion',
                 'initial_autogenerated_report',
                 'validated_by_human', 'baseline_drift', 'static_noise',
                 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker',
                 'strat_fold', 'filename_lr', 'filename_hr']

# Garder colonnes existantes + labels créés
score_cols = [f'y_score__{c}' for c in valid_codes]
binary_cols = [f'y__{c}' for c in valid_codes]
superclass_cols = [f'y_SUP__{sc}' for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']]

# Vérifier que les colonnes existent
metadata_cols_existing = [c for c in metadata_cols if c in df.columns]
all_cols = metadata_cols_existing + score_cols + binary_cols + superclass_cols

df_final = df[all_cols].copy()

# Sauvegarder
output_path = 'ptbxl_with_labels_expanded.csv'
df_final.to_csv(output_path)
print(f"  ✓ Sauvegardé: {output_path}")
print(f"    • Lignes    : {len(df_final):,}")
print(f"    • Colonnes  : {len(df_final.columns)}")
print(f"      - Metadata: {len(metadata_cols_existing)}")
print(f"      - Scores  : {len(score_cols)}")
print(f"      - Binaires: {len(binary_cols)}")
print(f"      - Superclass: {len(superclass_cols)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. STATISTIQUES FINALES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STATISTIQUES FINALES")
print("=" * 100)

print(f"\n📊 DISTRIBUTION LABELS:")
print(f"  • Codes SCP validés        : {len(valid_codes)}")
print(f"  • Enregistrements total    : {len(df_final):,}")
print(f"  • Labels binaires moyens/ECG: {df_final[binary_cols].sum(axis=1).mean():.2f}")

print(f"\n🏷️  SUPERCLASSES (distribution):")
for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
    col = f'y_SUP__{sc}'
    count = df_final[col].sum()
    pct = (count / len(df_final)) * 100
    print(f"  {sc:5s} : {count:5d} ({pct:5.2f}%)")

print(f"\n📂 FICHIERS GÉNÉRÉS:")
print(f"  • {output_path}")

print(f"\n✅ STEP 1 TERMINÉ")
print(f"   Prochaine étape: step2_signal_cleaning.py (nettoyage signaux)")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT LISTE CODES POUR MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════
# Sauvegarder liste codes pour utilisation dans modèle
label_config = {
    'valid_codes': valid_codes,
    'binary_cols': binary_cols,
    'score_cols': score_cols,
    'superclass_cols': superclass_cols,
    'n_labels_71': len(valid_codes),
    'n_labels_5': len(superclass_cols)
}

with open('label_config.json', 'w') as f:
    json.dump(label_config, f, indent=2)
print(f"\n💾 Configuration labels sauvegardée: label_config.json")
