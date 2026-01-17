import numpy as np
import pandas as pd

# Charger données
df_excel = pd.read_csv('ptbxl_from_excel_consolidated.csv')

# Exclure colonnes
exclude_cols = ['ecg_id', 'patient_id', 'filename_lr', 'filename_hr', 
                'report', 'strat_fold'] + \
               [f'scp_{x}' for x in ['SR', 'NORM', 'ABQRS', 'IMI', 'ASMI', 
                                     'LVH', 'NDT', 'LAFB', 'AFIB', 'ISC_', 
                                     'PVC', 'IRBBB', 'STD_', 'VCLVH', 'STACH', 
                                     'IVCD', '1AVB', 'SARRH']]

# Features Excel
excel_cols = [c for c in df_excel.columns 
              if c not in exclude_cols and 'scp_superclass' not in c]

print(f"Nombre total: {len(excel_cols)}")
print("\n32 Features Excel utilisées dans le modèle PURE:")
print("="*60)

# Grouper par catégorie
demographiques = [c for c in excel_cols if c in ['age', 'sex', 'height', 'weight', 'bmi']]
temporelles = [c for c in excel_cols if c in ['year', 'month', 'quarter', 'day_of_week']]
qualite = [c for c in excel_cols if 'quality' in c or c in ['baseline_drift', 'static_noise', 'extra_beats']]
metadata = [c for c in excel_cols if 'encoded' in c or c in ['is_validated', 'has_second_opinion', 'num_scp_codes']]

print("\n📊 CATÉGORIE 1: DÉMOGRAPHIQUES (5 features)")
for i, col in enumerate(demographiques, 1):
    print(f"  {i}. {col}")

print("\n📅 CATÉGORIE 2: TEMPORELLES (4 features)")
for i, col in enumerate(temporelles, 1):
    print(f"  {i}. {col}")

print("\n✅ CATÉGORIE 3: QUALITÉ SIGNAL (6 features)")
for i, col in enumerate(qualite, 1):
    print(f"  {i}. {col}")

print("\n🔧 CATÉGORIE 4: MÉTADONNÉES (3 features)")
for i, col in enumerate(metadata, 1):
    print(f"  {i}. {col}")

autres = [c for c in excel_cols 
          if c not in demographiques + temporelles + qualite + metadata]
if autres:
    print(f"\n📁 CATÉGORIE 5: AUTRES ({len(autres)} features)")
    for i, col in enumerate(autres, 1):
        print(f"  {i}. {col}")

print("\n" + "="*60)
print(f"TOTAL: {len(excel_cols)} features Excel")
