import pandas as pd
import sys

print("=" * 70)
print("VÉRIFICATION ET CRÉATION EXCEL AVEC filename_lr/filename_hr")
print("=" * 70)

# Vérifier les colonnes dans les CSV
print("\n1. Vérification des colonnes dans ptbxl_ml_features_train.csv...")
df_train = pd.read_csv('ptbxl_ml_features_train.csv')
print(f"   Total colonnes: {len(df_train.columns)}")

# Vérifier présence de filename_lr et filename_hr
if 'filename_lr' in df_train.columns and 'filename_hr' in df_train.columns:
    print("   ✓ filename_lr : PRÉSENT")
    print("   ✓ filename_hr : PRÉSENT")
    print(f"\n   Exemple:")
    print(f"   - filename_lr: {df_train['filename_lr'].iloc[0]}")
    print(f"   - filename_hr: {df_train['filename_hr'].iloc[0]}")
else:
    print("   ✗ ERREUR: Colonnes filename manquantes!")
    sys.exit(1)

# Créer Excel
print("\n2. Création du fichier Excel...")
filename = 'PTB_XL_ML_Features_WITH_FILENAMES.xlsx'

try:
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        print("   - Chargement Train...")
        df_train.to_excel(writer, sheet_name='Train', index=False)
        
        print("   - Chargement Val...")
        df_val = pd.read_csv('ptbxl_ml_features_val.csv')
        df_val.to_excel(writer, sheet_name='Val', index=False)
        
        print("   - Chargement Test...")
        df_test = pd.read_csv('ptbxl_ml_features_test.csv')
        df_test.to_excel(writer, sheet_name='Test', index=False)
    
    print(f"\n✓ SUCCÈS: {filename} créé!")
    print(f"  - Train: {len(df_train)} lignes × {len(df_train.columns)} colonnes")
    print(f"  - Val: {len(df_val)} lignes × {len(df_val.columns)} colonnes")
    print(f"  - Test: {len(df_test)} lignes × {len(df_test.columns)} colonnes")
    print(f"\n✓ Les colonnes filename_lr et filename_hr sont INCLUSES!")
    
except PermissionError:
    print(f"\n✗ ERREUR: Le fichier {filename} est ouvert dans Excel!")
    print("  → Fermez Excel et réessayez.")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERREUR: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
