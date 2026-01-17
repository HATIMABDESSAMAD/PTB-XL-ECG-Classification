"""
Script pour regrouper les 3 fichiers ML features dans un fichier Excel
Chaque fichier sera dans une feuille séparée (Train, Val, Test)
"""

import pandas as pd
from datetime import datetime

print("=" * 70)
print("REGROUPEMENT DES FICHIERS ML FEATURES DANS EXCEL")
print("=" * 70)

# Noms des fichiers
files = {
    'Train': 'ptbxl_ml_features_train.csv',
    'Val': 'ptbxl_ml_features_val.csv',
    'Test': 'ptbxl_ml_features_test.csv'
}

# Nom du fichier Excel de sortie
output_file = 'PTB_XL_ML_Features_Complete.xlsx'

try:
    # Créer un writer Excel
    print(f"\n➤ Création du fichier Excel: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        for sheet_name, csv_file in files.items():
            print(f"\n  • Chargement de {csv_file}...")
            df = pd.read_csv(csv_file)
            
            print(f"    ✓ {len(df):,} lignes × {len(df.columns)} colonnes")
            
            # Écrire dans la feuille Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"    ✓ Feuille '{sheet_name}' créée")
    
    print("\n" + "=" * 70)
    print("✓ FICHIER EXCEL CRÉÉ AVEC SUCCÈS !")
    print("=" * 70)
    
    print(f"\nFichier: {output_file}")
    print("Contient 3 feuilles:")
    print("  1. Train  - 17,182 enregistrements")
    print("  2. Val    -  2,137 enregistrements")
    print("  3. Test   -  2,162 enregistrements")
    print(f"\nDate création: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERREUR: Fichier non trouvé - {e}")
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
