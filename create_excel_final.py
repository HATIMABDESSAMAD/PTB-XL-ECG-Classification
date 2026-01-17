import pandas as pd

print("Création de l'Excel consolidé avec filename_lr et filename_hr...")
filename = 'PTB_XL_ML_Features_FINAL.xlsx'
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    pd.read_csv('ptbxl_ml_features_train.csv').to_excel(writer, sheet_name='Train', index=False)
    pd.read_csv('ptbxl_ml_features_val.csv').to_excel(writer, sheet_name='Val', index=False)
    pd.read_csv('ptbxl_ml_features_test.csv').to_excel(writer, sheet_name='Test', index=False)
print(f"✓ Excel créé: {filename}")
print("✓ Contient filename_lr et filename_hr pour lier aux signaux ECG!")
