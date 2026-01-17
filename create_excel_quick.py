import pandas as pd

print("Création de l'Excel consolidé...")
with pd.ExcelWriter('PTB_XL_ML_Features_Complete.xlsx', engine='openpyxl') as writer:
    pd.read_csv('ptbxl_ml_features_train.csv').to_excel(writer, sheet_name='Train', index=False)
    pd.read_csv('ptbxl_ml_features_val.csv').to_excel(writer, sheet_name='Val', index=False)
    pd.read_csv('ptbxl_ml_features_test.csv').to_excel(writer, sheet_name='Test', index=False)
print("✓ Excel créé avec succès!")
