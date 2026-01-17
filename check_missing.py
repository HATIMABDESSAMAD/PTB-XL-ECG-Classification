import numpy as np
import os
from pathlib import Path

# Charger IDs from all_features_engineered
data_train = np.load('all_features_engineered/W_engineered_train.npz')
ids_train = data_train['ecg_ids']

# Vérifier existence signaux
missing = []
for ecg_id in ids_train:
    path = f'cleaned_signals_100hz/X_clean_{ecg_id}.npz'
    if not os.path.exists(path):
        missing.append(ecg_id)

print(f"Total IDs in all_features_engineered: {len(ids_train)}")
print(f"Missing signals: {len(missing)}")
if missing:
    print(f"First 20 missing IDs: {missing[:20]}")
