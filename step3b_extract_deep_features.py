"""
=================================================================
STEP 3B: EXTRACTION FEATURES DEEP (CNN + TRANSFORMER)
=================================================================

Extrait les 64 features latentes du DeepBranch (CNN + Transformer)
pour chaque ECG en utilisant le modèle pré-entraîné.

INPUT:
  • cleaned_signals_100hz/*.npz (21,481 signaux)
  • models/best_model.pth (modèle entraîné)

OUTPUT:
  • deep_features/deep_features_train.npz (64 features × 17,182)
  • deep_features/deep_features_val.npz (64 features × 2,137)
  • deep_features/deep_features_test.npz (64 features × 2,162)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import modèle
from step5_wide_deep_model import DeepBranch

print("=" * 100)
print("STEP 3B: EXTRACTION FEATURES DEEP (CNN + Transformer)")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SIGNALS_DIR = Path('cleaned_signals_100hz')
MODEL_PATH = 'models/best_model.pth'
LABELS_FILE = 'ptbxl_from_excel_consolidated.csv'
OUTPUT_DIR = Path('deep_features')
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

print(f"\n⚙️  CONFIGURATION:")
print(f"  • Device: {DEVICE}")
print(f"  • Batch size: {BATCH_SIZE}")
print(f"  • Output: {OUTPUT_DIR}/")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[1/4] Chargement modèle pré-entraîné...")

# Créer DeepBranch
deep_branch = DeepBranch(
    n_leads=12,
    seq_len=1000,
    d_model=256,
    transformer_heads=8,
    transformer_layers=8,
    deep_features_dim=64,
    dropout=0.1
).to(DEVICE)

# Charger poids depuis modèle complet
if Path(MODEL_PATH).exists():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Extraire uniquement les poids du DeepBranch
    deep_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('deep_branch.'):
            # Retirer le préfixe 'deep_branch.'
            new_key = k.replace('deep_branch.', '')
            deep_state_dict[new_key] = v
    
    deep_branch.load_state_dict(deep_state_dict)
    print(f"  ✓ Modèle chargé depuis {MODEL_PATH}")
else:
    print(f"  ⚠️  Modèle non trouvé, utilisation modèle non-entraîné")

deep_branch.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[2/4] Chargement dataset...")

df = pd.read_csv(LABELS_FILE, index_col='ecg_id')

# Séparer par strat_fold
df_train = df[df['strat_fold'] <= 8].copy()
df_val = df[df['strat_fold'] == 9].copy()
df_test = df[df['strat_fold'] == 10].copy()

print(f"  ✓ Train: {len(df_train)} ECG")
print(f"  ✓ Val  : {len(df_val)} ECG")
print(f"  ✓ Test : {len(df_test)} ECG")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXTRACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_batch(ecg_ids, batch_size=32):
    """Extrait features Deep pour un batch d'ECG IDs"""
    all_features = []
    valid_ids = []
    
    # Traiter par batches
    for i in tqdm(range(0, len(ecg_ids), batch_size), desc="  Extracting"):
        batch_ids = ecg_ids[i:i+batch_size]
        batch_signals = []
        batch_valid_ids = []
        
        # Charger signaux du batch
        for ecg_id in batch_ids:
            signal_path = SIGNALS_DIR / f"X_clean_{ecg_id:05d}.npz"
            try:
                data = np.load(signal_path)
                signal = data['signal']  # (12, 1000)
                batch_signals.append(signal)
                batch_valid_ids.append(ecg_id)
            except:
                # Signal manquant
                continue
        
        if len(batch_signals) == 0:
            continue
        
        # Convertir en tensor
        X_batch = torch.from_numpy(np.array(batch_signals)).float().to(DEVICE)
        
        # Extraire features
        with torch.no_grad():
            deep_feats = deep_branch(X_batch)  # (batch, 64)
        
        all_features.append(deep_feats.cpu().numpy())
        valid_ids.extend(batch_valid_ids)
    
    # Concaténer tous les batches
    if len(all_features) > 0:
        features = np.vstack(all_features)
        return features, np.array(valid_ids)
    else:
        return np.array([]), np.array([])


print(f"\n[3/4] Extraction features Deep...")

# Train
print(f"\n  Train:")
deep_train, ids_train = extract_features_batch(df_train.index.values, BATCH_SIZE)
print(f"    ✓ Shape: {deep_train.shape}")

# Val
print(f"\n  Val:")
deep_val, ids_val = extract_features_batch(df_val.index.values, BATCH_SIZE)
print(f"    ✓ Shape: {deep_val.shape}")

# Test
print(f"\n  Test:")
deep_test, ids_test = extract_features_batch(df_test.index.values, BATCH_SIZE)
print(f"    ✓ Shape: {deep_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/4] Sauvegarde...")

np.savez_compressed(
    OUTPUT_DIR / 'deep_features_train.npz',
    features=deep_train,
    ecg_ids=ids_train
)

np.savez_compressed(
    OUTPUT_DIR / 'deep_features_val.npz',
    features=deep_val,
    ecg_ids=ids_val
)

np.savez_compressed(
    OUTPUT_DIR / 'deep_features_test.npz',
    features=deep_test,
    ecg_ids=ids_test
)

print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/deep_features_train.npz")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/deep_features_val.npz")
print(f"  ✓ Sauvegardé: {OUTPUT_DIR}/deep_features_test.npz")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("STATISTIQUES FINALES")
print("=" * 100)

print(f"\n📊 FEATURES DEEP EXTRAITES:")
print(f"  • Train: {deep_train.shape[0]} ECG × {deep_train.shape[1]} features")
print(f"  • Val  : {deep_val.shape[0]} ECG × {deep_val.shape[1]} features")
print(f"  • Test : {deep_test.shape[0]} ECG × {deep_test.shape[1]} features")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"  • {OUTPUT_DIR}/deep_features_train.npz")
print(f"  • {OUTPUT_DIR}/deep_features_val.npz")
print(f"  • {OUTPUT_DIR}/deep_features_test.npz")

print(f"\n✅ STEP 3B TERMINÉ")
print(f"   Prochaine étape: step4b_merge_all_features.py")
print("=" * 100)
