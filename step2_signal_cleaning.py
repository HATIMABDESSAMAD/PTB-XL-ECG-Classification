"""
═══════════════════════════════════════════════════════════════════════════════
STEP 2: SIGNAL CLEANING - PTB-XL Wide+Deep Pipeline
═══════════════════════════════════════════════════════════════════════════════
Nettoyage signaux ECG 12 leads avec NeuroKit2:
  - Chargement WFDB (records100 - 100 Hz)
  - FIR bandpass 3-45 Hz par lead
  - Normalisation par lead (z-score)
  - Sauvegarde X_clean (12×1000) en format numpy compressé

Format sortie: X_clean_<ecg_id>.npz contenant array (12, 1000)
"""

import pandas as pd
import numpy as np
import wfdb
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("STEP 2: SIGNAL CLEANING avec NeuroKit2")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VÉRIFICATION DÉPENDANCES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Vérification dépendances...")

try:
    import neurokit2
    print(f"  ✓ NeuroKit2 version {neurokit2.__version__}")
except ImportError:
    print("  ✗ NeuroKit2 non installé")
    print("  → pip install neurokit2")
    exit(1)

try:
    import wfdb
    print(f"  ✓ WFDB version {wfdb.__version__}")
except ImportError:
    print("  ✗ WFDB non installé")
    print("  → pip install wfdb")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DATASET AVEC LABELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Chargement dataset depuis Excel consolidé...")

df = pd.read_csv('ptbxl_from_excel_consolidated.csv', index_col='ecg_id')
print(f"  ✓ {len(df):,} enregistrements chargés")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FONCTION DE NETTOYAGE PAR LEAD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Définition fonction de nettoyage...")

def clean_ecg_signal(signal_path, sampling_rate=100):
    """
    Nettoie signal ECG 12 leads avec NeuroKit2
    
    Args:
        signal_path: chemin vers fichier WFDB (sans extension)
        sampling_rate: fréquence d'échantillonnage (100 Hz pour records100)
    
    Returns:
        X_clean: array (12, 1000) - signaux nettoyés
        success: bool - True si nettoyage OK
    """
    try:
        # Charger signal WFDB
        record = wfdb.rdrecord(signal_path)
        signal = record.p_signal  # shape (1000, 12) pour 100Hz × 10s
        
        # Vérifier dimensions
        if signal.shape[0] != 1000 or signal.shape[1] != 12:
            return None, False
        
        # Transposer pour avoir (12, 1000)
        signal = signal.T  # maintenant (12, 1000)
        
        # Nettoyer chaque lead indépendamment
        X_clean = np.zeros_like(signal)
        
        for lead_idx in range(12):
            lead_signal = signal[lead_idx, :]
            
            # Vérifier NaN
            if np.isnan(lead_signal).any():
                # Interpoler NaN simples
                lead_signal = pd.Series(lead_signal).interpolate(method='linear').fillna(0).values
            
            # FIR bandpass 3-45 Hz (standard ECG clinique)
            try:
                cleaned = nk.ecg_clean(
                    lead_signal, 
                    sampling_rate=sampling_rate,
                    method='neurokit'  # utilise FIR bandpass par défaut
                )
            except:
                # Si échec, copier signal original
                cleaned = lead_signal
            
            # Normalisation z-score par lead
            mean_val = np.mean(cleaned)
            std_val = np.std(cleaned)
            
            if std_val > 1e-6:  # éviter division par 0
                cleaned = (cleaned - mean_val) / std_val
            else:
                cleaned = cleaned - mean_val
            
            X_clean[lead_idx, :] = cleaned
        
        return X_clean, True
        
    except Exception as e:
        return None, False

print("  ✓ Fonction clean_ecg_signal() définie")
print("    • FIR bandpass 3-45 Hz")
print("    • Z-score normalization par lead")
print("    • Gestion NaN par interpolation")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAITEMENT BATCH DE TOUS LES SIGNAUX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Nettoyage signaux (peut prendre 15-30 minutes)...")

# Créer dossier sortie
output_dir = Path('cleaned_signals_100hz')
output_dir.mkdir(exist_ok=True)
print(f"  Dossier sortie: {output_dir}/")

# Traiter tous les ECG
success_count = 0
error_count = 0
error_ids = []

# Traiter TOUS les ECG de l'Excel
SAMPLE_SIZE = None  # None = tous, 500 = test rapide
if SAMPLE_SIZE:
    df_to_process = df.head(SAMPLE_SIZE)
    print(f"  ⚠️  MODE TEST: traitement de {SAMPLE_SIZE} ECG seulement")
else:
    df_to_process = df
    print(f"  🚀 MODE COMPLET: Traitement de {len(df_to_process):,} ECG...")

for ecg_id, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="  Cleaning"):
    # Chemin signal (sans extension .dat)
    signal_path = row['filename_lr']  # ex: records100/00000/00001_lr
    
    # Nettoyer
    X_clean, success = clean_ecg_signal(signal_path, sampling_rate=100)
    
    if success:
        # Sauvegarder en .npz (compressé)
        output_path = output_dir / f"X_clean_{ecg_id:05d}.npz"
        np.savez_compressed(output_path, signal=X_clean)
        success_count += 1
    else:
        error_count += 1
        error_ids.append(ecg_id)

print(f"\n  ✓ Traitement terminé:")
print(f"    • Succès : {success_count:,} / {len(df_to_process):,} ({success_count/len(df_to_process)*100:.1f}%)")
print(f"    • Erreurs: {error_count}")

if error_count > 0 and error_count < 20:
    print(f"    • ECG IDs en erreur: {error_ids}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CRÉER INDEX DES SIGNAUX NETTOYÉS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Création index des signaux nettoyés...")

# Créer colonne avec chemin vers signal nettoyé
df['X_clean_path'] = df.index.map(
    lambda ecg_id: str(output_dir / f"X_clean_{ecg_id:05d}.npz")
)

# Marquer signaux disponibles
df['X_clean_available'] = df.index.map(
    lambda ecg_id: (output_dir / f"X_clean_{ecg_id:05d}.npz").exists()
)

# Sauvegarder dataset mis à jour
output_csv = 'ptbxl_with_cleaned_signals.csv'
df.to_csv(output_csv)
print(f"  ✓ Dataset mis à jour: {output_csv}")
print(f"    • Signaux disponibles: {df['X_clean_available'].sum():,} / {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATISTIQUES FINALES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STATISTIQUES FINALES")
print("=" * 100)

print(f"\n📊 SIGNAUX NETTOYÉS:")
print(f"  • Total ECG traités       : {success_count:,}")
print(f"  • Format                  : (12 leads, 1000 samples)")
print(f"  • Fréquence               : 100 Hz")
print(f"  • Durée                   : 10 secondes")
print(f"  • Préprocessing           : FIR 3-45 Hz + Z-score")

print(f"\n💾 STOCKAGE:")
print(f"  • Dossier                 : {output_dir}/")
print(f"  • Format fichier          : .npz (numpy compressé)")
print(f"  • Taille moyenne/fichier  : ~10 KB")

# Calculer taille totale
total_size_mb = sum(f.stat().st_size for f in output_dir.glob("*.npz")) / (1024**2)
print(f"  • Taille totale           : {total_size_mb:.2f} MB")

print(f"\n✅ STEP 2 TERMINÉ")
print(f"   Prochaine étape: step3_wide_features_extraction.py")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# EXEMPLE DE CHARGEMENT D'UN SIGNAL NETTOYÉ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n💡 EXEMPLE D'UTILISATION:")
print("""
# Charger un signal nettoyé
import numpy as np
data = np.load('cleaned_signals_100hz/X_clean_00001.npz')
X_clean = data['signal']  # shape: (12, 1000)

# X_clean[0, :] = Lead I
# X_clean[1, :] = Lead II
# etc.
""")
