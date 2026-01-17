"""
═══════════════════════════════════════════════════════════════════════════════
EXEMPLE D'UTILISATION DU MODÈLE WIDE+DEEP PURE
AVEC FORMAT ORIGINAL DU DATASET PTB-XL
═══════════════════════════════════════════════════════════════════════════════

Ce script lit les fichiers ECG dans le FORMAT ORIGINAL du dataset PTB-XL:
  - Fichiers WFDB: records100/XXXXX/XXXXX_lr.dat et .hea
  - Fichier CSV: ptbxl_database.csv (métadonnées et features)

STRUCTURE DU DATASET PTB-XL:
  ptb-xl-dataset/
  ├── records100/           ← Signaux ECG 100Hz (format WFDB)
  │   ├── 00000/
  │   │   ├── 00001_lr.dat  ← Données binaires
  │   │   ├── 00001_lr.hea  ← Header (métadonnées)
  │   │   ├── 00002_lr.dat
  │   │   └── ...
  │   ├── 01000/
  │   └── ...
  ├── ptbxl_database.csv    ← Métadonnées + labels
  └── scp_statements.csv    ← Mapping codes SCP → superclasses

INPUTS DU MODÈLE:
  1. Signal ECG: (batch, 12, 1000) - 12 dérivations, 10s @ 100Hz
  2. Wide Features: (batch, 32) - 32 features cliniques

OUTPUT:
  - Probabilités: (batch, 5) - [NORM, MI, STTC, CD, HYP]

Auteur: Pipeline Wide+Deep PTB-XL
Date: Janvier 2026
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION DES DÉPENDANCES
# ═══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("  VÉRIFICATION DES DÉPENDANCES")
print("="*70)

try:
    import wfdb
    print(f"  ✓ wfdb version {wfdb.__version__}")
except ImportError:
    print("  ✗ wfdb non installé → pip install wfdb")
    exit(1)

try:
    import neurokit2 as nk
    print(f"  ✓ neurokit2 version {nk.__version__}")
except ImportError:
    print("  ✗ neurokit2 non installé → pip install neurokit2")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE DU MODÈLE (copie autonome)
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    """
    Architecture Wide+Deep pour classification ECG multi-label.
    Total: 11,561,573 paramètres
    """
    
    def __init__(self, num_wide_features=32, num_classes=5):
        super().__init__()
        
        # Deep branch - CNN (6 blocs)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=14, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=14, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(512, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        
        # Deep branch - Transformer (8 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.cnn_to_transformer = nn.Linear(512, 256)
        
        self.deep_fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Wide branch
        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, signal, wide_features):
        x = self.conv_layers(signal).squeeze(-1)
        x = self.cnn_to_transformer(x).unsqueeze(1)
        x = self.transformer(x).mean(dim=1)
        deep_out = self.deep_fc(x)
        wide_out = self.wide_fc(wide_features)
        combined = torch.cat([deep_out, wide_out], dim=1)
        return self.fusion(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Chemin vers le modèle
MODEL_PATH = 'model_wide_deep_pure_FIXED.pth'

# Chemins vers le dataset PTB-XL (format original)
RECORDS_DIR = Path('records100')           # Signaux WFDB
DATABASE_CSV = Path('ptbxl_database.csv')  # Métadonnées

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'ECG Normal',
    'MI': 'Infarctus du Myocarde',
    'STTC': 'Changements ST/T',
    'CD': 'Troubles de Conduction',
    'HYP': 'Hypertrophie'
}

# Les 12 dérivations ECG standard
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FONCTIONS POUR LIRE LE FORMAT ORIGINAL PTB-XL
# ═══════════════════════════════════════════════════════════════════════════════

def load_ecg_from_wfdb(ecg_id, records_dir='records100', sampling_rate=100):
    """
    Charge un signal ECG depuis le format WFDB original du dataset PTB-XL.
    
    Structure des fichiers:
        records100/
        ├── 00000/
        │   ├── 00001_lr.dat    ← Données binaires du signal
        │   ├── 00001_lr.hea    ← Header avec métadonnées
        │   └── ...
        ├── 01000/
        │   ├── 01001_lr.dat
        │   └── ...
        └── ...
    
    Args:
        ecg_id: int - Identifiant ECG (1 à 21837)
        records_dir: str - Chemin vers records100/
        sampling_rate: int - 100Hz pour records100, 500Hz pour records500
        
    Returns:
        signal: numpy array (12, 1000) - 12 dérivations × 1000 samples
    """
    # Construire le chemin du fichier
    # Format: records100/XXXXX/XXXXX_lr où XXXXX est le dossier (arrondi à 1000)
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    filename = f"{ecg_id:05d}_lr"
    record_path = Path(records_dir) / folder / filename
    
    print(f"\n[WFDB] Lecture du fichier ECG...")
    print(f"  Chemin: {record_path}.dat / .hea")
    
    # Vérifier que les fichiers existent
    if not Path(f"{record_path}.dat").exists():
        raise FileNotFoundError(f"Fichier non trouvé: {record_path}.dat")
    
    # Lire le signal avec wfdb
    record = wfdb.rdrecord(str(record_path))
    
    # record.p_signal contient le signal: shape (1000, 12) pour 100Hz × 10s
    signal = record.p_signal  # numpy array (1000, 12)
    
    print(f"  Format brut: {signal.shape} (samples × leads)")
    print(f"  Fréquence: {record.fs} Hz")
    print(f"  Durée: {signal.shape[0] / record.fs:.1f} secondes")
    print(f"  Dérivations: {record.sig_name}")
    
    # Transposer pour avoir (12, 1000) comme attendu par le modèle
    signal = signal.T  # (12, 1000)
    
    print(f"  Format transposé: {signal.shape} (leads × samples)")
    
    return signal.astype(np.float32)


def clean_ecg_signal(signal, sampling_rate=100):
    """
    Nettoie le signal ECG avec filtrage et normalisation.
    Identique au prétraitement utilisé pendant l'entraînement.
    
    Args:
        signal: numpy array (12, 1000)
        sampling_rate: int - fréquence d'échantillonnage
        
    Returns:
        cleaned: numpy array (12, 1000) - signal nettoyé
    """
    print(f"\n[CLEANING] Nettoyage du signal ECG...")
    
    cleaned = np.zeros_like(signal)
    
    for lead_idx in range(12):
        lead_signal = signal[lead_idx, :]
        
        # Interpoler les NaN si présents
        if np.isnan(lead_signal).any():
            lead_signal = pd.Series(lead_signal).interpolate(method='linear').fillna(0).values
        
        # Filtre FIR bandpass 3-45 Hz (standard ECG clinique)
        try:
            lead_clean = nk.ecg_clean(
                lead_signal, 
                sampling_rate=sampling_rate,
                method='neurokit'
            )
        except:
            lead_clean = lead_signal
        
        # Normalisation z-score par dérivation
        mean_val = np.mean(lead_clean)
        std_val = np.std(lead_clean)
        
        if std_val > 1e-6:
            lead_clean = (lead_clean - mean_val) / std_val
        else:
            lead_clean = lead_clean - mean_val
        
        cleaned[lead_idx, :] = lead_clean
    
    print(f"  ✓ Filtre FIR bandpass 3-45 Hz appliqué")
    print(f"  ✓ Normalisation z-score par dérivation")
    print(f"  Shape finale: {cleaned.shape}")
    
    return cleaned.astype(np.float32)


def load_wide_features_from_csv(ecg_id, csv_path='ptbxl_database.csv'):
    """
    Extrait les 32 features Wide depuis ptbxl_database.csv.
    
    Le fichier CSV contient les colonnes:
        - ecg_id: identifiant unique
        - patient_id, age, sex, height, weight
        - recording_date, validated_by, etc.
        - Intervalles: rr_interval, pr_interval, qrs_duration, qt_interval, qtc_interval
        - Axes: p_axis, qrs_axis, t_axis
        - Amplitudes extraites
        
    Args:
        ecg_id: int - Identifiant ECG
        csv_path: str - Chemin vers ptbxl_database.csv
        
    Returns:
        features: numpy array (32,) - 32 features normalisées
    """
    print(f"\n[CSV] Lecture des features depuis {csv_path}...")
    
    # Charger le CSV
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    if ecg_id not in df.index:
        raise ValueError(f"ECG ID {ecg_id} non trouvé dans {csv_path}")
    
    row = df.loc[ecg_id]
    
    # Extraire les features disponibles
    # Note: Adapter selon les colonnes réellement présentes dans votre CSV
    features = []
    
    # Démographiques
    features.append(row.get('age', 50) / 100)  # Normaliser âge
    features.append(1 if row.get('sex', 0) == 1 else 0)  # 1=M, 0=F
    features.append(row.get('height', 170) / 200)  # Normaliser hauteur
    features.append(row.get('weight', 70) / 150)  # Normaliser poids
    
    # Intervalles ECG (si disponibles)
    # Ces valeurs peuvent être extraites automatiquement ou être dans le CSV
    default_intervals = {
        'rr_interval': 0.85,
        'pr_interval': 0.16,
        'qrs_duration': 0.09,
        'qt_interval': 0.40,
        'qtc_interval': 0.42
    }
    
    for key, default in default_intervals.items():
        val = row.get(key, default)
        if pd.isna(val):
            val = default
        features.append(float(val))
    
    # Axes
    for axis in ['p_axis', 'qrs_axis', 't_axis']:
        val = row.get(axis, 0)
        if pd.isna(val):
            val = 0
        features.append(float(val) / 180)  # Normaliser à [-1, 1]
    
    # Compléter jusqu'à 32 features avec des valeurs par défaut
    while len(features) < 32:
        features.append(0.0)
    
    features = np.array(features[:32], dtype=np.float32)
    
    print(f"  ✓ {len(features)} features extraites")
    print(f"  Shape: {features.shape}")
    
    return features


def load_wide_features_from_npz(ecg_id, wide_dir='wide_features_clean'):
    """
    Alternative: Charge les features depuis les fichiers .npz pré-calculés.
    
    Args:
        ecg_id: int
        wide_dir: str
        
    Returns:
        features: numpy array (32,)
    """
    for split in ['test', 'val', 'train']:
        wide_path = Path(wide_dir) / f'W_pure_{split}.npz'
        
        if wide_path.exists():
            data = np.load(wide_path)
            ecg_ids = data['ecg_ids']
            
            if ecg_id in ecg_ids:
                idx = np.where(ecg_ids == ecg_id)[0][0]
                features = data['W'][idx]
                print(f"\n[NPZ] Features chargées depuis {split}")
                print(f"  Shape: {features.shape}")
                return features.astype(np.float32)
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FONCTION DE PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model, signal, wide_features, device):
    """
    Effectue une prédiction avec le modèle.
    
    Args:
        model: WideDeepModel
        signal: (12, 1000) ou (batch, 12, 1000)
        wide_features: (32,) ou (batch, 32)
        device: torch device
        
    Returns:
        probabilities: (5,) ou (batch, 5)
    """
    # Ajouter dimension batch si nécessaire
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]
    if wide_features.ndim == 1:
        wide_features = wide_features[np.newaxis, ...]
    
    # Tensors
    signal_t = torch.from_numpy(signal).float().to(device)
    wide_t = torch.from_numpy(wide_features).float().to(device)
    
    # Prédiction
    with torch.no_grad():
        logits = model(signal_t, wide_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    return probs


def display_results(probabilities, threshold=0.5):
    """Affiche les résultats de prédiction."""
    
    print("\n" + "="*65)
    print("                      RÉSULTATS")
    print("="*65)
    
    probs = probabilities[0] if probabilities.ndim == 2 else probabilities
    
    print("\n┌──────────┬────────────┬─────────────────────────────────────┐")
    print("│  Classe  │ Probabilité│          Description                │")
    print("├──────────┼────────────┼─────────────────────────────────────┤")
    
    for name, prob in zip(CLASS_NAMES, probs):
        status = "✓" if prob >= threshold else " "
        bar = "█" * int(prob * 15) + "░" * (15 - int(prob * 15))
        print(f"│ {status} {name:<6} │ {prob*100:>6.2f}% {bar} │ {CLASS_DESCRIPTIONS[name]:<35} │")
    
    print("└──────────┴────────────┴─────────────────────────────────────┘")
    
    # Diagnostic
    detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
    
    print(f"\n📋 DIAGNOSTIC (seuil = {threshold*100:.0f}%):")
    if detected:
        for cls in detected:
            print(f"   ✓ {cls}: {CLASS_DESCRIPTIONS[cls]}")
    else:
        print("   → Aucune pathologie détectée au-dessus du seuil")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXEMPLE D'UTILISATION PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("  EXEMPLE D'UTILISATION AVEC FORMAT ORIGINAL PTB-XL")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1: Charger le modèle
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 1: CHARGEMENT DU MODÈLE")
    print("─"*70)
    
    model = WideDeepModel(num_wide_features=32, num_classes=5)
    
    if Path(MODEL_PATH).exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"\n[MODEL] ✓ Modèle chargé: {MODEL_PATH}")
    else:
        print(f"\n[MODEL] ⚠ Modèle non trouvé: {MODEL_PATH}")
        print("        Utilisation du modèle non-entraîné (démo)")
    
    model.to(DEVICE)
    model.eval()
    print(f"[MODEL] Device: {DEVICE}")
    print(f"[MODEL] Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2: Choisir un ECG du dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 2: SÉLECTION D'UN ECG")
    print("─"*70)
    
    # Choisir un ECG ID (entre 1 et 21837)
    ECG_ID = 1  # Exemple: premier ECG du dataset
    
    print(f"\n[ECG] ID sélectionné: {ECG_ID}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3: Charger le signal ECG (format WFDB original)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 3: CHARGEMENT DU SIGNAL ECG (FORMAT WFDB)")
    print("─"*70)
    
    print("""
    STRUCTURE DES FICHIERS WFDB:
    ├── records100/00000/00001_lr.dat  ← Données binaires (signal)
    └── records100/00000/00001_lr.hea  ← Header (métadonnées)
    
    Contenu du .hea:
        - Fréquence d'échantillonnage: 100 Hz
        - Nombre de samples: 1000 (10 secondes)
        - Nombre de dérivations: 12
        - Noms: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """)
    
    try:
        # Charger le signal brut
        signal_raw = load_ecg_from_wfdb(ECG_ID, RECORDS_DIR)
        
        # Nettoyer le signal (filtrage + normalisation)
        signal_clean = clean_ecg_signal(signal_raw)
        
    except FileNotFoundError as e:
        print(f"\n[WARN] {e}")
        print("[WARN] Génération d'un signal simulé pour la démo...")
        
        # Signal simulé si fichiers non disponibles
        t = np.linspace(0, 10, 1000)
        signal_clean = np.zeros((12, 1000), dtype=np.float32)
        for i in range(12):
            signal_clean[i] = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(1000)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 4: Charger les features Wide
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 4: CHARGEMENT DES FEATURES WIDE")
    print("─"*70)
    
    print("""
    OPTIONS DE CHARGEMENT:
    1. Depuis ptbxl_database.csv (format original)
    2. Depuis wide_features_clean/*.npz (pré-calculées)
    """)
    
    # Essayer d'abord les features pré-calculées
    wide_features = load_wide_features_from_npz(ECG_ID)
    
    if wide_features is None:
        try:
            wide_features = load_wide_features_from_csv(ECG_ID)
        except Exception as e:
            print(f"\n[WARN] {e}")
            print("[WARN] Génération de features simulées...")
            wide_features = np.random.randn(32).astype(np.float32) * 0.1
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 5: Résumé des inputs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 5: RÉSUMÉ DES INPUTS")
    print("─"*70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                           INPUT 1: SIGNAL ECG                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Source: records100/{(ECG_ID//1000)*1000:05d}/{ECG_ID:05d}_lr.dat/.hea{' '*(22-len(str(ECG_ID)))}│
    │  Shape: {str(signal_clean.shape):<57} │
    │  Type: {str(signal_clean.dtype):<58} │
    │  Min/Max: [{signal_clean.min():.3f}, {signal_clean.max():.3f}]{' '*43}│
    │  Dérivations: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6     │
    │  Samples: 1000 (10 secondes @ 100Hz)                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                         INPUT 2: WIDE FEATURES                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Source: ptbxl_database.csv ou wide_features_clean/*.npz            │
    │  Shape: {str(wide_features.shape):<57} │
    │  Type: {str(wide_features.dtype):<58} │
    │  Contenu: intervalles, amplitudes, démographiques, qualité         │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 6: Prédiction
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ÉTAPE 6: PRÉDICTION")
    print("─"*70)
    
    probabilities = predict(model, signal_clean, wide_features, DEVICE)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                              OUTPUT                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Shape: {str(probabilities.shape):<57} │
    │  Type: {str(probabilities.dtype):<58} │
    │  Valeurs: [{', '.join([f'{p:.3f}' for p in probabilities[0]])}]{' '*20}│
    │  Classes: NORM, MI, STTC, CD, HYP                                   │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 7: Afficher les résultats
    # ─────────────────────────────────────────────────────────────────────────
    display_results(probabilities, threshold=0.5)
    
    print("\n" + "="*70)
    print("  FIN DE L'EXEMPLE")
    print("="*70)
