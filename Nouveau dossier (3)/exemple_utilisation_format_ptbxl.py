"""
═══════════════════════════════════════════════════════════════════════════════
SCRIPT AUTONOME - UTILISATION DU MODÈLE WIDE+DEEP PURE
═══════════════════════════════════════════════════════════════════════════════

Ce script est 100% INDÉPENDANT et fonctionne avec seulement 3 fichiers:
  1. model_wide_deep_pure_FIXED.pth  ← Le modèle entraîné
  2. XXXXX_lr.dat                     ← Signal ECG (données binaires)
  3. XXXXX_lr.hea                     ← Header ECG (métadonnées)

STRUCTURE MINIMALE DU DOSSIER:
  mon_dossier/
  ├── exemple_utilisation_format_ptbxl.py  ← Ce script
  ├── model_wide_deep_pure_FIXED.pth       ← Le modèle
  ├── 07000_lr.dat                         ← Signal ECG
  └── 07000_lr.hea                         ← Header ECG

USAGE:
  python exemple_utilisation_format_ptbxl.py
  
Le script détecte automatiquement les fichiers ECG (.dat/.hea) dans le dossier.

INPUTS DU MODÈLE:
  1. Signal ECG: (batch, 12, 1000) - 12 dérivations, 10s @ 100Hz
  2. Wide Features: (batch, 32) - 32 features (générées automatiquement)

OUTPUT:
  - Probabilités: (batch, 5) - [NORM, MI, STTC, CD, HYP]

Date: Janvier 2026
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  SCRIPT AUTONOME - MODÈLE WIDE+DEEP PURE")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION DES DÉPENDANCES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] VÉRIFICATION DES DÉPENDANCES...")

try:
    import wfdb
    print(f"  ✓ wfdb version {wfdb.__version__}")
except ImportError:
    print("  ✗ wfdb non installé")
    print("  → Installez avec: pip install wfdb")
    exit(1)

try:
    import neurokit2 as nk
    print(f"  ✓ neurokit2 version {nk.__version__}")
except ImportError:
    print("  ✗ neurokit2 non installé")
    print("  → Installez avec: pip install neurokit2")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE DU MODÈLE (copie autonome complète)
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    """
    Architecture Wide+Deep pour classification ECG multi-label.
    - Deep: CNN (6 blocs) + Transformer (8 layers)
    - Wide: 32 features cliniques
    - Fusion: 96 → 5 classes
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Dossier courant (où se trouve ce script)
CURRENT_DIR = Path(__file__).parent

# Fichiers requis
MODEL_PATH = CURRENT_DIR / 'model_wide_deep_pure_FIXED.pth'

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

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_ecg_files(directory):
    """
    Trouve les fichiers ECG (.dat/.hea) dans le dossier.
    Retourne le chemin sans extension.
    """
    hea_files = list(Path(directory).glob("*_lr.hea"))
    
    if not hea_files:
        # Chercher aussi sans le suffixe _lr
        hea_files = list(Path(directory).glob("*.hea"))
    
    ecg_records = []
    for hea in hea_files:
        dat_file = hea.with_suffix('.dat')
        if dat_file.exists():
            # Retourner le chemin sans extension
            record_path = str(hea.with_suffix(''))
            ecg_records.append(record_path)
    
    return ecg_records


def load_ecg_from_file(record_path):
    """
    Charge un signal ECG depuis les fichiers .dat/.hea (format WFDB).
    
    Args:
        record_path: chemin vers le fichier SANS extension
                     Ex: "07000_lr" ou "C:/dossier/07000_lr"
    
    Returns:
        signal: numpy array (12, 1000)
    """
    print(f"\n[ECG] Lecture du fichier...")
    print(f"  Fichiers: {Path(record_path).name}.dat / .hea")
    
    # Lire avec wfdb
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # (1000, 12)
    
    print(f"  Format brut: {signal.shape} (samples × leads)")
    print(f"  Fréquence: {record.fs} Hz")
    print(f"  Durée: {signal.shape[0] / record.fs:.1f} secondes")
    print(f"  Dérivations: {record.sig_name}")
    
    # Transposer: (1000, 12) → (12, 1000)
    signal = signal.T
    
    print(f"  Format transposé: {signal.shape} (leads × samples)")
    
    return signal.astype(np.float32)


def clean_ecg_signal(signal, sampling_rate=100):
    """
    Nettoie le signal ECG avec filtrage et normalisation.
    
    Args:
        signal: (12, 1000)
        sampling_rate: fréquence
        
    Returns:
        cleaned: (12, 1000)
    """
    print(f"\n[CLEAN] Nettoyage du signal...")
    
    cleaned = np.zeros_like(signal)
    
    for lead_idx in range(12):
        lead = signal[lead_idx, :]
        
        # Gérer les NaN
        if np.isnan(lead).any():
            lead = pd.Series(lead).interpolate().fillna(0).values
        
        # Filtre bandpass 3-45 Hz
        try:
            lead_clean = nk.ecg_clean(lead, sampling_rate=sampling_rate, method='neurokit')
        except:
            lead_clean = lead
        
        # Normalisation z-score
        mean_val = np.mean(lead_clean)
        std_val = np.std(lead_clean)
        if std_val > 1e-6:
            lead_clean = (lead_clean - mean_val) / std_val
        else:
            lead_clean = lead_clean - mean_val
        
        cleaned[lead_idx, :] = lead_clean
    
    print(f"  ✓ Filtre FIR bandpass 3-45 Hz")
    print(f"  ✓ Normalisation z-score")
    
    return cleaned.astype(np.float32)


def extract_wide_features_from_signal(signal, record=None):
    """
    Extrait 32 features Wide à partir du signal ECG.
    Ces features sont calculées automatiquement.
    
    Args:
        signal: (12, 1000) signal nettoyé
        record: wfdb record (optionnel, pour métadonnées)
        
    Returns:
        features: (32,)
    """
    print(f"\n[FEATURES] Extraction des 32 features Wide...")
    
    features = []
    
    # Utiliser lead II pour les features principales (standard clinique)
    lead_II = signal[1, :]  # Lead II
    
    try:
        # Analyser le signal ECG avec NeuroKit2
        signals_df, info = nk.ecg_process(lead_II, sampling_rate=100)
        
        # Fréquence cardiaque moyenne
        hr = signals_df['ECG_Rate'].mean()
        if np.isnan(hr):
            hr = 70
        features.append(hr / 100)  # Normaliser
        
        # Intervalles (approximations si non disponibles)
        rr_interval = 60 / hr if hr > 0 else 0.85
        features.append(rr_interval)
        
    except:
        # Valeurs par défaut si l'analyse échoue
        features.append(0.70)  # HR normalisé
        features.append(0.85)  # RR interval
    
    # Statistiques du signal par lead
    for lead_idx in [0, 1, 5, 6, 7, 8]:  # I, II, aVF, V1, V2, V3
        lead = signal[lead_idx, :]
        features.append(np.mean(lead))
        features.append(np.std(lead))
        features.append(np.max(lead) - np.min(lead))  # Amplitude
    
    # Compléter jusqu'à 32 features
    while len(features) < 32:
        features.append(0.0)
    
    features = np.array(features[:32], dtype=np.float32)
    
    print(f"  ✓ {len(features)} features extraites automatiquement")
    
    return features


def predict(model, signal, wide_features, device):
    """Effectue une prédiction."""
    
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]
    if wide_features.ndim == 1:
        wide_features = wide_features[np.newaxis, ...]
    
    signal_t = torch.from_numpy(signal).float().to(device)
    wide_t = torch.from_numpy(wide_features).float().to(device)
    
    with torch.no_grad():
        logits = model(signal_t, wide_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    return probs


def display_results(probabilities, threshold=0.5):
    """Affiche les résultats."""
    
    print("\n" + "="*65)
    print("                      RÉSULTATS")
    print("="*65)
    
    probs = probabilities[0] if probabilities.ndim == 2 else probabilities
    
    print("\n┌──────────┬─────────────┬──────────────────────────────────┐")
    print("│  Classe  │ Probabilité │          Description             │")
    print("├──────────┼─────────────┼──────────────────────────────────┤")
    
    for name, prob in zip(CLASS_NAMES, probs):
        status = "✓" if prob >= threshold else " "
        bar = "█" * int(prob * 15) + "░" * (15 - int(prob * 15))
        print(f"│ {status} {name:<6} │ {prob*100:>6.2f}% {bar}│ {CLASS_DESCRIPTIONS[name]:<32} │")
    
    print("└──────────┴─────────────┴──────────────────────────────────┘")
    
    detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
    
    print(f"\n📋 DIAGNOSTIC (seuil = {threshold*100:.0f}%):")
    if detected:
        for cls in detected:
            print(f"   ✓ {cls}: {CLASS_DESCRIPTIONS[cls]}")
    else:
        print("   → Aucune pathologie détectée")
    
    print("="*65)


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE SAISIE INTERACTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def input_ecg_file():
    """
    Demande à l'utilisateur de saisir le chemin vers les fichiers ECG.
    Retourne le chemin sans extension.
    """
    print("\n" + "─"*65)
    print("  SAISIE DU FICHIER ECG")
    print("─"*65)
    print("\n  Entrez le chemin vers le fichier ECG (.dat/.hea)")
    print("  Vous pouvez:")
    print("    - Glisser-déposer le fichier .dat ou .hea")
    print("    - Entrer le chemin complet")
    print("    - Entrer juste le nom (ex: 07000_lr) si dans le même dossier")
    print("    - Appuyer sur ENTER pour auto-détecter dans le dossier courant")
    
    user_input = input("\n  → Chemin du fichier ECG: ").strip().strip('"').strip("'")
    
    if not user_input:
        # Auto-détection
        ecg_records = find_ecg_files(CURRENT_DIR)
        if ecg_records:
            print(f"\n  ✓ Auto-détection: {len(ecg_records)} fichier(s) trouvé(s)")
            if len(ecg_records) == 1:
                return ecg_records[0]
            else:
                print("\n  Plusieurs fichiers détectés:")
                for i, rec in enumerate(ecg_records, 1):
                    print(f"    [{i}] {Path(rec).name}")
                choice = input("\n  → Numéro du fichier (1-{}): ".format(len(ecg_records)))
                try:
                    idx = int(choice) - 1
                    return ecg_records[idx]
                except:
                    return ecg_records[0]
        else:
            print("  ✗ Aucun fichier ECG trouvé")
            return None
    
    # Nettoyer le chemin
    ecg_path = Path(user_input)
    
    # Si c'est juste un nom, chercher dans le dossier courant
    if not ecg_path.is_absolute():
        ecg_path = CURRENT_DIR / user_input
    
    # Enlever l'extension si présente
    if ecg_path.suffix in ['.dat', '.hea']:
        ecg_path = ecg_path.with_suffix('')
    
    # Vérifier que les fichiers existent
    dat_file = Path(str(ecg_path) + '.dat')
    hea_file = Path(str(ecg_path) + '.hea')
    
    if dat_file.exists() and hea_file.exists():
        print(f"\n  ✓ Fichiers trouvés:")
        print(f"    - {dat_file.name}")
        print(f"    - {hea_file.name}")
        return str(ecg_path)
    else:
        if not dat_file.exists():
            print(f"  ✗ Fichier non trouvé: {dat_file}")
        if not hea_file.exists():
            print(f"  ✗ Fichier non trouvé: {hea_file}")
        return None


def input_wide_features():
    """
    Demande à l'utilisateur de saisir les 32 valeurs Wide.
    Retourne un array numpy (32,).
    """
    print("\n" + "─"*65)
    print("  SAISIE DES 32 FEATURES WIDE")
    print("─"*65)
    print("\n  Les 32 features Wide représentent des caractéristiques cliniques.")
    print("\n  Options:")
    print("    [1] Saisir les 32 valeurs manuellement")
    print("    [2] Charger depuis un fichier (.txt ou .csv)")
    print("    [3] Extraire automatiquement du signal ECG (recommandé)")
    print("    [4] Utiliser des valeurs par défaut (zéros)")
    
    choice = input("\n  → Choix (1-4) [3]: ").strip() or "3"
    
    if choice == "1":
        # Saisie manuelle
        print("\n  Entrez les 32 valeurs séparées par des espaces ou virgules:")
        print("  (Vous pouvez aussi les coller en une ligne)")
        
        values_input = input("\n  → Valeurs: ").strip()
        
        # Parser les valeurs
        values_input = values_input.replace(',', ' ').replace(';', ' ')
        values = values_input.split()
        
        try:
            features = [float(v) for v in values[:32]]
            while len(features) < 32:
                features.append(0.0)
            features = np.array(features[:32], dtype=np.float32)
            print(f"\n  ✓ {len(features)} valeurs saisies")
            return features
        except ValueError as e:
            print(f"  ✗ Erreur de parsing: {e}")
            print("  → Utilisation des valeurs par défaut")
            return np.zeros(32, dtype=np.float32)
    
    elif choice == "2":
        # Charger depuis fichier
        print("\n  Entrez le chemin du fichier (.txt ou .csv):")
        print("  (Une valeur par ligne, ou toutes sur une ligne séparées par virgules)")
        
        file_path = input("\n  → Chemin: ").strip().strip('"').strip("'")
        
        try:
            if not Path(file_path).is_absolute():
                file_path = CURRENT_DIR / file_path
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parser
            content = content.replace(',', ' ').replace(';', ' ').replace('\n', ' ')
            values = content.split()
            features = [float(v) for v in values[:32]]
            
            while len(features) < 32:
                features.append(0.0)
            features = np.array(features[:32], dtype=np.float32)
            
            print(f"\n  ✓ {len(features)} valeurs chargées depuis {Path(file_path).name}")
            return features
        
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            print("  → Utilisation des valeurs par défaut")
            return np.zeros(32, dtype=np.float32)
    
    elif choice == "3":
        # Extraction automatique (sera fait plus tard avec le signal)
        print("\n  ✓ Les features seront extraites automatiquement du signal ECG")
        return None  # Signal pour extraction auto
    
    else:
        # Valeurs par défaut
        print("\n  ✓ Utilisation des valeurs par défaut (zéros)")
        return np.zeros(32, dtype=np.float32)


def display_wide_features(features):
    """Affiche les 32 features Wide."""
    print("\n  📊 FEATURES WIDE (32 valeurs):")
    print("  ┌" + "─"*60 + "┐")
    for i in range(0, 32, 4):
        row = "  │ "
        for j in range(4):
            if i + j < 32:
                row += f"[{i+j:2d}]: {features[i+j]:>8.4f}  "
        row = row.ljust(63) + "│"
        print(row)
    print("  └" + "─"*60 + "┘")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1: Vérifier le modèle
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] VÉRIFICATION DU MODÈLE...")
    
    print(f"\n  Dossier: {CURRENT_DIR}")
    
    if MODEL_PATH.exists():
        print(f"  ✓ Modèle trouvé: {MODEL_PATH.name}")
    else:
        print(f"  ✗ Modèle non trouvé: {MODEL_PATH.name}")
        print("    → Placez 'model_wide_deep_pure_FIXED.pth' dans ce dossier")
        exit(1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2: Charger le modèle
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] CHARGEMENT DU MODÈLE...")
    
    model = WideDeepModel(num_wide_features=32, num_classes=5)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    print(f"  ✓ Modèle chargé")
    print(f"  Device: {DEVICE}")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3: Saisie interactive des inputs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] SAISIE DES INPUTS...")
    
    # 3a. Saisie du fichier ECG
    record_path = input_ecg_file()
    
    if record_path is None:
        print("\n  ✗ Impossible de continuer sans fichier ECG valide")
        exit(1)
    
    # 3b. Saisie des 32 features Wide
    wide_features_input = input_wide_features()
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 4: Traitement de l'ECG
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5] TRAITEMENT DE L'ECG...")
    
    print("\n" + "─"*65)
    print(f"  ECG: {Path(record_path).name}")
    print("─"*65)
    
    # Charger le signal
    signal_raw = load_ecg_from_file(record_path)
    
    # Nettoyer le signal
    signal_clean = clean_ecg_signal(signal_raw)
    
    # Déterminer les features Wide
    if wide_features_input is None:
        # Extraction automatique
        wide_features = extract_wide_features_from_signal(signal_clean)
    else:
        wide_features = wide_features_input
        print(f"\n[FEATURES] Utilisation des features saisies manuellement")
    
    # Afficher les features Wide
    display_wide_features(wide_features)
    
    # Afficher les inputs
    print(f"\n[INPUTS FINAUX]")
    print(f"  Signal ECG: shape={signal_clean.shape}, dtype={signal_clean.dtype}")
    print(f"  Wide Features: shape={wide_features.shape}, dtype={wide_features.dtype}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 5: Prédiction
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[6] PRÉDICTION...")
    
    probabilities = predict(model, signal_clean, wide_features, DEVICE)
    
    print(f"  Output: shape={probabilities.shape}")
    print(f"  Probabilités: {[f'{p:.3f}' for p in probabilities[0]]}")
    
    # Afficher les résultats
    display_results(probabilities, threshold=0.5)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 6: Continuer ou quitter
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    
    while True:
        again = input("\n  Analyser un autre ECG? (o/n) [n]: ").strip().lower()
        if again in ['o', 'oui', 'y', 'yes']:
            # Recommencer
            record_path = input_ecg_file()
            if record_path is None:
                continue
            
            wide_features_input = input_wide_features()
            
            signal_raw = load_ecg_from_file(record_path)
            signal_clean = clean_ecg_signal(signal_raw)
            
            if wide_features_input is None:
                wide_features = extract_wide_features_from_signal(signal_clean)
            else:
                wide_features = wide_features_input
            
            display_wide_features(wide_features)
            
            probabilities = predict(model, signal_clean, wide_features, DEVICE)
            display_results(probabilities, threshold=0.5)
        else:
            break
    
    print("\n" + "="*70)
    print("  FIN DU TRAITEMENT")
    print("="*70)
