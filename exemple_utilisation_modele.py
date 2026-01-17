"""
═══════════════════════════════════════════════════════════════════════════════
EXEMPLE D'UTILISATION DU MODÈLE WIDE+DEEP PURE
═══════════════════════════════════════════════════════════════════════════════

Ce script est AUTONOME et INDÉPENDANT.
Il nécessite uniquement:
  - model_wide_deep_pure_FIXED.pth (le modèle entraîné)
  - Un fichier ECG du dataset PTB-XL (optionnel, sinon données simulées)

INPUTS:
  1. Signal ECG: (batch, 12, 1000) - 12 dérivations, 1000 samples (10s @ 100Hz)
  2. Wide Features: (batch, 32) - 32 features cliniques Excel

OUTPUT:
  - Probabilités: (batch, 5) - 5 classes [NORM, MI, STTC, CD, HYP]

Auteur: Pipeline Wide+Deep PTB-XL
Date: Janvier 2026
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DÉFINITION DE L'ARCHITECTURE (copie autonome)
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    """
    Architecture Wide+Deep pour classification ECG multi-label.
    
    Wide Branch: 32 features cliniques → 32 features
    Deep Branch: Signal ECG (12, 1000) → CNN → Transformer → 64 features
    Fusion: 96 features → 5 classes
    
    Total: 11,561,573 paramètres
    """
    
    def __init__(self, num_wide_features=32, num_classes=5):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════════════════════
        # DEEP BRANCH - CNN ENCODER (6 blocs)
        # ═══════════════════════════════════════════════════════════════════════
        self.conv_layers = nn.Sequential(
            # Bloc 1: (B, 12, 1000) → (B, 64, 500)
            nn.Conv1d(12, 64, kernel_size=14, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Bloc 2: (B, 64, 500) → (B, 128, 250)
            nn.Conv1d(64, 128, kernel_size=14, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Bloc 3: (B, 128, 250) → (B, 256, 125)
            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Bloc 4: (B, 256, 125) → (B, 256, 62)
            nn.Conv1d(256, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Bloc 5: (B, 256, 62) → (B, 512, 31)
            nn.Conv1d(256, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Bloc 6: (B, 512, 31) → (B, 512, 1)
            nn.Conv1d(512, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # DEEP BRANCH - TRANSFORMER (8 layers)
        # ═══════════════════════════════════════════════════════════════════════
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # Projection CNN → Transformer
        self.cnn_to_transformer = nn.Linear(512, 256)
        
        # Deep features extraction: 256 → 64
        self.deep_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # WIDE BRANCH: 32 features → 32 features
        # ═══════════════════════════════════════════════════════════════════════
        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # FUSION: 64 + 32 = 96 → 5 classes
        # ═══════════════════════════════════════════════════════════════════════
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, signal, wide_features):
        """
        Forward pass du modèle.
        
        Args:
            signal: Tensor (batch, 12, 1000) - Signal ECG 12 dérivations
            wide_features: Tensor (batch, 32) - Features cliniques
            
        Returns:
            logits: Tensor (batch, 5) - Scores avant sigmoid
        """
        # Deep branch - CNN
        x = self.conv_layers(signal)      # (B, 512, 1)
        x = x.squeeze(-1)                  # (B, 512)
        
        # Deep branch - Transformer
        x = self.cnn_to_transformer(x)    # (B, 256)
        x = x.unsqueeze(1)                 # (B, 1, 256)
        x = self.transformer(x)            # (B, 1, 256)
        x = x.mean(dim=1)                  # (B, 256)
        
        deep_out = self.deep_fc(x)        # (B, 64)
        
        # Wide branch
        wide_out = self.wide_fc(wide_features)  # (B, 32)
        
        # Fusion
        combined = torch.cat([deep_out, wide_out], dim=1)  # (B, 96)
        logits = self.fusion(combined)    # (B, 5)
        
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Chemin vers le modèle
MODEL_PATH = 'model_wide_deep_pure_FIXED.pth'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes de sortie
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'ECG Normal',
    'MI': 'Infarctus du Myocarde (Myocardial Infarction)',
    'STTC': 'Changements ST/T (ST/T Changes)',
    'CD': 'Troubles de Conduction (Conduction Disturbances)',
    'HYP': 'Hypertrophie (Hypertrophy)'
}

# Les 32 features Wide (Excel)
WIDE_FEATURE_NAMES = [
    # Intervalles (5)
    'RR_interval', 'PR_interval', 'QRS_duration', 'QT_interval', 'QTc_interval',
    # Amplitudes (8)
    'P_amplitude', 'Q_amplitude', 'R_amplitude', 'S_amplitude', 'T_amplitude',
    'ST_amplitude_V1', 'ST_amplitude_V2', 'ST_amplitude_V3',
    # Axes (3)
    'P_axis', 'QRS_axis', 'T_axis',
    # Démographiques (4)
    'age', 'sex', 'height', 'weight',
    # Qualité signal (6)
    'baseline_drift', 'static_noise', 'burst_noise', 
    'electrodes_problems', 'extra_beats', 'pacemaker',
    # Validation (2)
    'validated_by', 'second_opinion',
    # Autres (4)
    'heart_rate', 'recording_date_numeric', 'device_id', 'nurse_id'
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path, device):
    """Charge le modèle pré-entraîné."""
    print(f"[INFO] Chargement du modèle depuis: {model_path}")
    
    # Créer l'architecture
    model = WideDeepModel(num_wide_features=32, num_classes=5)
    
    # Charger les poids
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"[INFO] ✓ Modèle chargé avec succès!")
    else:
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    model.to(device)
    model.eval()
    
    return model


def load_real_ecg(ecg_id, signals_dir='cleaned_signals_100hz'):
    """
    Charge un vrai signal ECG du dataset PTB-XL.
    
    Format des fichiers: X_clean_XXXXX.npz
    Contenu: 'signal' de shape (12, 1000)
    
    Les 12 dérivations sont:
        I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    signal_path = Path(signals_dir) / f'X_clean_{ecg_id:05d}.npz'
    
    if signal_path.exists():
        data = np.load(signal_path)
        signal = data['signal']  # (12, 1000)
        print(f"[INFO] ✓ Signal ECG {ecg_id} chargé: shape = {signal.shape}")
        return signal
    else:
        print(f"[WARN] Signal non trouvé: {signal_path}")
        return None


def load_real_wide_features(ecg_id, wide_dir='wide_features_clean'):
    """
    Charge les vraies features Wide du dataset.
    
    Format des fichiers: W_pure_test.npz (ou train/val)
    Contenu: 'W' de shape (N, 32), 'ecg_ids' de shape (N,)
    """
    for split in ['test', 'val', 'train']:
        wide_path = Path(wide_dir) / f'W_pure_{split}.npz'
        
        if wide_path.exists():
            data = np.load(wide_path)
            ecg_ids = data['ecg_ids']
            
            if ecg_id in ecg_ids:
                idx = np.where(ecg_ids == ecg_id)[0][0]
                wide_features = data['W'][idx]  # (32,)
                print(f"[INFO] ✓ Wide features {ecg_id} chargées depuis {split}: shape = {wide_features.shape}")
                return wide_features
    
    print(f"[WARN] Wide features non trouvées pour ECG {ecg_id}")
    return None


def simulate_ecg_signal():
    """
    Simule un signal ECG réaliste (12 dérivations, 1000 samples @ 100Hz).
    Utilisé quand les données réelles ne sont pas disponibles.
    """
    print("[INFO] Génération d'un signal ECG simulé...")
    
    t = np.linspace(0, 10, 1000)  # 10 secondes @ 100Hz
    signal = np.zeros((12, 1000))
    
    # Fréquence cardiaque ~ 70 bpm
    heart_rate = 70
    rr_interval = 60 / heart_rate  # secondes
    
    for lead in range(12):
        # Composante de base (onde sinusoïdale)
        signal[lead] = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        
        # Ajout de bruit réaliste
        signal[lead] += 0.05 * np.random.randn(1000)
        
        # Variation par dérivation
        signal[lead] *= (0.8 + 0.4 * np.random.rand())
    
    print(f"[INFO] ✓ Signal simulé: shape = {signal.shape}")
    return signal.astype(np.float32)


def simulate_wide_features():
    """
    Simule 32 features cliniques réalistes.
    Utilisé quand les données réelles ne sont pas disponibles.
    """
    print("[INFO] Génération de features Wide simulées...")
    
    features = np.array([
        # Intervalles (ms, normalisés)
        0.85,   # RR_interval (~850ms = 70bpm)
        0.16,   # PR_interval (~160ms)
        0.09,   # QRS_duration (~90ms)
        0.40,   # QT_interval (~400ms)
        0.42,   # QTc_interval (~420ms)
        
        # Amplitudes (mV, normalisés)
        0.15,   # P_amplitude
        -0.1,   # Q_amplitude
        1.2,    # R_amplitude
        -0.3,   # S_amplitude
        0.3,    # T_amplitude
        0.05,   # ST_amplitude_V1
        0.08,   # ST_amplitude_V2
        0.1,    # ST_amplitude_V3
        
        # Axes (degrés, normalisés)
        45,     # P_axis
        60,     # QRS_axis
        40,     # T_axis
        
        # Démographiques
        55,     # age
        1,      # sex (1=M, 0=F)
        175,    # height (cm)
        75,     # weight (kg)
        
        # Qualité signal (0=bon, 1=problème)
        0,      # baseline_drift
        0,      # static_noise
        0,      # burst_noise
        0,      # electrodes_problems
        0,      # extra_beats
        0,      # pacemaker
        
        # Validation
        1,      # validated_by
        0,      # second_opinion
        
        # Autres
        70,     # heart_rate
        0.5,    # recording_date_numeric
        1,      # device_id
        1       # nurse_id
    ], dtype=np.float32)
    
    print(f"[INFO] ✓ Wide features simulées: shape = {features.shape}")
    return features


def predict(model, signal, wide_features, device):
    """
    Effectue une prédiction avec le modèle.
    
    Args:
        model: WideDeepModel chargé
        signal: numpy array (12, 1000) ou (batch, 12, 1000)
        wide_features: numpy array (32,) ou (batch, 32)
        device: torch device
        
    Returns:
        probabilities: numpy array (5,) ou (batch, 5)
        predictions: dict avec les probabilités par classe
    """
    # Ajouter dimension batch si nécessaire
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]  # (1, 12, 1000)
    if wide_features.ndim == 1:
        wide_features = wide_features[np.newaxis, ...]  # (1, 32)
    
    # Convertir en tensors
    signal_tensor = torch.from_numpy(signal).float().to(device)
    wide_tensor = torch.from_numpy(wide_features).float().to(device)
    
    # Prédiction
    with torch.no_grad():
        logits = model(signal_tensor, wide_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    
    return probabilities


def display_results(probabilities, threshold=0.5):
    """Affiche les résultats de manière formatée."""
    
    print("\n" + "="*60)
    print("                    RÉSULTATS DE PRÉDICTION")
    print("="*60)
    
    probs = probabilities[0] if probabilities.ndim == 2 else probabilities
    
    print("\n┌────────────┬────────────────┬──────────────────────────────────┐")
    print("│   Classe   │  Probabilité   │          Description             │")
    print("├────────────┼────────────────┼──────────────────────────────────┤")
    
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        status = "✓" if prob >= threshold else " "
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        desc = CLASS_DESCRIPTIONS[name][:30]
        print(f"│ {status} {name:<7} │ {prob*100:>6.2f}% {bar} │ {desc:<32} │")
    
    print("└────────────┴────────────────┴──────────────────────────────────┘")
    
    # Diagnostic final
    detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
    
    print(f"\n📋 DIAGNOSTIC (seuil = {threshold*100:.0f}%):")
    if detected:
        for cls in detected:
            print(f"   ✓ {cls}: {CLASS_DESCRIPTIONS[cls]}")
    else:
        print("   Aucune pathologie détectée au-dessus du seuil")
    
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXEMPLE D'UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("="*70)
    print("     EXEMPLE D'UTILISATION - MODÈLE WIDE+DEEP PURE")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1: Charger le modèle
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ÉTAPE 1] Chargement du modèle...")
    
    try:
        model = load_model(MODEL_PATH, DEVICE)
        print(f"[INFO] Device utilisé: {DEVICE}")
        print(f"[INFO] Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        print("[INFO] Assurez-vous que 'model_wide_deep_pure_FIXED.pth' est dans le dossier courant")
        exit(1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2: Préparer les inputs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ÉTAPE 2] Préparation des inputs...")
    
    # Option A: Charger des données réelles (si disponibles)
    ECG_ID = 1  # Exemple: ECG ID 1
    
    signal = load_real_ecg(ECG_ID)
    wide_features = load_real_wide_features(ECG_ID)
    
    # Option B: Utiliser des données simulées (si fichiers non disponibles)
    if signal is None:
        signal = simulate_ecg_signal()
    if wide_features is None:
        wide_features = simulate_wide_features()
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3: Afficher les inputs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ÉTAPE 3] Résumé des inputs...")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                         INPUTS                                  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Signal ECG:                                                    │")
    print(f"│    • Shape: {str(signal.shape):<50} │")
    print(f"│    • Type: {str(signal.dtype):<51} │")
    print(f"│    • Min/Max: [{signal.min():.3f}, {signal.max():.3f}]" + " "*35 + "│")
    print(f"│    • 12 dérivations: I, II, III, aVR, aVL, aVF, V1-V6          │")
    print(f"│    • 1000 samples = 10 secondes @ 100Hz                        │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Wide Features:                                                 │")
    print(f"│    • Shape: {str(wide_features.shape):<50} │")
    print(f"│    • Type: {str(wide_features.dtype):<51} │")
    print(f"│    • 32 features cliniques (intervalles, amplitudes, etc.)     │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 4: Effectuer la prédiction
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ÉTAPE 4] Prédiction...")
    
    probabilities = predict(model, signal, wide_features, DEVICE)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                         OUTPUT                                   │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Probabilités:                                                  │")
    print(f"│    • Shape: {str(probabilities.shape):<50} │")
    print(f"│    • Type: {str(probabilities.dtype):<51} │")
    print(f"│    • Valeurs: [{', '.join([f'{p:.3f}' for p in probabilities[0]])}]" + " "*14 + "│")
    print(f"│    • 5 classes: NORM, MI, STTC, CD, HYP                         │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 5: Afficher les résultats
    # ─────────────────────────────────────────────────────────────────────────
    display_results(probabilities, threshold=0.5)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXEMPLE DE BATCH (plusieurs ECG à la fois)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("     EXEMPLE BATCH (3 ECG simultanés)")
    print("="*70)
    
    # Créer un batch de 3 ECG simulés
    batch_signals = np.stack([simulate_ecg_signal() for _ in range(3)])
    batch_wide = np.stack([simulate_wide_features() for _ in range(3)])
    
    print(f"\n[INFO] Batch signals shape: {batch_signals.shape}")
    print(f"[INFO] Batch wide shape: {batch_wide.shape}")
    
    # Prédiction batch
    batch_probs = predict(model, batch_signals, batch_wide, DEVICE)
    
    print(f"\n[INFO] Batch output shape: {batch_probs.shape}")
    print("\n[INFO] Résultats par ECG:")
    for i in range(3):
        probs = batch_probs[i]
        detected = [CLASS_NAMES[j] for j, p in enumerate(probs) if p >= 0.5]
        print(f"  ECG {i+1}: {detected if detected else ['Normal']}")
    
    print("\n" + "="*70)
    print("     FIN DE L'EXEMPLE")
    print("="*70)
