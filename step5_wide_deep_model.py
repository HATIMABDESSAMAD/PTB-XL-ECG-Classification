"""
═══════════════════════════════════════════════════════════════════════════════
STEP 5: WIDE+DEEP MODEL ARCHITECTURE - PTB-XL Pipeline
═══════════════════════════════════════════════════════════════════════════════
Architecture hybride Wide+Deep en PyTorch:

DEEP BRANCH (signaux ECG 12 leads):
  X_clean (12, 1000) → Conv1D blocks → Transformer → DeepFeatures

WIDE BRANCH (features cliniques + metadata):
  W_features (D_wide) → MLP → WideFeatures

FUSION:
  [DeepFeatures | WideFeatures] → FC Head → logits (K classes)
  
Loss: BCEWithLogitsLoss (multi-label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 100)
print("STEP 5: WIDE+DEEP MODEL ARCHITECTURE (PyTorch)")
print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEEP BRANCH: CNN1D + Transformer pour signaux ECG
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Block convolutionnel 1D avec BatchNorm, ReLU, Dropout, MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding pour Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class DeepBranch(nn.Module):
    """
    Branch Deep pour signaux ECG - Architecture CNN + Transformer
    
    Input: (batch, 12, 1000) - 12 leads × 1000 samples (100Hz × 10s)
    
    Pipeline:
      (12, 1000)
      ↓ Conv1D: out=128, k=14, s=3, p=2   + BN + ReLU
      (128, T1)
      ↓ Conv1D: out=256, k=14, s=3, p=0   + BN + ReLU
      (256, T2)
      ↓ Conv1D: out=256, k=10, s=2, p=0   + BN + ReLU
      (256, T3)
      ↓ Conv1D: out=256, k=10, s=2, p=0   + BN + ReLU
      (256, T4)
      ↓ Conv1D: out=256, k=10, s=1, p=0   + BN + ReLU
      (256, T5)
      ↓ Conv1D: out=256, k=10, s=1, p=0   + BN + ReLU
      (256, T6)
      ↓ + Positional Encoding
      ↓ Transformer Encoder: N=8 layers, heads=8, d_model=256, dropout=0.1
      (seq_len, 256)
      ↓ Global Average Pool
      (256,)
      ↓ FC → 64
      (64,) = DeepFeatures
    
    Output: (batch, 64) - vecteur de features
    """
    def __init__(self, 
                 n_leads=12, 
                 seq_len=1000,
                 d_model=256,
                 transformer_heads=8,
                 transformer_layers=8,
                 deep_features_dim=64,
                 dropout=0.1):
        super().__init__()
        
        # ────────────────────────────────────────────────────────────────────
        # A. CNN1D Encoder (6 couches convolutionnelles)
        # ────────────────────────────────────────────────────────────────────
        # Conv1: (12, 1000) → (128, T1)
        self.conv1 = nn.Conv1d(n_leads, 128, kernel_size=14, stride=3, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Conv2: (128, T1) → (256, T2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Conv3: (256, T2) → (256, T3)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Conv4: (256, T3) → (256, T4)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Conv5: (256, T4) → (256, T5)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Conv6: (256, T5) → (256, T6)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.bn6 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # ────────────────────────────────────────────────────────────────────
        # B. Transformer Encoder (8 layers)
        # ────────────────────────────────────────────────────────────────────
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer layers (N=8, heads=8, d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * 4,  # 1024
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # ────────────────────────────────────────────────────────────────────
        # C. Global Pooling + Projection finale
        # ────────────────────────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_deep = nn.Linear(d_model, deep_features_dim)
        self.dropout_fc = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, 12, 1000)
        
        # ═══════════════════════════════════════════════════════════════════
        # A. CNN Encoder (6 Conv1D layers)
        # ═══════════════════════════════════════════════════════════════════
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Conv4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Conv5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        # Conv6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        # x: (batch, 256, T6)
        
        # ═══════════════════════════════════════════════════════════════════
        # B. Transformer Encoder
        # ═══════════════════════════════════════════════════════════════════
        # Permuter pour Transformer: (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, T6, 256)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer (8 layers)
        x = self.transformer(x)  # (batch, T6, 256)
        
        # ═══════════════════════════════════════════════════════════════════
        # C. Global Pooling + FC
        # ═══════════════════════════════════════════════════════════════════
        # Global average pooling sur dimension temporelle
        x = x.permute(0, 2, 1)  # (batch, 256, T6)
        x = self.global_pool(x).squeeze(-1)  # (batch, 256)
        
        # Projection finale: 256 → 64
        x = self.fc_deep(x)  # (batch, 64)
        x = self.dropout_fc(x)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WIDE BRANCH: MLP pour features tabulaires
# ═══════════════════════════════════════════════════════════════════════════════

class WideBranch(nn.Module):
    """
    Branch Wide pour features cliniques + metadata
    
    Input: (batch, D_wide) - features préprocessées
    Output: (batch, wide_features_dim) - vecteur de features
    """
    def __init__(self, 
                 input_dim,
                 hidden_dims=[256, 128],
                 wide_features_dim=64,
                 dropout=0.3):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Projection finale
        layers.append(nn.Linear(in_dim, wide_features_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, D_wide)
        return self.mlp(x)  # (batch, wide_features_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WIDE+DEEP MODEL COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    """
    Modèle hybride Wide+Deep pour PTB-XL
    
    Inputs:
        - X_signal: (batch, 12, 1000) - signaux ECG
        - W_features: (batch, D_wide) - features tabulaires
    
    Output:
        - logits: (batch, n_classes) - scores multi-label
    """
    def __init__(self,
                 n_classes,
                 wide_input_dim,
                 # Deep branch params
                 deep_d_model=256,
                 deep_transformer_heads=8,
                 deep_transformer_layers=8,
                 deep_features_dim=64,
                 # Wide branch params
                 wide_hidden_dims=[256, 128],
                 wide_features_dim=64,
                 # Fusion params
                 fusion_hidden_dim=128,
                 dropout=0.1):
        super().__init__()
        
        # ────────────────────────────────────────────────────────────────────
        # A. BRANCHES
        # ────────────────────────────────────────────────────────────────────
        self.deep_branch = DeepBranch(
            n_leads=12,
            seq_len=1000,
            d_model=deep_d_model,
        )
        
        self.wide_branch = WideBranch(
            input_dim=wide_input_dim,
            hidden_dims=wide_hidden_dims,
            wide_features_dim=wide_features_dim,
            dropout=dropout
        )
        
        # ────────────────────────────────────────────────────────────────────
        # B. FUSION HEAD
        # ────────────────────────────────────────────────────────────────────
        fusion_input_dim = deep_features_dim + wide_features_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, n_classes)
        )
    
    def forward(self, X_signal, W_features):
        """
        Args:
            X_signal: (batch, 12, 1000)
            W_features: (batch, D_wide)
        
        Returns:
            logits: (batch, n_classes)
        """
        # Branches
        deep_feats = self.deep_branch(X_signal)      # (batch, deep_features_dim)
        wide_feats = self.wide_branch(W_features)    # (batch, wide_features_dim)
        
        # Concatenation
        fused = torch.cat([deep_feats, wide_feats], dim=1)  # (batch, deep_dim + wide_dim)
        
        # Classification head
        logits = self.fusion(fused)  # (batch, n_classes)
        
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BASELINES (Deep only, Wide only)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepOnlyModel(nn.Module):
    """Baseline: Deep branch seulement"""
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        
        self.deep_branch = DeepBranch(
            deep_features_dim=256,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, X_signal):
        feats = self.deep_branch(X_signal)
        return self.classifier(feats)


class WideOnlyModel(nn.Module):
    """Baseline: Wide branch seulement"""
    def __init__(self, n_classes, wide_input_dim, dropout=0.3):
        super().__init__()
        
        self.wide_branch = WideBranch(
            input_dim=wide_input_dim,
            wide_features_dim=128,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, W_features):
        feats = self.wide_branch(W_features)
        return self.classifier(feats)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXEMPLE UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("TEST ARCHITECTURE")
    print("=" * 100)
    
    # Paramètres exemple
    batch_size = 8
    n_classes = 71  # 71 codes SCP (ou 5 pour superclasses)
    wide_input_dim = 50  # Exemple: ~50 features Wide
    
    # Créer modèle
    model = WideDeepModel(
        n_classes=n_classes,
        wide_input_dim=wide_input_dim
    )
    
    print(f"\n📊 Modèle Wide+Deep créé")
    print(f"  • Classes: {n_classes}")
    print(f"  • Wide input dim: {wide_input_dim}")
    
    # Compter paramètres
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Paramètres total: {n_params:,}")
    
    # Test forward pass
    X_signal = torch.randn(batch_size, 12, 1000)
    W_features = torch.randn(batch_size, wide_input_dim)
    
    print(f"\n🔍 Test forward pass:")
    print(f"  • X_signal: {X_signal.shape}")
    print(f"  • W_features: {W_features.shape}")
    
    with torch.no_grad():
        logits = model(X_signal, W_features)
    
    print(f"  • Logits: {logits.shape}")
    print(f"  ✓ Forward pass OK")
    
    # Test baselines
    print(f"\n📈 Baselines:")
    
    model_deep = DeepOnlyModel(n_classes=n_classes)
    n_params_deep = sum(p.numel() for p in model_deep.parameters() if p.requires_grad)
    print(f"  • DeepOnly: {n_params_deep:,} params")
    
    model_wide = WideOnlyModel(n_classes=n_classes, wide_input_dim=wide_input_dim)
    n_params_wide = sum(p.numel() for p in model_wide.parameters() if p.requires_grad)
    print(f"  • WideOnly: {n_params_wide:,} params")
    
    print(f"\n✅ ARCHITECTURE VALIDÉE")
    print(f"   Prochaine étape: step6_training.py")
    print("=" * 100)
