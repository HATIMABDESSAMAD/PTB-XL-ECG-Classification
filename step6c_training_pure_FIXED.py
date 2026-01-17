"""
═══════════════════════════════════════════════════════════════════════════════
STEP 6C: ENTRAÎNEMENT WIDE+DEEP - VERSION PURE (CORRIGÉE)
═══════════════════════════════════════════════════════════════════════════════
BUG CORRIGÉ: Utilisation du mapping SCP → Superclass via scp_statements.csv
Avant: Comparaison directe code == 'NORM' (FAUX)
Après: Mapping via scp_statements.csv diagnostic_class (CORRECT)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PATIENCE = 10
NUM_CLASSES = 5
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

print("="*80)
print("STEP 6C: ENTRAÎNEMENT WIDE+DEEP - VERSION PURE (CORRIGÉE)")
print("="*80)
print(f"\n💡 Architecture:")
print(f"  • Wide: 32 features Excel uniquement")
print(f"  • Deep: Signal (12, 1000) → CNN+Transformer → 64")
print(f"  • Fusion: 32 + 64 = 96 features")
print(f"  • Device: {DEVICE}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# CORRECTION: Charger le mapping SCP → Superclass
# ═══════════════════════════════════════════════════════════════════════════════

print("[MAPPING] Chargement du mapping SCP → Superclass...")
scp_df = pd.read_csv('scp_statements.csv', index_col=0)

# Créer dictionnaire code → superclass
SCP_TO_SUPERCLASS = {}
for code in scp_df.index:
    if pd.notna(scp_df.loc[code, 'diagnostic_class']):
        SCP_TO_SUPERCLASS[code] = scp_df.loc[code, 'diagnostic_class']

print(f"  ✓ {len(SCP_TO_SUPERCLASS)} codes SCP mappés vers superclasses")

# Afficher quelques exemples
print("\n  Exemples de mapping:")
examples = ['SR', 'NORM', 'IMI', 'ASMI', 'NDT', 'LAFB', 'LVH']
for ex in examples:
    if ex in SCP_TO_SUPERCLASS:
        print(f"    {ex} → {SCP_TO_SUPERCLASS[ex]}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET avec mapping corrigé
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepDataset(Dataset):
    def __init__(self, signal_dir, wide_dir, split, scp_mapping):
        """
        Args:
            signal_dir: Path vers cleaned_signals_100hz/
            wide_dir: Path vers wide_features_clean/
            split: 'train', 'val', ou 'test'
            scp_mapping: Dictionnaire code → superclass
        """
        # Charger Wide features
        wide_data = np.load(wide_dir / f'W_pure_{split}.npz', allow_pickle=True)
        self.W = torch.FloatTensor(wide_data['W'])
        self.ecg_ids = wide_data['ecg_ids']
        
        # Charger labels depuis CSV
        df_labels = pd.read_csv('ptbxl_database.csv', usecols=['ecg_id', 'scp_codes'])
        df_labels = df_labels.set_index('ecg_id')
        
        # Charger signaux
        self.signal_dir = signal_dir
        self.samples = []
        
        for ecg_id, W_feat in zip(self.ecg_ids, self.W):
            signal_file = signal_dir / f'X_clean_{ecg_id:05d}.npz'
            
            if signal_file.exists() and ecg_id in df_labels.index:
                # ═══════════════════════════════════════════════════════════════
                # CORRECTION: Utiliser le mapping SCP → Superclass
                # ═══════════════════════════════════════════════════════════════
                scp_codes = eval(df_labels.loc[ecg_id, 'scp_codes'])
                y_multi = np.zeros(5, dtype=np.float32)  # Float pour BCE loss
                
                for code, likelihood in scp_codes.items():
                    if code in scp_mapping:
                        superclass = scp_mapping[code]
                        if superclass == 'NORM': y_multi[0] = 1
                        elif superclass == 'MI': y_multi[1] = 1
                        elif superclass == 'STTC': y_multi[2] = 1
                        elif superclass == 'CD': y_multi[3] = 1
                        elif superclass == 'HYP': y_multi[4] = 1
                
                self.samples.append({
                    'ecg_id': ecg_id,
                    'signal_file': signal_file,
                    'W': W_feat,
                    'y': y_multi
                })
        
        # Afficher distribution des labels
        labels_array = np.array([s['y'] for s in self.samples])
        print(f"\n  ✓ {split.upper()}: {len(self.samples)} samples")
        print(f"    Distribution: NORM={labels_array[:,0].sum():.0f}, MI={labels_array[:,1].sum():.0f}, "
              f"STTC={labels_array[:,2].sum():.0f}, CD={labels_array[:,3].sum():.0f}, HYP={labels_array[:,4].sum():.0f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger signal
        data = np.load(sample['signal_file'], allow_pickle=True)
        X = torch.FloatTensor(data['signal'])  # (12, 1000)
        y = torch.FloatTensor(sample['y'])     # (5,) - Float pour BCE
        
        return X, sample['W'], y


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE (identique)
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    def __init__(self, num_wide_features=32, num_classes=5):
        super().__init__()
        
        # Deep branch - CNN
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=14, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=14, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(512, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Deep branch - Transformer
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
        
        # Deep features extraction
        self.deep_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Wide branch
        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Fusion layer
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
        # Deep branch
        x = self.conv_layers(signal)
        x = x.squeeze(-1)  # (batch, 512)
        
        # Transformer
        x = self.cnn_to_transformer(x)  # (batch, 256)
        x = x.unsqueeze(1)  # (batch, 1, 256)
        x = self.transformer(x)
        x = x.mean(dim=1)  # (batch, 256)
        
        deep_out = self.deep_fc(x)  # (batch, 64)
        
        # Wide branch
        wide_out = self.wide_fc(wide_features)  # (batch, 32)
        
        # Fusion
        combined = torch.cat([deep_out, wide_out], dim=1)  # (batch, 96)
        return self.fusion(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X, W, y in tqdm(loader, desc='Train', leave=False):
        X, W, y = X.to(device), W.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(X, W)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(out).detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculer AUC par classe
    aucs = []
    for i in range(5):
        if all_labels[:, i].sum() > 0 and (1 - all_labels[:, i]).sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    
    return total_loss / len(loader), np.mean(aucs) if aucs else 0

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, W, y in tqdm(loader, desc='Eval', leave=False):
            X, W, y = X.to(device), W.to(device), y.to(device)
            
            out = model(X, W)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(out).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculer AUC par classe
    aucs = []
    auc_per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        if all_labels[:, i].sum() > 0 and (1 - all_labels[:, i]).sum() > 0:
            auc_val = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc_val)
            auc_per_class[name] = auc_val
    
    return total_loss / len(loader), np.mean(aucs) if aucs else 0, auc_per_class, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("CHARGEMENT DES DONNÉES")
print("="*80)

signal_dir = Path('cleaned_signals_100hz')
wide_dir = Path('wide_features_clean')

train_dataset = WideDeepDataset(signal_dir, wide_dir, 'train', SCP_TO_SUPERCLASS)
val_dataset = WideDeepDataset(signal_dir, wide_dir, 'val', SCP_TO_SUPERCLASS)
test_dataset = WideDeepDataset(signal_dir, wide_dir, 'test', SCP_TO_SUPERCLASS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("\n" + "="*80)
print("INITIALISATION DU MODÈLE")
print("="*80)

model = WideDeepModel(num_wide_features=32, num_classes=5)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

print("\n" + "="*80)
print("ENTRAÎNEMENT")
print("="*80)

history = {
    'train_loss': [], 'val_loss': [],
    'train_auc': [], 'val_auc': [],
    'val_auc_per_class': []
}

best_val_auc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    
    train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_auc, val_auc_class, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_auc)
    
    # Log
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auc'].append(train_auc)
    history['val_auc'].append(val_auc)
    history['val_auc_per_class'].append(val_auc_class)
    
    print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc*100:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val AUC: {val_auc*100:.2f}%")
    print(f"  Per class: " + " | ".join([f"{k}: {v*100:.1f}%" for k, v in val_auc_class.items()]))
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'model_wide_deep_pure_FIXED.pth')
        print(f"  ✓ Nouveau meilleur modèle sauvegardé!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  ⚠️ Early stopping après {PATIENCE} epochs sans amélioration")
            break

# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION FINALE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("ÉVALUATION FINALE SUR TEST SET")
print("="*80)

# Charger meilleur modèle
model.load_state_dict(torch.load('model_wide_deep_pure_FIXED.pth', weights_only=True))

test_loss, test_auc, test_auc_class, test_preds, test_labels = eval_epoch(
    model, test_loader, criterion, DEVICE
)

print(f"\n  Test Loss: {test_loss:.4f}")
print(f"  Test AUC Macro: {test_auc*100:.2f}%")
print(f"\n  AUC par classe:")
for name, auc_val in test_auc_class.items():
    print(f"    {name}: {auc_val*100:.2f}%")

# Sauvegarder historique
history['test_auc'] = test_auc
history['test_auc_per_class'] = test_auc_class

with open('history_pure_FIXED.json', 'w') as f:
    json.dump(history, f, indent=2)

# Sauvegarder prédictions pour ROC curves
np.savez('predictions_pure_FIXED.npz', 
         preds=test_preds, 
         labels=test_labels,
         class_names=CLASS_NAMES)

print("\n" + "="*80)
print("RÉSUMÉ FINAL")
print("="*80)
print(f"\n  ✓ Modèle sauvegardé: model_wide_deep_pure_FIXED.pth")
print(f"  ✓ Historique: history_pure_FIXED.json")
print(f"  ✓ Prédictions: predictions_pure_FIXED.npz")
print(f"\n  🎯 AUC Macro Test: {test_auc*100:.2f}%")
print("="*80)
