"""
STEP 6D: ENTRAÎNEMENT WIDE+DEEP - VERSION HYBRIDE
Wide = 32 Excel + 26 NeuroKit2 = 58 features
Deep = Signal → CNN+Transformer → 64 features
Total = 122 features (58 + 64), légère redondance (NeuroKit2)
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

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PATIENCE = 10
NUM_CLASSES = 5

print("="*80)
print("STEP 6D: ENTRAÎNEMENT WIDE+DEEP - VERSION HYBRIDE")
print("="*80)
print(f"\n💡 Architecture:")
print(f"  • Wide: 32 Excel + 25 NeuroKit2 = 57 features")
print(f"  • Deep: Signal (12, 1000) → CNN+Transformer → 64")
print(f"  • Fusion: 57 + 64 = 121 features")
print(f"  • Device: {DEVICE}")
print()

# ==================== DATASET ====================
class WideDeepDataset(Dataset):
    def __init__(self, signal_dir, wide_dir, split):
        """
        Args:
            signal_dir: Path vers cleaned_signals_100hz/
            wide_dir: Path vers wide_features_clean/
            split: 'train', 'val', ou 'test'
        """
        # Charger Wide features
        wide_data = np.load(wide_dir / f'W_hybrid_{split}.npz', allow_pickle=True)
        self.W = torch.FloatTensor(wide_data['W'])
        self.ecg_ids = wide_data['ecg_ids']
        
        # Charger labels depuis CSV
        import pandas as pd
        df_labels = pd.read_csv('ptbxl_database.csv', usecols=['ecg_id', 'scp_codes'])
        df_labels = df_labels.set_index('ecg_id')
        
        # Charger signaux
        self.signal_dir = signal_dir
        self.samples = []
        
        for ecg_id, W_feat in zip(self.ecg_ids, self.W):
            signal_file = signal_dir / f'X_clean_{ecg_id:05d}.npz'
            
            if signal_file.exists() and ecg_id in df_labels.index:
                # Convertir labels en y_multi
                scp_codes = eval(df_labels.loc[ecg_id, 'scp_codes'])
                y_multi = np.zeros(5, dtype=np.int64)
                for code, _ in scp_codes.items():
                    if code == 'NORM': y_multi[0] = 1
                    elif code == 'MI': y_multi[1] = 1
                    elif code == 'STTC': y_multi[2] = 1
                    elif code == 'CD': y_multi[3] = 1
                    elif code == 'HYP': y_multi[4] = 1
                
                self.samples.append({
                    'ecg_id': ecg_id,
                    'signal_file': signal_file,
                    'W': W_feat,
                    'y': y_multi
                })
        
        print(f"  ✓ {split.upper()}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger signal
        data = np.load(sample['signal_file'], allow_pickle=True)
        X = torch.FloatTensor(data['signal'])  # (12, 1000)
        y = torch.LongTensor(sample['y'])       # (5,)
        
        return X, sample['W'], y


# ==================== ARCHITECTURE ====================
class WideDeepModel(nn.Module):
    def __init__(self, num_wide_features=57, num_classes=5):
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
        
        # Wide branch (MLP plus large)
        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 57)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 57, 128),  # 64 deep + 57 wide = 121
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, X_signal, W_features):
        # Deep branch
        X = self.conv_layers(X_signal)  # (batch, 512, 1)
        X = X.squeeze(-1)                # (batch, 512)
        X = self.cnn_to_transformer(X)   # (batch, 256)
        X = X.unsqueeze(1)               # (batch, 1, 256)
        X = self.transformer(X)          # (batch, 1, 256)
        X = X.squeeze(1)                 # (batch, 256)
        deep_features = self.deep_fc(X)  # (batch, 64)
        
        # Wide branch
        wide_features = self.wide_fc(W_features)  # (batch, 57)
        
        # Fusion
        combined = torch.cat([deep_features, wide_features], dim=1)  # (batch, 121)
        out = self.fusion(combined)  # (batch, 5)
        
        return out


# ==================== TRAINING ====================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for X, W, y in tqdm(loader, desc='Train', leave=False):
        X, W, y = X.to(DEVICE), W.to(DEVICE), y.float().to(DEVICE)
        
        optimizer.zero_grad()
        out = model(X, W)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, W, y in tqdm(loader, desc='Eval', leave=False):
            X, W, y = X.to(DEVICE), W.to(DEVICE), y.float().to(DEVICE)
            
            out = model(X, W)
            loss = criterion(out, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # Calculer AUC par classe (ignorer les classes sans variance)
    auc_per_class = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # Au moins 1 positif et 1 négatif
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_per_class.append(auc)
            except:
                pass
    
    # Calculer moyennes
    auc_macro = np.mean(auc_per_class) if len(auc_per_class) > 0 else 0.0
    
    # Micro: utiliser toutes les prédictions si au moins une classe valide
    try:
        auc_micro = roc_auc_score(all_labels.ravel(), all_probs.ravel())
    except:
        auc_micro = 0.0
    
    return total_loss / len(loader), auc_macro, auc_micro


# ==================== MAIN ====================
if __name__ == '__main__':
    print("[1/4] Chargement des données...")
    signal_dir = Path('cleaned_signals_100hz')
    wide_dir = Path('wide_features_clean')
    
    train_dataset = WideDeepDataset(signal_dir, wide_dir, 'train')
    val_dataset = WideDeepDataset(signal_dir, wide_dir, 'val')
    test_dataset = WideDeepDataset(signal_dir, wide_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n[2/4] Création du modèle...")
    model = WideDeepModel(num_wide_features=57, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Paramètres: {total_params:,} ({trainable_params:,} trainable)")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n[3/4] Entraînement...")
    best_val_auc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc_macro, val_auc_micro = evaluate(model, val_loader, criterion)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc_macro': val_auc_macro,
            'val_auc_micro': val_auc_micro
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val AUC Macro: {val_auc_macro:.4f} - "
              f"Val AUC Micro: {val_auc_micro:.4f}")
        
        # Early stopping
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            torch.save(model.state_dict(), 'model_wide_deep_hybrid.pth')
            print(f"  ✓ Meilleur modèle sauvegardé (AUC: {best_val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠ Early stopping à epoch {epoch+1}")
                break
    
    print(f"\n[4/4] Évaluation finale...")
    model.load_state_dict(torch.load('model_wide_deep_hybrid.pth'))
    test_loss, test_auc_macro, test_auc_micro = evaluate(model, test_loader, criterion)
    
    print(f"\n{'='*80}")
    print("RÉSULTATS FINAUX - WIDE+DEEP HYBRIDE")
    print(f"{'='*80}")
    print(f"📊 Test Loss      : {test_loss:.4f}")
    print(f"📊 Test AUC Macro : {test_auc_macro:.4f}")
    print(f"📊 Test AUC Micro : {test_auc_micro:.4f}")
    print(f"{'='*80}")
    
    # Sauvegarder historique
    with open('history_hybrid.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ STEP 6D TERMINÉ")
    print(f"  • Modèle: model_wide_deep_hybrid.pth")
    print(f"  • Historique: history_hybrid.json")
