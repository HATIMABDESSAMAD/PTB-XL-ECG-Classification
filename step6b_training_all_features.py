"""
STEP 6B: TRAINING WIDE+DEEP AVEC 124 FEATURES (Excel 32 + Deep 64 + NeuroKit2 27)
====================================================================================
Entraînement du modèle Wide+Deep avec toutes les features fusionnées
Comparaison avec step6_training.py (25 features seulement)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import du modèle depuis step5
from step5_wide_deep_model import WideDeepModel

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Tâche
    task_mode = '5superclass'  # '30scp' ou '5superclass'
    
    # Fichiers
    labels_file = 'ptbxl_from_excel_consolidated.csv'
    label_config_file = 'label_config_from_excel.json'
    wide_dir = 'all_features_engineered'  # CHANGEMENT: utiliser features avec engineering
    signals_dir = 'cleaned_signals_100hz'
    models_dir = 'models'
    
    # Hyperparamètres
    batch_size = 32
    learning_rate = 1e-4
    dropout = 0.1
    num_epochs = 50
    early_stopping_patience = 10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Seed
    seed = 42

# ============================================================================
# DATASET
# ============================================================================

class PTBXLWideDeepDataset(Dataset):
    """Dataset pour Wide+Deep"""
    def __init__(self, wide_features, signal_ids, labels, filenames):
        self.W = torch.FloatTensor(wide_features)
        self.signal_ids = signal_ids
        self.filenames = filenames
        self.y = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.W)
    
    def __getitem__(self, idx):
        W = self.W[idx]
        filename = self.filenames[idx]
        y = self.y[idx]
        
        # Charger signal en utilisant filename_lr (ex: records100/00000/00001_lr)
        # Extraire juste le numéro (00001) du filename
        signal_id = filename.split('/')[-1]  # 00001_lr
        signal_path = Path(config.signals_dir) / f'X_clean_{signal_id.replace("_lr", "")}.npz'
        signal_data = np.load(signal_path)
        X = signal_data['signal']  # (12, 1000) - CORRECTION: clé est 'signal' pas 'X_clean'
        X = torch.FloatTensor(X)
        
        return X, W, y  # CORRECTION: ordre (X_signal, W_features, y) pour correspondre à forward()

# ============================================================================
# FONCTIONS UTILES
# ============================================================================

def load_data(config):
    """Charge les données Wide+Deep"""
    print("[1/3] Chargement des données...")
    
    # 1. Labels
    df = pd.read_csv(config.labels_file)
    label_config = json.load(open(config.label_config_file))
    
    if config.task_mode == '5superclass':
        label_cols = label_config.get('superclass_cols', 
                                      label_config.get('superclass_cols_excel', []))
        n_classes = 5
    else:
        label_cols = label_config.get('scp_cols', 
                                      label_config.get('scp_cols_excel', []))
        n_classes = 30
    
    # 2. Wide features fusionnées (122 features engineered)
    data_train = np.load(f'{config.wide_dir}/W_engineered_train.npz')
    data_val = np.load(f'{config.wide_dir}/W_engineered_val.npz')
    data_test = np.load(f'{config.wide_dir}/W_engineered_test.npz')
    
    W_train = data_train['W']
    W_val = data_val['W']
    W_test = data_test['W']
    
    ids_train = data_train['ecg_ids']
    ids_val = data_val['ecg_ids']
    ids_test = data_test['ecg_ids']
    
    filenames_train = data_train['filenames']
    filenames_val = data_val['filenames']
    filenames_test = data_test['filenames']
    
    # 3. Labels
    df_train = df[df['ecg_id'].isin(ids_train)].set_index('ecg_id').loc[ids_train].reset_index()
    df_val = df[df['ecg_id'].isin(ids_val)].set_index('ecg_id').loc[ids_val].reset_index()
    df_test = df[df['ecg_id'].isin(ids_test)].set_index('ecg_id').loc[ids_test].reset_index()
    
    y_train = df_train[label_cols].values
    y_val = df_val[label_cols].values
    y_test = df_test[label_cols].values
    
    print(f"  ✓ Train: {len(W_train)} ECG, {W_train.shape[1]} features Wide (engineered)")
    print(f"  ✓ Val  : {len(W_val)} ECG, {W_val.shape[1]} features Wide (engineered)")
    print(f"  ✓ Test : {len(W_test)} ECG, {W_test.shape[1]} features Wide (engineered)")
    print(f"  ✓ Classes: {n_classes} ({config.task_mode})")
    
    return {
        'train': (W_train, ids_train, y_train, filenames_train),
        'val': (W_val, ids_val, y_val, filenames_val),
        'test': (W_test, ids_test, y_test, filenames_test),
        'n_classes': n_classes,
        'wide_input_dim': W_train.shape[1]  # 122 engineered
    }

def create_dataloaders(data, config):
    """Crée les DataLoaders"""
    print("\n[2/3] Création DataLoaders...")
    
    train_dataset = PTBXLWideDeepDataset(*data['train'])  # W, ids, y, filenames
    val_dataset = PTBXLWideDeepDataset(*data['val'])
    test_dataset = PTBXLWideDeepDataset(*data['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches  : {len(val_loader)}")
    print(f"  ✓ Test batches : {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
    """Entraîne une epoch"""
    model.train()
    total_loss = 0
    
    for X, W, y in tqdm(loader, desc="  Training", leave=False):
        X, W, y = X.to(device), W.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X, W)  # Ordre correct: (X_signal, W_features)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Évalue le modèle"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, W, y in tqdm(loader, desc="  Evaluating", leave=False):
            X, W, y = X.to(device), W.to(device), y.to(device)
            
            logits = model(X, W)  # Ordre correct: (X_signal, W_features)
            loss = criterion(logits, y)
            
            probs = torch.sigmoid(logits)
            
            total_loss += loss.item()
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # AUC par classe
    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
    
    avg_loss = total_loss / len(loader)
    avg_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    return avg_loss, avg_auc, auc_scores

def train_model(config):
    """Pipeline complet d'entraînement"""
    
    print("="*85)
    print("STEP 6B: TRAINING WIDE+DEEP (122 ENGINEERED FEATURES)")
    print("="*85)
    print()
    print(f"⚙️  CONFIGURATION:")
    print(f"  • Task: {config.task_mode}")
    print(f"  • Wide features: {config.wide_dir} (122 engineered)")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Learning rate: {config.learning_rate}")
    print(f"  • Dropout: {config.dropout}")
    print(f"  • Device: {config.device}")
    print()
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Charger données
    data = load_data(config)
    train_loader, val_loader, test_loader = create_dataloaders(data, config)
    
    # Modèle
    print("\n[3/3] Création du modèle...")
    model = WideDeepModel(
        n_classes=data['n_classes'],
        wide_input_dim=data['wide_input_dim'],  # 124 au lieu de 25
        deep_d_model=256,
        deep_transformer_layers=8,
        deep_features_dim=64,
        dropout=config.dropout
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Paramètres totaux: {total_params/1e6:.2f}M")
    
    # Loss + Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Entraînement
    print("\n" + "="*85)
    print("ENTRAÎNEMENT")
    print("="*85)
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    Path(config.models_dir).mkdir(exist_ok=True)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Val
        val_loss, val_auc, _ = evaluate(model, val_loader, criterion, config.device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}  |  Val AUC: {val_auc:.4f}")
        
        # Early stopping + save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save
            torch.save(model.state_dict(), 
                      f'{config.models_dir}/best_model_122feat_engineered.pth')
            print(f"  ✅ Nouveau meilleur modèle sauvegardé (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            print(f"\n⚠️  Early stopping à epoch {epoch+1}")
            break
    
    # Test final
    print("\n" + "="*85)
    print("ÉVALUATION FINALE")
    print("="*85)
    
    model.load_state_dict(torch.load(f'{config.models_dir}/best_model_122feat_engineered.pth'))
    test_loss, test_auc_macro, test_auc_per_class = evaluate(
        model, test_loader, criterion, config.device
    )
    
    # AUC micro (moyenne pondérée)
    all_preds_test = []
    all_labels_test = []
    model.eval()
    with torch.no_grad():
        for X, W, y in test_loader:
            X, W, y = X.to(config.device), W.to(config.device), y.to(config.device)
            logits = model(X, W)  # Ordre correct: (X_signal, W_features)
            probs = torch.sigmoid(logits)
            all_preds_test.append(probs.cpu().numpy())
            all_labels_test.append(y.cpu().numpy())
    
    all_preds_test = np.vstack(all_preds_test)
    all_labels_test = np.vstack(all_labels_test)
    
    test_auc_micro = roc_auc_score(
        all_labels_test.ravel(), 
        all_preds_test.ravel()
    )
    
    print(f"\n📊 RÉSULTATS (Best Epoch: {best_epoch}):")
    print(f"  • Val AUC (macro) : {best_val_auc:.4f}")
    print(f"  • Test AUC (macro): {test_auc_macro:.4f}")
    print(f"  • Test AUC (micro): {test_auc_micro:.4f}")
    print()
    
    print("📈 AUC par classe (Test):")
    for i, auc in enumerate(test_auc_per_class):
        print(f"  • Classe {i}: {auc:.4f}")
    
    print("\n" + "="*85)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*85)
    
    # Sauvegarde résultats
    results = {
        'config': {
            'task_mode': config.task_mode,
            'wide_features': config.wide_dir,
            'wide_input_dim': data['wide_input_dim'],
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'dropout': config.dropout
        },
        'results': {
            'best_epoch': best_epoch,
            'val_auc_macro': float(best_val_auc),
            'test_auc_macro': float(test_auc_macro),
            'test_auc_micro': float(test_auc_micro),
            'test_auc_per_class': [float(x) for x in test_auc_per_class]
        }
    }
    
    with open(f'{config.models_dir}/results_122feat_engineered.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Résultats sauvegardés: {config.models_dir}/results_122feat_engineered.json")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    config = Config()
    train_model(config)
