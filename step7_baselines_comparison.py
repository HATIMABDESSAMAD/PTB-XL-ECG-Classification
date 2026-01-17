"""
STEP 7: COMPARAISON DES BASELINES
==================================
Compare 4 approches:
1. XGBoost (Wide only - features tabulaires)
2. Deep only (CNN+Transformer - signaux uniquement)
3. Wide only (MLP - features tabulaires)
4. Wide+Deep (modèle complet)
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
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import modèles
from step5_wide_deep_model import DeepBranch, WideBranch, WideDeepModel

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Données
    labels_file = 'ptbxl_from_excel_consolidated.csv'
    label_config_file = 'label_config_from_excel.json'
    wide_dir = 'all_features_engineered'
    signals_dir = 'cleaned_signals_100hz'
    models_dir = 'models'
    
    # Hyperparamètres
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    early_stopping_patience = 10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

config = Config()

# ============================================================================
# DATASET
# ============================================================================

class WideOnlyDataset(Dataset):
    """Dataset pour Wide only (features uniquement)"""
    def __init__(self, wide_features, labels):
        self.W = torch.FloatTensor(wide_features)
        self.y = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.W)
    
    def __getitem__(self, idx):
        return self.W[idx], self.y[idx]

class DeepOnlyDataset(Dataset):
    """Dataset pour Deep only (signaux uniquement)"""
    def __init__(self, signal_ids, labels, filenames):
        self.signal_ids = signal_ids
        self.filenames = filenames
        self.y = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.signal_ids)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        signal_id = filename.split('/')[-1]
        signal_path = Path(config.signals_dir) / f'X_clean_{signal_id.replace("_lr", "")}.npz'
        signal_data = np.load(signal_path)
        X = signal_data['signal']
        X = torch.FloatTensor(X)
        return X, self.y[idx]

class WideDeepDataset(Dataset):
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
        signal_id = filename.split('/')[-1]
        signal_path = Path(config.signals_dir) / f'X_clean_{signal_id.replace("_lr", "")}.npz'
        signal_data = np.load(signal_path)
        X = signal_data['signal']
        X = torch.FloatTensor(X)
        return X, W, self.y[idx]

# ============================================================================
# FONCTIONS COMMUNES
# ============================================================================

def load_data():
    """Charge toutes les données"""
    print("\n[1/7] Chargement des données...")
    
    # Labels
    df = pd.read_csv(config.labels_file)
    label_config = json.load(open(config.label_config_file))
    label_cols = label_config.get('superclass_cols', label_config.get('superclass_cols_excel', []))
    
    # Wide features
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
    
    # Labels alignés
    df_train = df[df['ecg_id'].isin(ids_train)].set_index('ecg_id').loc[ids_train].reset_index()
    df_val = df[df['ecg_id'].isin(ids_val)].set_index('ecg_id').loc[ids_val].reset_index()
    df_test = df[df['ecg_id'].isin(ids_test)].set_index('ecg_id').loc[ids_test].reset_index()
    
    y_train = df_train[label_cols].values
    y_val = df_val[label_cols].values
    y_test = df_test[label_cols].values
    
    print(f"  ✓ Train: {len(W_train)} ECG")
    print(f"  ✓ Val  : {len(W_val)} ECG")
    print(f"  ✓ Test : {len(W_test)} ECG")
    
    return {
        'W_train': W_train, 'W_val': W_val, 'W_test': W_test,
        'ids_train': ids_train, 'ids_val': ids_val, 'ids_test': ids_test,
        'filenames_train': filenames_train, 'filenames_val': filenames_val, 'filenames_test': filenames_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'n_classes': len(label_cols),
        'wide_input_dim': W_train.shape[1]
    }

def evaluate_torch(model, loader, device):
    """Évalue un modèle PyTorch"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            if len(batch) == 2:  # Wide only ou Deep only
                X, y = batch
                X, y = X.to(device), y.to(device)
                logits = model(X)
            else:  # Wide+Deep
                X, W, y = batch
                X, W, y = X.to(device), W.to(device), y.to(device)
                logits = model(X, W)
            
            probs = torch.sigmoid(logits)
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
    
    auc_macro = np.mean(auc_scores)
    auc_micro = roc_auc_score(all_labels.ravel(), all_preds.ravel())
    
    return auc_macro, auc_micro, auc_scores

# ============================================================================
# BASELINE 1: XGBoost (Wide only)
# ============================================================================

def train_xgboost(data):
    """Entraîne XGBoost sur features tabulaires"""
    print("\n[2/7] Baseline 1: XGBoost (Wide only)...")
    
    results = {}
    
    for i in range(data['n_classes']):
        print(f"  Classe {i}...")
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        
        model.fit(data['W_train'], data['y_train'][:, i])
        pred_val = model.predict_proba(data['W_val'])[:, 1]
        pred_test = model.predict_proba(data['W_test'])[:, 1]
        
        results[f'class_{i}'] = {
            'pred_val': pred_val,
            'pred_test': pred_test
        }
    
    # Calculer AUC
    all_preds_val = np.column_stack([results[f'class_{i}']['pred_val'] for i in range(data['n_classes'])])
    all_preds_test = np.column_stack([results[f'class_{i}']['pred_test'] for i in range(data['n_classes'])])
    
    auc_val_macro = np.mean([roc_auc_score(data['y_val'][:, i], all_preds_val[:, i]) 
                             for i in range(data['n_classes']) if len(np.unique(data['y_val'][:, i])) > 1])
    auc_val_micro = roc_auc_score(data['y_val'].ravel(), all_preds_val.ravel())
    
    auc_test_macro = np.mean([roc_auc_score(data['y_test'][:, i], all_preds_test[:, i]) 
                              for i in range(data['n_classes']) if len(np.unique(data['y_test'][:, i])) > 1])
    auc_test_micro = roc_auc_score(data['y_test'].ravel(), all_preds_test.ravel())
    
    print(f"  ✓ Val AUC macro : {auc_val_macro:.4f}")
    print(f"  ✓ Test AUC macro: {auc_test_macro:.4f}")
    print(f"  ✓ Test AUC micro: {auc_test_micro:.4f}")
    
    return {
        'name': 'XGBoost (Wide only)',
        'val_auc_macro': auc_val_macro,
        'val_auc_micro': auc_val_micro,
        'test_auc_macro': auc_test_macro,
        'test_auc_micro': auc_test_micro
    }

# ============================================================================
# BASELINE 2: Deep only
# ============================================================================

def train_deep_only(data):
    """Entraîne Deep only (signaux uniquement)"""
    print("\n[3/7] Baseline 2: Deep only (CNN+Transformer)...")
    
    # Datasets
    train_dataset = DeepOnlyDataset(data['ids_train'], data['y_train'], data['filenames_train'])
    val_dataset = DeepOnlyDataset(data['ids_val'], data['y_val'], data['filenames_val'])
    test_dataset = DeepOnlyDataset(data['ids_test'], data['y_test'], data['filenames_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Modèle
    deep_branch = DeepBranch(n_leads=12, seq_len=1000, d_model=256, 
                            transformer_heads=8, transformer_layers=8, 
                            deep_features_dim=64, dropout=0.1).to(config.device)
    
    classifier = nn.Linear(64, data['n_classes']).to(config.device)
    
    model = nn.Sequential(deep_branch, classifier)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Entraînement
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        for X, y in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config.num_epochs}", leave=False):
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_auc_macro, _, _ = evaluate_torch(model, val_loader, config.device)
        
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            patience_counter = 0
            torch.save(model.state_dict(), f'{config.models_dir}/deep_only_best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            break
    
    # Test
    model.load_state_dict(torch.load(f'{config.models_dir}/deep_only_best.pth'))
    test_auc_macro, test_auc_micro, _ = evaluate_torch(model, test_loader, config.device)
    
    print(f"  ✓ Val AUC macro : {best_val_auc:.4f}")
    print(f"  ✓ Test AUC macro: {test_auc_macro:.4f}")
    print(f"  ✓ Test AUC micro: {test_auc_micro:.4f}")
    
    return {
        'name': 'Deep only (CNN+Transformer)',
        'val_auc_macro': best_val_auc,
        'test_auc_macro': test_auc_macro,
        'test_auc_micro': test_auc_micro
    }

# ============================================================================
# BASELINE 3: Wide only (MLP)
# ============================================================================

def train_wide_only(data):
    """Entraîne Wide only (MLP sur features)"""
    print("\n[4/7] Baseline 3: Wide only (MLP)...")
    
    # Datasets
    train_dataset = WideOnlyDataset(data['W_train'], data['y_train'])
    val_dataset = WideOnlyDataset(data['W_val'], data['y_val'])
    test_dataset = WideOnlyDataset(data['W_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Modèle
    model = WideBranch(
        input_dim=data['wide_input_dim'],
        hidden_dims=[256, 128],
        wide_features_dim=64,
        dropout=0.1
    ).to(config.device)
    
    classifier = nn.Linear(64, data['n_classes']).to(config.device)
    model = nn.Sequential(model, classifier)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Entraînement
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        for W, y in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config.num_epochs}", leave=False):
            W, y = W.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            logits = model(W)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_auc_macro, _, _ = evaluate_torch(model, val_loader, config.device)
        
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            patience_counter = 0
            torch.save(model.state_dict(), f'{config.models_dir}/wide_only_best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            break
    
    # Test
    model.load_state_dict(torch.load(f'{config.models_dir}/wide_only_best.pth'))
    test_auc_macro, test_auc_micro, _ = evaluate_torch(model, test_loader, config.device)
    
    print(f"  ✓ Val AUC macro : {best_val_auc:.4f}")
    print(f"  ✓ Test AUC macro: {test_auc_macro:.4f}")
    print(f"  ✓ Test AUC micro: {test_auc_micro:.4f}")
    
    return {
        'name': 'Wide only (MLP)',
        'val_auc_macro': best_val_auc,
        'test_auc_macro': test_auc_macro,
        'test_auc_micro': test_auc_micro
    }

# ============================================================================
# BASELINE 4: Wide+Deep (déjà entraîné)
# ============================================================================

def load_wide_deep_results():
    """Charge les résultats Wide+Deep"""
    print("\n[5/7] Baseline 4: Wide+Deep (déjà entraîné)...")
    
    results_file = f'{config.models_dir}/results_122feat_engineered.json'
    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"  ✓ Val AUC macro : {results['results']['val_auc_macro']:.4f}")
        print(f"  ✓ Test AUC macro: {results['results']['test_auc_macro']:.4f}")
        print(f"  ✓ Test AUC micro: {results['results']['test_auc_micro']:.4f}")
        
        return {
            'name': 'Wide+Deep (122 feat)',
            'val_auc_macro': results['results']['val_auc_macro'],
            'test_auc_macro': results['results']['test_auc_macro'],
            'test_auc_micro': results['results']['test_auc_micro']
        }
    else:
        print("  ⚠️  Résultats Wide+Deep non trouvés!")
        return None

# ============================================================================
# COMPARAISON
# ============================================================================

def compare_results(all_results):
    """Compare tous les résultats"""
    print("\n[6/7] Comparaison des résultats...")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('test_auc_macro', ascending=False)
    
    print("\n" + "="*85)
    print("TABLEAU COMPARATIF")
    print("="*85)
    print(df_results.to_string(index=False))
    
    # Sauvegarde
    df_results.to_csv(f'{config.models_dir}/baselines_comparison.csv', index=False)
    
    with open(f'{config.models_dir}/baselines_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Sauvegardé: {config.models_dir}/baselines_comparison.csv")
    print(f"💾 Sauvegardé: {config.models_dir}/baselines_comparison.json")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*85)
    print("STEP 7: COMPARAISON BASELINES")
    print("="*85)
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Charger données
    data = load_data()
    
    # Entraîner baselines
    all_results = []
    
    # 1. XGBoost
    result_xgb = train_xgboost(data)
    all_results.append(result_xgb)
    
    # 2. Deep only
    result_deep = train_deep_only(data)
    all_results.append(result_deep)
    
    # 3. Wide only
    result_wide = train_wide_only(data)
    all_results.append(result_wide)
    
    # 4. Wide+Deep
    result_wide_deep = load_wide_deep_results()
    if result_wide_deep:
        all_results.append(result_wide_deep)
    
    # Comparaison
    compare_results(all_results)
    
    print("\n" + "="*85)
    print("✅ STEP 7 TERMINÉ")
    print("="*85)

if __name__ == '__main__':
    main()
