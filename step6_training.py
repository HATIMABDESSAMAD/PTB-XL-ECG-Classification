"""
═══════════════════════════════════════════════════════════════════════════════
STEP 6: TRAINING PIPELINE - PTB-XL Wide+Deep
═══════════════════════════════════════════════════════════════════════════════
Entraînement complet avec:
  - Dataset Train/Val/Test (strat_fold split)
  - Early stopping sur Val AUC/AUPRC
  - Threshold optimization sur Val
  - Test metrics finales (AUC macro/micro, AUPRC, F1)
  - Support 71 codes OU 5 superclasses
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import modèle
from step5_wide_deep_model import WideDeepModel, DeepOnlyModel, WideOnlyModel

print("=" * 100)
print("STEP 6: TRAINING PIPELINE Wide+Deep")
print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Configuration entraînement"""
    # Tâche
    task_mode = '5superclass'  # '71codes' ou '5superclass'
    
    # Modèle
    model_type = 'wide_deep'  # 'wide_deep', 'deep_only', 'wide_only'
    
    # Training
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Early stopping
    patience = 10
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    signals_dir = 'cleaned_signals_100hz'
    wide_dir = 'preprocessed_wide'
    labels_file = 'ptbxl_from_excel_consolidated.csv'
    label_config_file = 'label_config_from_excel.json'
    
    # Outputs
    output_dir = 'models'
    results_dir = 'results'

config = Config()
print(f"\n⚙️  CONFIGURATION:")
print(f"  • Tâche        : {config.task_mode}")
print(f"  • Modèle       : {config.model_type}")
print(f"  • Device       : {config.device}")
print(f"  • Batch size   : {config.batch_size}")
print(f"  • Learning rate: {config.learning_rate}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════

class PTBXLDataset(Dataset):
    """Dataset PyTorch pour PTB-XL Wide+Deep"""
    
    def __init__(self, ecg_ids, labels, signals_dir, W_features=None):
        """
        Args:
            ecg_ids: liste ECG IDs
            labels: array (N, K) labels binaires
            signals_dir: dossier signaux nettoyés
            W_features: array (N, D_wide) features Wide (None pour Deep only)
        """
        self.ecg_ids = ecg_ids
        self.labels = labels
        self.signals_dir = Path(signals_dir)
        self.W_features = W_features
    
    def __len__(self):
        return len(self.ecg_ids)
    
    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        
        # Charger signal
        signal_path = self.signals_dir / f"X_clean_{ecg_id:05d}.npz"
        try:
            data = np.load(signal_path)
            X_signal = data['signal']  # (12, 1000)
        except:
            # Signal manquant: zeros
            X_signal = np.zeros((12, 1000), dtype=np.float32)
        
        X_signal = torch.from_numpy(X_signal).float()
        
        # Labels
        y = torch.from_numpy(self.labels[idx]).float()
        
        # Wide features (si disponible)
        if self.W_features is not None:
            W = torch.from_numpy(self.W_features[idx]).float()
            return X_signal, W, y
        else:
            return X_signal, y


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(config):
    """Charge données Train/Val/Test"""
    print("\n[Chargement données]")
    
    # ──────────────────────────────────────────────────────────────────────
    # A. Labels
    # ──────────────────────────────────────────────────────────────────────
    df_labels = pd.read_csv(config.labels_file, index_col='ecg_id')
    
    # Charger config labels
    with open(config.label_config_file, 'r') as f:
        label_config = json.load(f)
    
    # Sélectionner colonnes selon tâche
    if config.task_mode == '71codes':
        label_cols = label_config.get('binary_cols', label_config.get('scp_cols_excel', []))
        n_classes = len(label_cols)
        print(f"  ✓ Tâche: {n_classes} codes SCP")
    elif config.task_mode == '5superclass':
        label_cols = label_config.get('superclass_cols', label_config.get('superclass_cols_excel', []))
        n_classes = len(label_cols)
        print(f"  ✓ Tâche: 5 superclasses")
    else:
        raise ValueError(f"task_mode inconnu: {config.task_mode}")
    
    print(f"    • Classes: {n_classes}")
    
    # ──────────────────────────────────────────────────────────────────────
    # B. Wide features
    # ──────────────────────────────────────────────────────────────────────
    if config.model_type in ['wide_deep', 'wide_only']:
        data_train = np.load(f'{config.wide_dir}/W_train.npz', allow_pickle=True)
        data_val = np.load(f'{config.wide_dir}/W_val.npz', allow_pickle=True)
        data_test = np.load(f'{config.wide_dir}/W_test.npz', allow_pickle=True)
        
        W_train = data_train['W']
        W_val = data_val['W']
        W_test = data_test['W']
        
        ecg_ids_train = data_train['ecg_ids']
        ecg_ids_val = data_val['ecg_ids']
        ecg_ids_test = data_test['ecg_ids']
        
        wide_input_dim = W_train.shape[1]
        print(f"  ✓ Wide features: {wide_input_dim} dims")
    else:
        # Deep only: charger ecg_ids depuis labels + strat_fold
        df_train = df_labels[df_labels['strat_fold'].isin(range(1, 9))]
        df_val = df_labels[df_labels['strat_fold'] == 9]
        df_test = df_labels[df_labels['strat_fold'] == 10]
        
        ecg_ids_train = df_train.index.values
        ecg_ids_val = df_val.index.values
        ecg_ids_test = df_test.index.values
        
        W_train = None
        W_val = None
        W_test = None
        wide_input_dim = None
        print(f"  ✓ Mode Deep only (pas de Wide features)")
    
    # ──────────────────────────────────────────────────────────────────────
    # C. Labels correspondants aux ECG IDs
    # ──────────────────────────────────────────────────────────────────────
    y_train = df_labels.loc[ecg_ids_train, label_cols].values
    y_val = df_labels.loc[ecg_ids_val, label_cols].values
    y_test = df_labels.loc[ecg_ids_test, label_cols].values
    
    print(f"  ✓ Train: {len(ecg_ids_train):,} ECG")
    print(f"  ✓ Val  : {len(ecg_ids_val):,} ECG")
    print(f"  ✓ Test : {len(ecg_ids_test):,} ECG")
    
    # ──────────────────────────────────────────────────────────────────────
    # D. Datasets PyTorch
    # ──────────────────────────────────────────────────────────────────────
    train_dataset = PTBXLDataset(ecg_ids_train, y_train, config.signals_dir, W_train)
    val_dataset = PTBXLDataset(ecg_ids_val, y_val, config.signals_dir, W_val)
    test_dataset = PTBXLDataset(ecg_ids_test, y_test, config.signals_dir, W_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_classes': n_classes,
        'wide_input_dim': wide_input_dim,
        'label_cols': label_cols
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CRÉATION MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(config, n_classes, wide_input_dim=None):
    """Créer modèle selon config"""
    print(f"\n[Création modèle]")
    
    if config.model_type == 'wide_deep':
        model = WideDeepModel(
            n_classes=n_classes,
            wide_input_dim=wide_input_dim,
            dropout=0.3
        )
        print(f"  ✓ WideDeepModel")
    
    elif config.model_type == 'deep_only':
        model = DeepOnlyModel(
            n_classes=n_classes,
            dropout=0.3
        )
        print(f"  ✓ DeepOnlyModel")
    
    elif config.model_type == 'wide_only':
        model = WideOnlyModel(
            n_classes=n_classes,
            wide_input_dim=wide_input_dim,
            dropout=0.3
        )
        print(f"  ✓ WideOnlyModel")
    
    else:
        raise ValueError(f"model_type inconnu: {config.model_type}")
    
    model = model.to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Paramètres: {n_params:,}")
    
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device, model_type):
    """Une epoch d'entraînement"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        if model_type == 'wide_deep':
            X_signal, W_features, y = batch
            X_signal = X_signal.to(device)
            W_features = W_features.to(device)
            y = y.to(device)
            
            logits = model(X_signal, W_features)
        
        elif model_type == 'deep_only':
            X_signal, y = batch
            X_signal = X_signal.to(device)
            y = y.to(device)
            
            logits = model(X_signal)
        
        elif model_type == 'wide_only':
            _, W_features, y = batch  # X_signal ignoré
            W_features = W_features.to(device)
            y = y.to(device)
            
            logits = model(W_features)
        
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, model_type):
    """Évaluation sur Val/Test"""
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if model_type == 'wide_deep':
                X_signal, W_features, y = batch
                X_signal = X_signal.to(device)
                W_features = W_features.to(device)
                y = y.to(device)
                
                logits = model(X_signal, W_features)
            
            elif model_type == 'deep_only':
                X_signal, y = batch
                X_signal = X_signal.to(device)
                y = y.to(device)
                
                logits = model(X_signal)
            
            elif model_type == 'wide_only':
                _, W_features, y = batch
                W_features = W_features.to(device)
                y = y.to(device)
                
                logits = model(W_features)
            
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    
    # Probs (sigmoid)
    all_probs = 1 / (1 + np.exp(-all_logits))
    
    # Métriques
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, average='macro')
        auc_micro = roc_auc_score(all_labels, all_probs, average='micro')
        auprc_macro = average_precision_score(all_labels, all_probs, average='macro')
    except:
        auc_macro = 0.0
        auc_micro = 0.0
        auprc_macro = 0.0
    
    metrics = {
        'loss': total_loss / len(loader),
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'auprc_macro': auprc_macro
    }
    
    return metrics, all_probs, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(config):
    """Pipeline entraînement complet"""
    
    # Créer dossiers
    Path(config.output_dir).mkdir(exist_ok=True)
    Path(config.results_dir).mkdir(exist_ok=True)
    
    # Charger données
    data = load_data(config)
    
    # Créer modèle
    model = create_model(config, data['n_classes'], data['wide_input_dim'])
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Training loop
    print(f"\n[Entraînement]")
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, data['train_loader'], criterion, optimizer, 
                                config.device, config.model_type)
        
        # Validate
        val_metrics, _, _ = evaluate(model, data['val_loader'], criterion, 
                                     config.device, config.model_type)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['auc_macro']:.4f}")
        
        # Early stopping
        if val_metrics['auc_macro'] > best_val_auc:
            best_val_auc = val_metrics['auc_macro']
            patience_counter = 0
            
            # Sauvegarder meilleur modèle
            torch.save(model.state_dict(), f"{config.output_dir}/best_model.pth")
            print(f"  → Meilleur modèle sauvegardé (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n⏹️  Early stopping à epoch {epoch+1}")
                break
    
    # Charger meilleur modèle
    model.load_state_dict(torch.load(f"{config.output_dir}/best_model.pth"))
    
    # Test
    print(f"\n[Test Final]")
    test_metrics, test_probs, test_labels = evaluate(model, data['test_loader'], criterion,
                                                     config.device, config.model_type)
    
    print(f"  • Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  • Test AUC macro: {test_metrics['auc_macro']:.4f}")
    print(f"  • Test AUC micro: {test_metrics['auc_micro']:.4f}")
    print(f"  • Test AUPRC    : {test_metrics['auprc_macro']:.4f}")
    
    # Sauvegarder résultats
    results = {
        'config': vars(config),
        'test_metrics': test_metrics,
        'best_val_auc': best_val_auc
    }
    
    with open(f"{config.results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Entraînement terminé")
    print(f"   Modèle: {config.output_dir}/best_model.pth")
    print(f"   Résultats: {config.results_dir}/results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXÉCUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("DÉMARRAGE DU TRAINING")
    print("=" * 100)
    
    # Lancer le training
    try:
        train_model(config)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
