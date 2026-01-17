"""
═══════════════════════════════════════════════════════════════════════════════
STEP 9B: COURBES ROC - UTILISANT LE MEME MAPPING QUE L'ENTRAINEMENT
═══════════════════════════════════════════════════════════════════════════════
Le modele a ete entraine avec les codes exacts NORM/MI/STTC/CD/HYP dans scp_codes
On doit utiliser le meme mapping pour l'evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_COLORS = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

print("="*80)
print("STEP 9B: COURBES ROC - 5 SUPERCLASSES")
print("="*80)
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGER LE MAPPING SCP → SUPERCLASS (comme dans l'entrainement original)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Chargement du mapping SCP -> Superclass...")

scp_df = pd.read_csv('scp_statements.csv', index_col=0)

# Mapping code → superclass (diagnostic_class)
SCP_TO_SUPERCLASS = {}
for code in scp_df.index:
    if pd.notna(scp_df.loc[code, 'diagnostic_class']):
        SCP_TO_SUPERCLASS[code] = scp_df.loc[code, 'diagnostic_class']

print(f"  {len(SCP_TO_SUPERCLASS)} codes SCP mappes vers superclasses")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET avec mapping correct
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepDataset(Dataset):
    def __init__(self, signal_dir, wide_dir, split, scp_mapping):
        wide_data = np.load(wide_dir / f'W_pure_{split}.npz', allow_pickle=True)
        self.W = torch.FloatTensor(wide_data['W'])
        self.ecg_ids = wide_data['ecg_ids']
        
        df_labels = pd.read_csv('ptbxl_database.csv', usecols=['ecg_id', 'scp_codes'])
        df_labels = df_labels.set_index('ecg_id')
        
        self.signal_dir = signal_dir
        self.samples = []
        
        for ecg_id, W_feat in zip(self.ecg_ids, self.W):
            signal_file = signal_dir / f'X_clean_{ecg_id:05d}.npz'
            
            if signal_file.exists() and ecg_id in df_labels.index:
                scp_codes = eval(df_labels.loc[ecg_id, 'scp_codes'])
                y_multi = np.zeros(5, dtype=np.float32)
                
                # Utiliser le mapping SCP -> Superclass
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
        
        # Afficher distribution
        labels_array = np.array([s['y'] for s in self.samples])
        print(f"  {split.upper()}: {len(self.samples)} samples")
        print(f"    Distribution: NORM={labels_array[:,0].sum():.0f}, MI={labels_array[:,1].sum():.0f}, "
              f"STTC={labels_array[:,2].sum():.0f}, CD={labels_array[:,3].sum():.0f}, HYP={labels_array[:,4].sum():.0f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample['signal_file'], allow_pickle=True)
        X = torch.FloatTensor(data['signal'])
        y = torch.FloatTensor(sample['y'])
        return X, sample['W'], y

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODELE
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepModel(nn.Module):
    def __init__(self, num_wide_features=32, num_classes=5):
        super().__init__()
        
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
        
        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, signal, wide_features):
        x = self.conv_layers(signal)
        x = x.squeeze(-1)
        x = self.cnn_to_transformer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        deep_out = self.deep_fc(x)
        wide_out = self.wide_fc(wide_features)
        combined = torch.cat([deep_out, wide_out], dim=1)
        return self.fusion(combined)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHARGER MODELE ET EVALUER
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Chargement du modele...")
model = WideDeepModel(num_wide_features=32, num_classes=5)
checkpoint = torch.load('model_wide_deep_pure.pth', map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print("  Modele charge!")

print("\n[3/5] Chargement des donnees...")
signal_dir = Path('cleaned_signals_100hz')
wide_dir = Path('wide_features_clean')
test_dataset = WideDeepDataset(signal_dir, wide_dir, 'test', SCP_TO_SUPERCLASS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n[4/5] Generation des predictions...")
all_probs = []
all_labels = []

with torch.no_grad():
    for X, W, y in tqdm(test_loader, desc='Predictions'):
        X, W = X.to(DEVICE), W.to(DEVICE)
        out = model(X, W)
        probs = torch.sigmoid(out).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)

print(f"  Shape predictions: {all_probs.shape}")
print(f"  Shape labels: {all_labels.shape}")
print(f"  Labels per class: {all_labels.sum(axis=0)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CALCULER AUC ET COURBES ROC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Generation des courbes ROC...")

# Calculer AUC par classe
auc_scores = []
roc_data = {}

for i, name in enumerate(CLASS_NAMES):
    n_pos = all_labels[:, i].sum()
    n_neg = len(all_labels) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'thresholds': thresholds}
        auc_scores.append(roc_auc)
        print(f"  {name}: AUC = {roc_auc:.4f} (n_pos={n_pos:.0f}, n_neg={n_neg:.0f})")
    else:
        print(f"  {name}: SKIP (n_pos={n_pos}, n_neg={n_neg})")
        roc_data[name] = None

auc_macro = np.mean(auc_scores) if auc_scores else 0
print(f"\n  AUC MACRO: {auc_macro:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Courbes ROC individuelles
# ═══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    ax = axes[i]
    
    if roc_data[name] is not None:
        fpr = roc_data[name]['fpr']
        tpr = roc_data[name]['tpr']
        roc_auc = roc_data[name]['auc']
        thresholds = roc_data[name]['thresholds']
        
        ax.plot(fpr, tpr, color=color, lw=3, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.3, color=color)
        
        # Point optimal (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        ax.scatter([fpr[best_idx]], [tpr[best_idx]], marker='*', s=200, 
                   c='red', edgecolors='black', zorder=5)
        ax.annotate(f'Seuil optimal: {thresholds[best_idx]:.2f}', 
                    xy=(fpr[best_idx], tpr[best_idx]),
                    xytext=(fpr[best_idx]+0.15, tpr[best_idx]-0.1),
                    fontsize=10, color='red', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Pas assez de\ndonnees', ha='center', va='center', fontsize=12)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux Faux Positifs (1 - Specificite)', fontsize=11)
    ax.set_ylabel('Taux Vrais Positifs (Sensibilite)', fontsize=11)
    ax.set_title(f'Courbe ROC - {name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

# Cellule resume
axes[5].axis('off')
summary_text = f"""RESUME PERFORMANCES
{'='*30}

Modele: Wide+Deep Pure
AUC Macro: {auc_macro*100:.2f}%

Distribution Test:
"""
for name in CLASS_NAMES:
    if roc_data[name]:
        summary_text += f"  {name}: {roc_data[name]['auc']*100:.2f}%\n"

axes[5].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Courbes ROC par Superclasse - Modele Wide+Deep Pure', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('roc_curves_per_class.png', dpi=150, bbox_inches='tight')
print("\n  Sauvegarde: roc_curves_per_class.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Toutes les courbes ROC combinees
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(10, 10))

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    if roc_data[name] is not None:
        fpr = roc_data[name]['fpr']
        tpr = roc_data[name]['tpr']
        roc_auc = roc_data[name]['auc']
        ax2.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.4f})')

# Courbe micro-average
try:
    fpr_micro, tpr_micro, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax2.plot(fpr_micro, tpr_micro, color='black', lw=3, linestyle='--', 
             label=f'Micro-avg (AUC = {auc_micro:.4f})')
except:
    pass

ax2.plot([0, 1], [0, 1], 'gray', lw=2, linestyle=':', alpha=0.5, label='Random (AUC = 0.50)')

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Taux Faux Positifs (1 - Specificite)', fontsize=13)
ax2.set_ylabel('Taux Vrais Positifs (Sensibilite)', fontsize=13)
ax2.set_title(f'Courbes ROC Combinees - 5 Superclasses\nModele Wide+Deep Pure (AUC Macro: {auc_macro*100:.2f}%)', 
              fontsize=15, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('roc_curves_combined.png', dpi=150, bbox_inches='tight')
print("  Sauvegarde: roc_curves_combined.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Matrices de confusion
# ═══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
axes3 = axes3.flatten()

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    ax = axes3[i]
    
    if roc_data[name] is not None:
        # Trouver seuil optimal
        fpr = roc_data[name]['fpr']
        tpr = roc_data[name]['tpr']
        thresholds = roc_data[name]['thresholds']
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        y_pred = (all_probs[:, i] >= best_threshold).astype(int)
        cm = confusion_matrix(all_labels[:, i], y_pred, labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negatif', 'Positif'],
                    yticklabels=['Negatif', 'Positif'],
                    annot_kws={'size': 14})
        ax.set_xlabel('Prediction', fontsize=11)
        ax.set_ylabel('Verite', fontsize=11)
        ax.set_title(f'{name}\nAUC: {roc_data[name]["auc"]:.4f} | Seuil: {best_threshold:.2f}', 
                    fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax.set_title(name, fontsize=12, fontweight='bold')

# Resume dans cellule 6
axes3[5].axis('off')
metrics_text = "METRIQUES PAR CLASSE\n" + "="*35 + "\n\n"
metrics_text += f"{'Classe':<8} {'AUC':>8} {'Sensib.':>10} {'Specif.':>10}\n"
metrics_text += "-"*38 + "\n"

for i, name in enumerate(CLASS_NAMES):
    if roc_data[name] is not None:
        # Calculer sensibilite et specificite au seuil optimal
        fpr = roc_data[name]['fpr']
        tpr = roc_data[name]['tpr']
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        sensibilite = tpr[best_idx]
        specificite = 1 - fpr[best_idx]
        
        metrics_text += f"{name:<8} {roc_data[name]['auc']*100:>7.2f}% {sensibilite*100:>9.2f}% {specificite*100:>9.2f}%\n"

metrics_text += "-"*38 + f"\n{'MACRO':<8} {auc_macro*100:>7.2f}%"

axes3[5].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11,
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Matrices de Confusion par Superclasse (Seuil Optimal)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices_per_class.png', dpi=150, bbox_inches='tight')
print("  Sauvegarde: confusion_matrices_per_class.png")

# ═══════════════════════════════════════════════════════════════════════════════
# RESUME FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RESUME DES PERFORMANCES")
print("="*80)

print(f"\n{'Classe':<10} {'AUC':>10} {'Sensibilite':>12} {'Specificite':>12}")
print("-"*46)

for i, name in enumerate(CLASS_NAMES):
    if roc_data[name] is not None:
        fpr = roc_data[name]['fpr']
        tpr = roc_data[name]['tpr']
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        print(f"{name:<10} {roc_data[name]['auc']*100:>9.2f}% {tpr[best_idx]*100:>11.2f}% {(1-fpr[best_idx])*100:>11.2f}%")

print("-"*46)
print(f"{'MACRO':<10} {auc_macro*100:>9.2f}%")

print("\n" + "="*80)
print("3 figures generees:")
print("  1. roc_curves_per_class.png")
print("  2. roc_curves_combined.png") 
print("  3. confusion_matrices_per_class.png")
print("="*80)

plt.show()
