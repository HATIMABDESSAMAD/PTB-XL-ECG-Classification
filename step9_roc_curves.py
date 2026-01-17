"""
═══════════════════════════════════════════════════════════════════════════════
STEP 9: COURBES ROC POUR LES 5 SUPERCLASSES - MEILLEUR MODELE (PURE)
═══════════════════════════════════════════════════════════════════════════════
Génère les courbes ROC individuelles pour chaque superclasse + courbe combinée
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_COLORS = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

print("="*80)
print("STEP 9: COURBES ROC - 5 SUPERCLASSES (Meilleur Modele Pure)")
print("="*80)
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MAPPING SCP CODES → SUPERCLASSES
# ═══════════════════════════════════════════════════════════════════════════════

# Charger le mapping depuis scp_statements.csv
import pandas as pd
scp_df = pd.read_csv('scp_statements.csv', index_col=0)
# Créer le dictionnaire de mapping code → superclass
SCP_TO_SUPERCLASS = {}
for code in scp_df.index:
    if pd.notna(scp_df.loc[code, 'diagnostic_class']):
        SCP_TO_SUPERCLASS[code] = scp_df.loc[code, 'diagnostic_class']

print(f"  Mapping: {len(SCP_TO_SUPERCLASS)} codes SCP → 5 superclasses")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DÉFINITION DU DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class WideDeepDataset(Dataset):
    def __init__(self, signal_dir, wide_dir, split):
        wide_data = np.load(wide_dir / f'W_pure_{split}.npz', allow_pickle=True)
        self.W = torch.FloatTensor(wide_data['W'])
        self.ecg_ids = wide_data['ecg_ids']
        
        import pandas as pd
        df_labels = pd.read_csv('ptbxl_database.csv', usecols=['ecg_id', 'scp_codes'])
        df_labels = df_labels.set_index('ecg_id')
        
        self.signal_dir = signal_dir
        self.samples = []
        
        for ecg_id, W_feat in zip(self.ecg_ids, self.W):
            signal_file = signal_dir / f'X_clean_{ecg_id:05d}.npz'
            
            if signal_file.exists() and ecg_id in df_labels.index:
                scp_codes = eval(df_labels.loc[ecg_id, 'scp_codes'])
                y_multi = np.zeros(5, dtype=np.int64)
                
                # Mapper chaque code vers sa superclasse
                for code, likelihood in scp_codes.items():
                    if code in SCP_TO_SUPERCLASS:
                        superclass = SCP_TO_SUPERCLASS[code]
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
        
        print(f"  [Dataset] {split.upper()}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample['signal_file'], allow_pickle=True)
        X = torch.FloatTensor(data['signal'])
        y = torch.LongTensor(sample['y'])
        return X, sample['W'], y

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DÉFINITION DU MODÈLE
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
# 3. CHARGER LE MODÈLE ET FAIRE LES PRÉDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/4] Chargement du modele...")

model = WideDeepModel(num_wide_features=32, num_classes=5)
checkpoint = torch.load('model_wide_deep_pure.pth', map_location=DEVICE, weights_only=True)
# Le checkpoint est directement le state_dict
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print("  Modele charge avec succes!")

print("\n[2/4] Chargement des donnees de test...")
signal_dir = Path('cleaned_signals_100hz')
wide_dir = Path('wide_features_clean')
test_dataset = WideDeepDataset(signal_dir, wide_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n[3/4] Generation des predictions...")
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
print(f"  Predictions: {all_probs.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CALCULER LES COURBES ROC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/4] Generation des courbes ROC...")

# Figure 1: Courbes ROC individuelles (5 subplots)
fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

roc_data = {}
for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    ax = axes[i]
    ax.plot(fpr, tpr, color=color, lw=3, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.3, color=color)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificite)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensibilite)', fontsize=11)
    ax.set_title(f'Courbe ROC - {name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ajouter point optimal (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], marker='o', s=100, 
               c='red', edgecolors='black', zorder=5, label=f'Optimal: ({fpr[best_idx]:.2f}, {tpr[best_idx]:.2f})')
    ax.annotate(f'Seuil: {thresholds[best_idx]:.2f}', 
                xy=(fpr[best_idx], tpr[best_idx]),
                xytext=(fpr[best_idx]+0.1, tpr[best_idx]-0.1),
                fontsize=10, color='red')

# Cacher le 6ème subplot (vide)
axes[5].axis('off')
axes[5].text(0.5, 0.5, 'Wide+Deep Pure\n94.29% AUC Macro\n\n32 Excel + Signal\n= 96 Features', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Courbes ROC par Superclasse - Modele Wide+Deep Pure', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('roc_curves_per_class.png', dpi=150, bbox_inches='tight')
print("  Sauvegarde: roc_curves_per_class.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Toutes les courbes ROC sur un seul graphique
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(10, 10))

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    fpr, tpr = roc_data[name]['fpr'], roc_data[name]['tpr']
    roc_auc = roc_data[name]['auc']
    ax2.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.4f})')

# Courbe micro-average
fpr_micro, tpr_micro, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
auc_micro = auc(fpr_micro, tpr_micro)
ax2.plot(fpr_micro, tpr_micro, color='black', lw=3, linestyle='--', 
         label=f'Micro-avg (AUC = {auc_micro:.4f})')

ax2.plot([0, 1], [0, 1], 'gray', lw=2, linestyle=':', alpha=0.5, label='Random (AUC = 0.50)')

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate (1 - Specificite)', fontsize=13)
ax2.set_ylabel('True Positive Rate (Sensibilite)', fontsize=13)
ax2.set_title('Courbes ROC Combinees - 5 Superclasses\nModele Wide+Deep Pure (94.29% AUC Macro)', 
              fontsize=15, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

# Zone AUC moyenne
ax2.axhspan(0.9, 1.0, alpha=0.1, color='green')
ax2.text(0.5, 0.95, 'Excellente Performance', ha='center', fontsize=10, color='darkgreen')

plt.tight_layout()
plt.savefig('roc_curves_combined.png', dpi=150, bbox_inches='tight')
print("  Sauvegarde: roc_curves_combined.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Matrice de confusion pour chaque classe (seuil=0.5)
# ═══════════════════════════════════════════════════════════════════════════════

from sklearn.metrics import confusion_matrix
import seaborn as sns

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
axes3 = axes3.flatten()

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    y_pred = (all_probs[:, i] >= 0.5).astype(int)
    cm = confusion_matrix(all_labels[:, i], y_pred)
    
    ax = axes3[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'],
                annot_kws={'size': 14})
    ax.set_xlabel('Prediction', fontsize=11)
    ax.set_ylabel('Verite', fontsize=11)
    ax.set_title(f'Confusion Matrix - {name}\nAUC: {roc_data[name]["auc"]:.4f}', fontsize=12, fontweight='bold')

# Stats dans le 6ème subplot
axes3[5].axis('off')
stats_text = "RESUME PERFORMANCES\n" + "="*30 + "\n\n"
auc_macro = np.mean([roc_data[name]['auc'] for name in CLASS_NAMES])
for name in CLASS_NAMES:
    stats_text += f"{name}: {roc_data[name]['auc']*100:.2f}%\n"
stats_text += f"\n" + "="*30 + f"\nAUC MACRO: {auc_macro*100:.2f}%"
axes3[5].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Matrices de Confusion par Superclasse (Seuil = 0.5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices_per_class.png', dpi=150, bbox_inches='tight')
print("  Sauvegarde: confusion_matrices_per_class.png")

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RESUME DES PERFORMANCES PAR CLASSE")
print("="*80)

print("\n+----------+-----------+-----------+-----------+-----------+")
print(f"| {'Classe':<8} | {'AUC':>9} | {'Sensib.':>9} | {'Specif.':>9} | {'F1-Score':>9} |")
print("+----------+-----------+-----------+-----------+-----------+")

from sklearn.metrics import precision_recall_fscore_support

for i, name in enumerate(CLASS_NAMES):
    y_pred = (all_probs[:, i] >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels[:, i], y_pred, average='binary', zero_division=0)
    
    # Specificite
    tn = np.sum((all_labels[:, i] == 0) & (y_pred == 0))
    fp = np.sum((all_labels[:, i] == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"| {name:<8} | {roc_data[name]['auc']*100:>8.2f}% | {recall*100:>8.2f}% | {specificity*100:>8.2f}% | {f1:>9.4f} |")

print("+----------+-----------+-----------+-----------+-----------+")
print(f"| {'MACRO':8} | {auc_macro*100:>8.2f}% |           |           |           |")
print("+----------+-----------+-----------+-----------+-----------+")

print("\n" + "="*80)
print("3 figures generees avec succes!")
print("  1. roc_curves_per_class.png")
print("  2. roc_curves_combined.png")
print("  3. confusion_matrices_per_class.png")
print("="*80)

plt.show()
