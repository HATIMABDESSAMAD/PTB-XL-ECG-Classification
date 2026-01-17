"""
═══════════════════════════════════════════════════════════════════════════════
STEP 9C: COURBES ROC FINALES - MODELE CORRIGÉ
═══════════════════════════════════════════════════════════════════════════════
Génère les courbes ROC pour les 5 superclasses avec le modèle corrigé
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Configuration
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_COLORS = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

print("="*80)
print("STEP 9C: COURBES ROC FINALES - MODELE CORRIGÉ")
print("="*80)

# Charger prédictions sauvegardées
data = np.load('predictions_pure_FIXED.npz', allow_pickle=True)
all_probs = data['preds']
all_labels = data['labels']

print(f"\n  Shape predictions: {all_probs.shape}")
print(f"  Shape labels: {all_labels.shape}")
print(f"  Labels per class: {all_labels.sum(axis=0)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CALCULER AUC ET COURBES ROC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[ROC] Calcul des courbes ROC...")

auc_scores = []
roc_data = {}

for i, name in enumerate(CLASS_NAMES):
    n_pos = all_labels[:, i].sum()
    n_neg = len(all_labels) - n_pos
    
    fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'thresholds': thresholds}
    auc_scores.append(roc_auc)
    print(f"  {name}: AUC = {roc_auc*100:.2f}% (n_pos={n_pos:.0f}, n_neg={n_neg:.0f})")

auc_macro = np.mean(auc_scores)
print(f"\n  🎯 AUC MACRO: {auc_macro*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Courbes ROC individuelles
# ═══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.flatten()

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    ax = axes[i]
    
    fpr = roc_data[name]['fpr']
    tpr = roc_data[name]['tpr']
    roc_auc = roc_data[name]['auc']
    thresholds = roc_data[name]['thresholds']
    
    # Courbe ROC
    ax.plot(fpr, tpr, color=color, lw=3, label=f'AUC = {roc_auc*100:.2f}%')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random (50%)')
    ax.fill_between(fpr, tpr, alpha=0.3, color=color)
    
    # Point optimal (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], marker='*', s=300, 
               c='red', edgecolors='black', zorder=5, label='Seuil optimal')
    
    # Annotation seuil
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]
    ax.annotate(f'Sens: {sens*100:.1f}%\nSpec: {spec*100:.1f}%', 
                xy=(fpr[best_idx], tpr[best_idx]),
                xytext=(fpr[best_idx]+0.15, tpr[best_idx]-0.15),
                fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux Faux Positifs (1 - Spécificité)', fontsize=11)
    ax.set_ylabel('Taux Vrais Positifs (Sensibilité)', fontsize=11)
    ax.set_title(f'Courbe ROC - {name}', fontsize=14, fontweight='bold', color=color)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

# Cellule résumé
axes[5].axis('off')
summary_text = f"""
╔══════════════════════════════════════╗
║   RÉSUMÉ DES PERFORMANCES            ║
║   Modèle: Wide+Deep Pure (Corrigé)   ║
╠══════════════════════════════════════╣
║                                      ║
║   AUC MACRO: {auc_macro*100:.2f}%              ║
║                                      ║
║   Par classe:                        ║
║   • NORM:  {roc_data['NORM']['auc']*100:.2f}%                  ║
║   • MI:    {roc_data['MI']['auc']*100:.2f}%                  ║
║   • STTC:  {roc_data['STTC']['auc']*100:.2f}%                  ║
║   • CD:    {roc_data['CD']['auc']*100:.2f}%                  ║
║   • HYP:   {roc_data['HYP']['auc']*100:.2f}%                  ║
║                                      ║
╚══════════════════════════════════════╝
"""
axes[5].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=13,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.suptitle('Courbes ROC par Superclasse - Modèle Wide+Deep Pure (Corrigé)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('roc_curves_FIXED_per_class.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Sauvegardé: roc_curves_FIXED_per_class.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Toutes les courbes ROC combinées
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(12, 10))

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    fpr = roc_data[name]['fpr']
    tpr = roc_data[name]['tpr']
    roc_auc = roc_data[name]['auc']
    ax2.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc*100:.2f}%)')

# Courbe micro-average
fpr_micro, tpr_micro, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
auc_micro = auc(fpr_micro, tpr_micro)
ax2.plot(fpr_micro, tpr_micro, color='black', lw=3, linestyle='--', 
         label=f'Micro-avg (AUC = {auc_micro*100:.2f}%)')

ax2.plot([0, 1], [0, 1], 'gray', lw=2, linestyle=':', alpha=0.5, label='Random (50%)')

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Taux Faux Positifs (1 - Spécificité)', fontsize=14)
ax2.set_ylabel('Taux Vrais Positifs (Sensibilité)', fontsize=14)
ax2.set_title(f'Courbes ROC Combinées - 5 Superclasses\nModèle Wide+Deep Pure (AUC Macro: {auc_macro*100:.2f}%)', 
              fontsize=16, fontweight='bold')
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('roc_curves_FIXED_combined.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: roc_curves_FIXED_combined.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Matrices de confusion
# ═══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 11))
axes3 = axes3.flatten()

for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    ax = axes3[i]
    
    # Trouver seuil optimal
    fpr = roc_data[name]['fpr']
    tpr = roc_data[name]['tpr']
    thresholds = roc_data[name]['thresholds']
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (all_probs[:, i] >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels[:, i], y_pred, labels=[0, 1])
    
    # Calculer métriques
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Négatif', 'Positif'],
                yticklabels=['Négatif', 'Positif'],
                annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Prédiction', fontsize=12)
    ax.set_ylabel('Vérité', fontsize=12)
    ax.set_title(f'{name}\nAUC: {roc_data[name]["auc"]*100:.1f}% | Sens: {sensitivity*100:.1f}% | Spec: {specificity*100:.1f}%', 
                fontsize=12, fontweight='bold', color=color)

# Résumé dans cellule 6
axes3[5].axis('off')
metrics_text = """
╔════════════════════════════════════════════╗
║        MÉTRIQUES PAR CLASSE                ║
╠════════════════════════════════════════════╣
║  Classe   AUC     Sensib.   Spécif.        ║
║  ────────────────────────────────────────  ║"""

for i, name in enumerate(CLASS_NAMES):
    fpr = roc_data[name]['fpr']
    tpr = roc_data[name]['tpr']
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    sensibilite = tpr[best_idx]
    specificite = 1 - fpr[best_idx]
    
    metrics_text += f"\n║  {name:<6} {roc_data[name]['auc']*100:>6.2f}%   {sensibilite*100:>6.2f}%   {specificite*100:>6.2f}%    ║"

metrics_text += f"""
║  ────────────────────────────────────────  ║
║  MACRO   {auc_macro*100:>6.2f}%                        ║
╚════════════════════════════════════════════╝
"""

axes3[5].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11,
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Matrices de Confusion par Superclasse (Seuil Optimal Youden)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('confusion_matrices_FIXED.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: confusion_matrices_FIXED.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Comparaison Ancien vs Nouveau modèle
# ═══════════════════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(12, 6))

x = np.arange(len(CLASS_NAMES))
width = 0.35

# Ancien modèle (bugué)
old_auc = [94.29, 35.78, 37.76, 37.62, 33.22]

# Nouveau modèle (corrigé)
new_auc = [roc_data[name]['auc']*100 for name in CLASS_NAMES]

bars1 = ax4.bar(x - width/2, old_auc, width, label='Ancien (Bugué)', color='#e74c3c', alpha=0.7)
bars2 = ax4.bar(x + width/2, new_auc, width, label='Nouveau (Corrigé)', color='#27ae60', alpha=0.9)

# Ligne random
ax4.axhline(y=50, color='gray', linestyle='--', lw=2, label='Random (50%)')

ax4.set_xlabel('Superclasse', fontsize=14)
ax4.set_ylabel('AUC (%)', fontsize=14)
ax4.set_title('Comparaison: Ancien Modèle (Bugué) vs Nouveau Modèle (Corrigé)', 
              fontsize=16, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(CLASS_NAMES, fontsize=12)
ax4.set_ylim([0, 100])
ax4.legend(fontsize=12)
ax4.grid(True, axis='y', linestyle='--', alpha=0.7)

# Annotations
for bar1, bar2, old, new in zip(bars1, bars2, old_auc, new_auc):
    ax4.annotate(f'{old:.1f}%', xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')
    ax4.annotate(f'{new:.1f}%', xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e8449')

# Boîte résumé
old_macro = np.mean(old_auc)
new_macro = np.mean(new_auc)
improvement = new_macro - old_macro

summary_box = f"AUC Macro:\nAncien: {old_macro:.1f}%\nNouveau: {new_macro:.1f}%\nAmélioration: +{improvement:.1f}%"
ax4.text(0.02, 0.98, summary_box, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.savefig('comparison_old_vs_new.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_old_vs_new.png")

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RÉSUMÉ FINAL - COMPARAISON")
print("="*80)

print(f"\n{'Classe':<10} {'Ancien (Bugué)':>15} {'Nouveau (Corrigé)':>18} {'Δ':>10}")
print("-"*55)

for i, name in enumerate(CLASS_NAMES):
    old = old_auc[i]
    new = new_auc[i]
    delta = new - old
    print(f"{name:<10} {old:>14.2f}% {new:>17.2f}% {delta:>+9.2f}%")

print("-"*55)
print(f"{'MACRO':<10} {old_macro:>14.2f}% {new_macro:>17.2f}% {improvement:>+9.2f}%")

print("\n" + "="*80)
print("4 figures générées:")
print("  1. roc_curves_FIXED_per_class.png  - ROC par classe")
print("  2. roc_curves_FIXED_combined.png   - ROC combinées")
print("  3. confusion_matrices_FIXED.png    - Matrices de confusion")
print("  4. comparison_old_vs_new.png       - Comparaison ancien/nouveau")
print("="*80)

plt.show()
