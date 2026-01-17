"""
═══════════════════════════════════════════════════════════════════════════════
STEP 8: COMPARAISON COMPLÈTE DES APPROCHES - VISUALISATIONS
═══════════════════════════════════════════════════════════════════════════════
Génère des plots comparatifs de toutes les approches testées:
- Baseline NeuroKit2 (step6)
- Wide+Deep Redondant (step6b)
- XGBoost Baseline
- Wide+Deep Pure (step6c) ⭐
- Wide+Deep Hybride (step6d)
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Configuration matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("="*80)
print("📊 STEP 8: COMPARAISON DES PERFORMANCES - TOUTES APPROCHES")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DONNÉES DE PERFORMANCE DE TOUTES LES APPROCHES
# ═══════════════════════════════════════════════════════════════════════════════

# Résultats de toutes les approches testées
results = {
    'Baseline\nNeuroKit2\n(step6)': {
        'auc_macro': 86.41,
        'auc_micro': 97.50,
        'features_wide': 25,
        'features_deep': 64,
        'total_features': 89,
        'epochs': 25,
        'description': '25 NeuroKit2 + Signal',
        'color': '#e74c3c',  # Rouge
        'has_signal': True
    },
    'Wide+Deep\nRedondant\n(step6b)': {
        'auc_macro': 89.83,
        'auc_micro': 98.80,
        'features_wide': 122,
        'features_deep': 64,
        'total_features': 186,
        'epochs': 20,
        'description': '122 (Excel+NK2+Deep) + Signal',
        'color': '#e67e22',  # Orange
        'has_signal': True
    },
    'XGBoost\nBaseline\n(step7)': {
        'auc_macro': 90.34,
        'auc_micro': 98.90,
        'features_wide': 122,
        'features_deep': 0,
        'total_features': 122,
        'epochs': 0,
        'description': '122 tabular features (no signal)',
        'color': '#9b59b6',  # Violet
        'has_signal': False
    },
    'Wide+Deep\nHybride\n(step6d)': {
        'auc_macro': 94.25,
        'auc_micro': 99.30,
        'features_wide': 57,
        'features_deep': 64,
        'total_features': 121,
        'epochs': 19,
        'description': '57 (Excel+NK2) + Signal',
        'color': '#3498db',  # Bleu
        'has_signal': True
    },
    'Wide+Deep\nPure ⭐\n(step6c)': {
        'auc_macro': 94.29,
        'auc_micro': 99.30,
        'features_wide': 32,
        'features_deep': 64,
        'total_features': 96,
        'epochs': 22,
        'description': '32 Excel + Signal',
        'color': '#27ae60',  # Vert
        'has_signal': True
    }
}

# Ordre pour affichage (du pire au meilleur)
order = [
    'Baseline\nNeuroKit2\n(step6)',
    'Wide+Deep\nRedondant\n(step6b)',
    'XGBoost\nBaseline\n(step7)',
    'Wide+Deep\nHybride\n(step6d)',
    'Wide+Deep\nPure ⭐\n(step6c)'
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARGER HISTORIQUES D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Chargement des historiques...")

history_pure = []
history_hybrid = []

try:
    with open('history_pure.json', 'r') as f:
        history_pure = json.load(f)
    print(f"  ✓ history_pure.json: {len(history_pure)} epochs")
except:
    print("  ⚠ history_pure.json non trouvé")

try:
    with open('history_hybrid.json', 'r') as f:
        history_hybrid = json.load(f)
    print(f"  ✓ history_hybrid.json: {len(history_hybrid)} epochs")
except:
    print("  ⚠ history_hybrid.json non trouvé")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 1: COMPARAISON AUC MACRO (BAR CHART)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Génération Figure 1: Comparaison AUC Macro...")

fig1, ax1 = plt.subplots(figsize=(12, 7))

x = np.arange(len(order))
auc_values = [results[name]['auc_macro'] for name in order]
colors = [results[name]['color'] for name in order]

bars = ax1.bar(x, auc_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Ajouter valeurs sur les barres
for bar, val in zip(bars, auc_values):
    height = bar.get_height()
    ax1.annotate(f'{val:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')

# Ligne de référence pour le meilleur baseline
ax1.axhline(y=90.34, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='XGBoost Baseline (90.34%)')

ax1.set_xlabel('Approche', fontsize=13)
ax1.set_ylabel('AUC Macro (%)', fontsize=13)
ax1.set_title('🏆 Comparaison des Performances: AUC Macro par Approche', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(order, fontsize=10)
ax1.set_ylim(82, 97)
ax1.legend(loc='upper left', fontsize=11)

# Ajouter grille
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig('comparison_auc_macro.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_auc_macro.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 2: COMPARAISON AUC MACRO vs MICRO (GROUPED BAR)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Génération Figure 2: AUC Macro vs Micro...")

fig2, ax2 = plt.subplots(figsize=(14, 7))

x = np.arange(len(order))
width = 0.35

auc_macro = [results[name]['auc_macro'] for name in order]
auc_micro = [results[name]['auc_micro'] for name in order]

bars1 = ax2.bar(x - width/2, auc_macro, width, label='AUC Macro', color='#3498db', edgecolor='black')
bars2 = ax2.bar(x + width/2, auc_micro, width, label='AUC Micro', color='#e74c3c', edgecolor='black')

# Valeurs sur barres
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Approche', fontsize=13)
ax2.set_ylabel('AUC (%)', fontsize=13)
ax2.set_title('📊 Comparaison AUC Macro vs Micro par Approche', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(order, fontsize=10)
ax2.set_ylim(82, 101)
ax2.legend(loc='upper left', fontsize=12)
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('comparison_macro_vs_micro.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_macro_vs_micro.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 3: NOMBRE DE FEATURES vs PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/5] Génération Figure 3: Features vs Performance...")

fig3, ax3 = plt.subplots(figsize=(12, 8))

for name in order:
    r = results[name]
    marker = 's' if r['has_signal'] else 'o'
    size = 300 if '⭐' in name else 200
    ax3.scatter(r['total_features'], r['auc_macro'], 
               c=r['color'], s=size, marker=marker, 
               edgecolors='black', linewidths=2, alpha=0.85,
               label=name.replace('\n', ' '))

# Annotations
for name in order:
    r = results[name]
    offset = (10, 10) if 'Pure' in name else (10, -15)
    ax3.annotate(f"{r['auc_macro']:.2f}%",
                xy=(r['total_features'], r['auc_macro']),
                xytext=offset, textcoords='offset points',
                fontsize=11, fontweight='bold')

ax3.set_xlabel('Nombre Total de Features (Wide + Deep)', fontsize=13)
ax3.set_ylabel('AUC Macro (%)', fontsize=13)
ax3.set_title('🔍 Relation Features vs Performance\n(■ = avec signal, ● = sans signal)', fontsize=15, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9, ncol=2)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xlim(70, 200)
ax3.set_ylim(84, 96)

# Zone optimale
ax3.axvspan(90, 100, alpha=0.2, color='green', label='Zone Optimale')
ax3.text(95, 85, 'Zone\nOptimale', ha='center', fontsize=10, color='darkgreen', fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_features_vs_performance.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_features_vs_performance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 4: COURBES D'APPRENTISSAGE (PURE vs HYBRID)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Génération Figure 4: Courbes d'apprentissage...")

if history_pure and history_hybrid:
    fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 4a: Loss ---
    ax4a = axes[0]
    epochs_pure = [h['epoch'] for h in history_pure]
    train_loss_pure = [h['train_loss'] for h in history_pure]
    val_loss_pure = [h['val_loss'] for h in history_pure]
    
    epochs_hybrid = [h['epoch'] for h in history_hybrid]
    train_loss_hybrid = [h['train_loss'] for h in history_hybrid]
    val_loss_hybrid = [h['val_loss'] for h in history_hybrid]
    
    ax4a.plot(epochs_pure, train_loss_pure, 'g-', linewidth=2, label='Pure - Train', alpha=0.8)
    ax4a.plot(epochs_pure, val_loss_pure, 'g--', linewidth=2, label='Pure - Val', alpha=0.8)
    ax4a.plot(epochs_hybrid, train_loss_hybrid, 'b-', linewidth=2, label='Hybride - Train', alpha=0.8)
    ax4a.plot(epochs_hybrid, val_loss_hybrid, 'b--', linewidth=2, label='Hybride - Val', alpha=0.8)
    
    ax4a.set_xlabel('Epoch', fontsize=12)
    ax4a.set_ylabel('Loss (BCE)', fontsize=12)
    ax4a.set_title('📉 Courbes de Loss: Pure vs Hybride', fontsize=14, fontweight='bold')
    ax4a.legend(loc='upper right', fontsize=10)
    ax4a.grid(True, linestyle='--', alpha=0.7)
    ax4a.set_ylim(0.04, 0.18)
    
    # --- Plot 4b: AUC ---
    ax4b = axes[1]
    val_auc_pure = [h['val_auc_macro'] * 100 for h in history_pure]
    val_auc_hybrid = [h['val_auc_macro'] * 100 for h in history_hybrid]
    
    ax4b.plot(epochs_pure, val_auc_pure, 'g-o', linewidth=2, markersize=5, label='Pure (32 Excel)', alpha=0.8)
    ax4b.plot(epochs_hybrid, val_auc_hybrid, 'b-s', linewidth=2, markersize=5, label='Hybride (57 Excel+NK2)', alpha=0.8)
    
    # Marquer les meilleurs points
    best_pure_idx = np.argmax(val_auc_pure)
    best_hybrid_idx = np.argmax(val_auc_hybrid)
    
    ax4b.scatter([epochs_pure[best_pure_idx]], [val_auc_pure[best_pure_idx]], 
                c='green', s=200, marker='*', zorder=5, edgecolors='black', linewidths=2)
    ax4b.scatter([epochs_hybrid[best_hybrid_idx]], [val_auc_hybrid[best_hybrid_idx]], 
                c='blue', s=200, marker='*', zorder=5, edgecolors='black', linewidths=2)
    
    ax4b.annotate(f'Best: {val_auc_pure[best_pure_idx]:.2f}%', 
                 xy=(epochs_pure[best_pure_idx], val_auc_pure[best_pure_idx]),
                 xytext=(5, 10), textcoords='offset points', fontsize=10, color='green', fontweight='bold')
    ax4b.annotate(f'Best: {val_auc_hybrid[best_hybrid_idx]:.2f}%', 
                 xy=(epochs_hybrid[best_hybrid_idx], val_auc_hybrid[best_hybrid_idx]),
                 xytext=(5, -15), textcoords='offset points', fontsize=10, color='blue', fontweight='bold')
    
    ax4b.set_xlabel('Epoch', fontsize=12)
    ax4b.set_ylabel('Validation AUC Macro (%)', fontsize=12)
    ax4b.set_title('📈 Courbes d\'AUC: Pure vs Hybride', fontsize=14, fontweight='bold')
    ax4b.legend(loc='lower right', fontsize=10)
    ax4b.grid(True, linestyle='--', alpha=0.7)
    ax4b.set_ylim(90, 96)
    
    plt.tight_layout()
    plt.savefig('comparison_learning_curves.png', dpi=150, bbox_inches='tight')
    print("  ✓ Sauvegardé: comparison_learning_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 5: TABLEAU RÉCAPITULATIF VISUEL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/6] Génération Figure 5: Tableau récapitulatif...")

fig5, ax5 = plt.subplots(figsize=(16, 8))
ax5.axis('off')

# Données du tableau
columns = ['Approche', 'Features\nWide', 'Features\nDeep', 'Total\nFeatures', 'AUC\nMacro', 'AUC\nMicro', 'Gain vs\nBaseline']

table_data = []
baseline_auc = 90.34  # XGBoost

for name in order:
    r = results[name]
    gain = r['auc_macro'] - baseline_auc
    gain_str = f"+{gain:.2f}%" if gain > 0 else f"{gain:.2f}%"
    
    row = [
        name.replace('\n', ' '),
        str(r['features_wide']),
        str(r['features_deep']),
        str(r['total_features']),
        f"{r['auc_macro']:.2f}%",
        f"{r['auc_micro']:.2f}%",
        gain_str
    ]
    table_data.append(row)

# Créer tableau
table = ax5.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#3498db'] * len(columns)
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Colorer les cellules
for i, name in enumerate(order):
    # Colorer selon la performance
    if 'Pure' in name:
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor('#d5f5e3')  # Vert clair
    elif 'Hybride' in name:
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor('#d6eaf8')  # Bleu clair
    elif 'XGBoost' in name:
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor('#f5eef8')  # Violet clair
    elif 'Redondant' in name:
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor('#fdebd0')  # Orange clair
    else:
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor('#fadbd8')  # Rouge clair

# Header styling
for j in range(len(columns)):
    table[(0, j)].set_text_props(color='white', fontweight='bold')

ax5.set_title('📋 TABLEAU RÉCAPITULATIF: COMPARAISON DE TOUTES LES APPROCHES', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('comparison_summary_table.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_summary_table.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 6: RADAR CHART (MULTI-CRITÈRES)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7/7] Génération Figure 6: Radar Chart multi-critères...")

fig6, ax6 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Critères normalisés (0-100)
categories = ['AUC Macro', 'AUC Micro', 'Simplicité\n(moins de features)', 
              'Efficacité\n(features/AUC)', 'Utilise Signal']

# Calculer scores normalisés
def normalize_inverse(val, min_val, max_val):
    """Plus petit = mieux (comme features)"""
    return 100 - ((val - min_val) / (max_val - min_val) * 100)

def normalize(val, min_val, max_val):
    """Plus grand = mieux"""
    return (val - min_val) / (max_val - min_val) * 100

# Min/max pour normalisation
all_features = [results[name]['total_features'] for name in order]
all_auc_macro = [results[name]['auc_macro'] for name in order]
all_auc_micro = [results[name]['auc_micro'] for name in order]

scores = {}
for name in order:
    r = results[name]
    scores[name] = [
        normalize(r['auc_macro'], min(all_auc_macro)-1, max(all_auc_macro)+1),  # AUC Macro
        normalize(r['auc_micro'], min(all_auc_micro)-1, max(all_auc_micro)+1),  # AUC Micro
        normalize_inverse(r['total_features'], min(all_features)-10, max(all_features)+10),  # Simplicité
        r['auc_macro'] / r['total_features'] * 100,  # Efficacité (AUC/features)
        100 if r['has_signal'] else 0  # Utilise signal
    ]

# Angles pour le radar
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Fermer le cercle

# Plot pour chaque modèle
for name in ['Wide+Deep\nPure ⭐\n(step6c)', 'Wide+Deep\nHybride\n(step6d)', 'XGBoost\nBaseline\n(step7)']:
    values = scores[name]
    values += values[:1]  # Fermer le cercle
    ax6.plot(angles, values, 'o-', linewidth=2, label=name.replace('\n', ' '), color=results[name]['color'])
    ax6.fill(angles, values, alpha=0.25, color=results[name]['color'])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, size=11)
ax6.set_ylim(0, 100)
ax6.set_title('🎯 Comparaison Multi-Critères (Top 3 Modèles)', fontsize=14, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

plt.tight_layout()
plt.savefig('comparison_radar_chart.png', dpi=150, bbox_inches='tight')
print("  ✓ Sauvegardé: comparison_radar_chart.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. AFFICHER RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 RÉSUMÉ DES PERFORMANCES")
print("="*80)

print("\n┌" + "─"*76 + "┐")
print(f"│ {'Approche':<30} │ {'AUC Macro':>12} │ {'AUC Micro':>12} │ {'Gain':>10} │")
print("├" + "─"*76 + "┤")

for name in order:
    r = results[name]
    gain = r['auc_macro'] - baseline_auc
    gain_str = f"+{gain:.2f}%" if gain > 0 else f"{gain:.2f}%"
    display_name = name.replace('\n', ' ')[:28]
    print(f"│ {display_name:<30} │ {r['auc_macro']:>10.2f}% │ {r['auc_micro']:>10.2f}% │ {gain_str:>10} │")

print("└" + "─"*76 + "┘")

print("\n🏆 MEILLEUR MODÈLE: Wide+Deep Pure (step6c)")
print(f"   • AUC Macro: 94.29% (+3.95% vs XGBoost baseline)")
print(f"   • Features: 32 Wide (Excel) + 64 Deep (Signal) = 96 total")
print(f"   • Avantage: Plus simple ET plus performant!")

print("\n💡 INSIGHTS CLÉS:")
print("   1. Éliminer la redondance (Deep features dans Wide) = +4.46% AUC")
print("   2. NeuroKit2 features sont INUTILES (-0.04% vs Pure)")
print("   3. Le signal brut + Excel = combinaison optimale")
print("   4. Moins de features = Meilleure généralisation")

print("\n" + "="*80)
print("✅ 7 figures générées avec succès!")
print("="*80)

# Afficher les plots
plt.show()
