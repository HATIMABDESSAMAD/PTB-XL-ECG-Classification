"""
═══════════════════════════════════════════════════════════════════════════════
Analyse Exploratoire de Données (EDA) Simplifiée - PTB-XL ECG Database
Version: 1.0.3 - Optimisée
Date: December 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ast
from collections import Counter
from datetime import datetime

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Configuration des figures
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

print("═" * 80)
print("CHARGEMENT DES DONNÉES PTB-XL")
print("═" * 80)

# Chargement des données
df = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
scp_df = pd.read_csv('scp_statements.csv', index_col=0)

# Conversion des codes SCP
df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Conversion des colonnes de qualité en binaire
quality_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
for col in quality_cols:
    df[col] = df[col].notna().astype(int)

print(f"✓ Dataset chargé: {df.shape[0]:,} enregistrements ECG")
print(f"✓ {df['patient_id'].nunique():,} patients uniques")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  1. VUE D'ENSEMBLE DU DATASET")
print("="*80 + "\n")

print(f"📊 Dimensions: {df.shape[0]:,} enregistrements × {df.shape[1]} variables")
print(f"📅 Période: {df['recording_date'].min()} à {df['recording_date'].max()}")
print()

print("Statistiques démographiques:")
print(f"  • Âge moyen: {df['age'].mean():.1f} ans (écart-type: {df['age'].std():.1f})")
print(f"  • Sexe: {(df['sex']==0).sum():,} femmes ({(df['sex']==0).sum()/len(df)*100:.1f}%), "
      f"{(df['sex']==1).sum():,} hommes ({(df['sex']==1).sum()/len(df)*100:.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. VISUALISATIONS DÉMOGRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  2. ANALYSES DÉMOGRAPHIQUES")
print("="*80 + "\n")

fig = plt.figure(figsize=(18, 10))

# Distribution âge
ax1 = plt.subplot(2, 3, 1)
df['age'].hist(bins=50, edgecolor='black', alpha=0.7, color='steelblue', ax=ax1)
ax1.axvline(df['age'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["age"].mean():.1f}')
ax1.set_xlabel('Âge (années)')
ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution de l\'Âge')
ax1.legend()
ax1.grid(alpha=0.3)

# Distribution par sexe
ax2 = plt.subplot(2, 3, 2)
sex_counts = df['sex'].value_counts()
colors = ['#FF69B4', '#4169E1']
labels = ['Femme', 'Homme']
wedges, texts, autotexts = ax2.pie(sex_counts.values, labels=labels, autopct='%1.1f%%',
                                     startangle=90, colors=colors)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
ax2.set_title('Distribution par Sexe')

# Poids
ax3 = plt.subplot(2, 3, 3)
df['weight'].dropna().hist(bins=40, edgecolor='black', alpha=0.7, color='orange', ax=ax3)
ax3.set_xlabel('Poids (kg)')
ax3.set_ylabel('Fréquence')
ax3.set_title(f'Distribution du Poids (n={df["weight"].notna().sum()})')
ax3.grid(alpha=0.3)

# Boxplot âge par sexe
ax4 = plt.subplot(2, 3, 4)
df_plot = df.copy()
df_plot['sex'] = df_plot['sex'].map({0: 'Femme', 1: 'Homme'})
sns.boxplot(data=df_plot, x='sex', y='age', ax=ax4, palette=['#FF69B4', '#4169E1'])
ax4.set_ylabel('Âge (années)')
ax4.set_title('Âge par Sexe')
ax4.grid(axis='y', alpha=0.3)

# Taille
ax5 = plt.subplot(2, 3, 5)
df['height'].dropna().hist(bins=40, edgecolor='black', alpha=0.7, color='green', ax=ax5)
ax5.set_xlabel('Taille (cm)')
ax5.set_ylabel('Fréquence')
ax5.set_title(f'Distribution de la Taille (n={df["height"].notna().sum()})')
ax5.grid(alpha=0.3)

# IMC
ax6 = plt.subplot(2, 3, 6)
df_bmi = df[['height', 'weight']].dropna()
df_bmi['bmi'] = df_bmi['weight'] / ((df_bmi['height'] / 100) ** 2)
df_bmi['bmi'].hist(bins=40, edgecolor='black', alpha=0.7, color='purple', ax=ax6)
ax6.axvline(25, color='orange', linestyle='--', label='Surpoids (25)')
ax6.axvline(30, color='red', linestyle='--', label='Obésité (30)')
ax6.set_xlabel('IMC (kg/m²)')
ax6.set_ylabel('Fréquence')
ax6.set_title(f'IMC (n={len(df_bmi)})')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('EDA_01_Demographics.png', dpi=300, bbox_inches='tight')
print("✓ Sauvegardé: EDA_01_Demographics.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE DES DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  3. ANALYSE DES DIAGNOSTICS")
print("="*80 + "\n")

# Extraction tous les codes SCP
all_scp_codes = []
for codes_dict in df['scp_codes']:
    if isinstance(codes_dict, dict):
        all_scp_codes.extend(list(codes_dict.keys()))

scp_counter = Counter(all_scp_codes)
print(f"📊 {len(scp_counter)} codes SCP uniques")
print(f"🔝 Code le plus fréquent: {scp_counter.most_common(1)[0][0]} ({scp_counter.most_common(1)[0][1]:,} occurrences)")
print()

print("Top 15 des codes SCP:")
for i, (code, count) in enumerate(scp_counter.most_common(15), 1):
    desc = scp_df.loc[code, 'description'] if code in scp_df.index else 'N/A'
    print(f"  {i:2d}. {code:10s} - {desc[:45]:45s} ({count:5,})")
print()

# Visualisation
fig = plt.figure(figsize=(18, 10))

# Top 15 codes SCP
ax1 = plt.subplot(2, 2, 1)
top_15 = scp_counter.most_common(15)
codes, counts = zip(*top_15)
colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(codes)))
bars = ax1.barh(range(len(codes)), counts, color=colors_gradient)
ax1.set_yticks(range(len(codes)))
ax1.set_yticklabels(codes)
ax1.set_xlabel('Nombre d\'occurrences')
ax1.set_title('Top 15 des Codes SCP')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Nombre de codes par enregistrement
ax2 = plt.subplot(2, 2, 2)
codes_per_record = [len(codes) if isinstance(codes, dict) else 0 for codes in df['scp_codes']]
ax2.hist(codes_per_record, bins=range(0, max(codes_per_record)+2), edgecolor='black', alpha=0.7, color='teal')
ax2.set_xlabel('Nombre de codes par enregistrement')
ax2.set_ylabel('Fréquence')
ax2.set_title(f'Codes par Enregistrement (moy: {np.mean(codes_per_record):.2f})')
ax2.axvline(np.mean(codes_per_record), color='red', linestyle='--', linewidth=2)
ax2.grid(alpha=0.3)

# Classes diagnostiques
ax3 = plt.subplot(2, 2, 3)
diag_classes = scp_df[scp_df['diagnostic'] == 1.0]['diagnostic_class'].value_counts()
colors_diag = plt.cm.Set3(np.linspace(0, 1, len(diag_classes)))
wedges, texts, autotexts = ax3.pie(diag_classes.values, labels=diag_classes.index, 
                                     autopct='%1.1f%%', colors=colors_diag, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
ax3.set_title('Classes Diagnostiques (Codes SCP)')

# Catégories de déclarations
ax4 = plt.subplot(2, 2, 4)
statement_cats = scp_df['Statement Category'].value_counts()
bars = ax4.bar(range(len(statement_cats)), statement_cats.values, color='skyblue', edgecolor='black')
ax4.set_xticks(range(len(statement_cats)))
ax4.set_xticklabels(statement_cats.index, rotation=45, ha='right')
ax4.set_ylabel('Nombre de codes')
ax4.set_title('Catégories de Déclarations SCP')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDA_02_Diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Sauvegardé: EDA_02_Diagnostics.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSE TEMPORELLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  4. ANALYSE TEMPORELLE")
print("="*80 + "\n")

df['recording_date'] = pd.to_datetime(df['recording_date'], errors='coerce')
df['year'] = df['recording_date'].dt.year
df['month'] = df['recording_date'].dt.month
df['day_of_week'] = df['recording_date'].dt.dayofweek

print(f"📅 Période: {df['recording_date'].min()} à {df['recording_date'].max()}")
print(f"📊 Durée: {(df['recording_date'].max() - df['recording_date'].min()).days} jours")
print()

fig = plt.figure(figsize=(18, 8))

# Évolution annuelle
ax1 = plt.subplot(1, 3, 1)
yearly = df['year'].value_counts().sort_index()
ax1.plot(yearly.index, yearly.values, marker='o', linewidth=2, markersize=8, color='steelblue')
ax1.fill_between(yearly.index, yearly.values, alpha=0.3, color='steelblue')
ax1.set_xlabel('Année')
ax1.set_ylabel('Nombre d\'enregistrements')
ax1.set_title('Évolution Annuelle')
ax1.grid(alpha=0.3)

# Distribution mensuelle
ax2 = plt.subplot(1, 3, 2)
monthly = df['month'].value_counts().sort_index()
month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
colors_months = plt.cm.Set3(np.linspace(0, 1, 12))
ax2.bar(monthly.index, monthly.values, color=colors_months, edgecolor='black')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=45, ha='right')
ax2.set_ylabel('Nombre d\'enregistrements')
ax2.set_title('Distribution Mensuelle')
ax2.grid(axis='y', alpha=0.3)

# Par jour de la semaine
ax3 = plt.subplot(1, 3, 3)
dow = df['day_of_week'].value_counts().sort_index()
day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
colors_days = ['#4ECDC4' if i < 5 else '#FF6B6B' for i in range(7)]
ax3.bar(range(7), [dow.get(i, 0) for i in range(7)], color=colors_days, edgecolor='black')
ax3.set_xticks(range(7))
ax3.set_xticklabels(day_names)
ax3.set_ylabel('Nombre d\'enregistrements')
ax3.set_title('Distribution par Jour')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDA_03_Temporal.png', dpi=300, bbox_inches='tight')
print("✓ Sauvegardé: EDA_03_Temporal.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. QUALITÉ DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  5. QUALITÉ DES DONNÉES")
print("="*80 + "\n")

# Score de qualité
df['quality_score'] = 6 - df[quality_cols].sum(axis=1)

print(f"🎯 Score de qualité moyen: {df['quality_score'].mean():.2f}/6")
print(f"✅ Enregistrements validés: {df['validated_by_human'].sum():,} ({df['validated_by_human'].sum()/len(df)*100:.1f}%)")
print()

print("Taux de problèmes de qualité:")
for col in quality_cols:
    count = df[col].sum()
    pct = count / len(df) * 100
    print(f"  • {col:25s}: {count:5,} ({pct:5.1f}%)")
print()

fig = plt.figure(figsize=(18, 8))

# Distribution score qualité
ax1 = plt.subplot(1, 3, 1)
quality_counts = df['quality_score'].value_counts().sort_index()
colors_quality = ['#FF0000', '#FF4500', '#FFA500', '#FFD700', '#ADFF2F', '#32CD32', '#228B22']
bars = ax1.bar(quality_counts.index, quality_counts.values,
              color=[colors_quality[int(i)] for i in quality_counts.index], edgecolor='black')
ax1.set_xlabel('Score de Qualité')
ax1.set_ylabel('Nombre d\'enregistrements')
ax1.set_title('Distribution du Score de Qualité')
ax1.grid(axis='y', alpha=0.3)

# Taux de problèmes
ax2 = plt.subplot(1, 3, 2)
quality_issues_pct = pd.Series({col: df[col].sum() / len(df) * 100 for col in quality_cols})
quality_issues_pct = quality_issues_pct.sort_values(ascending=True)
colors_issues = ['#90EE90' if x < 5 else '#FFD700' if x < 15 else '#FF6B6B' for x in quality_issues_pct.values]
bars = ax2.barh(range(len(quality_issues_pct)), quality_issues_pct.values, color=colors_issues, edgecolor='black')
ax2.set_yticks(range(len(quality_issues_pct)))
ax2.set_yticklabels([col.replace('_', ' ').title() for col in quality_issues_pct.index])
ax2.set_xlabel('Pourcentage (%)')
ax2.set_title('Taux de Problèmes')
ax2.grid(axis='x', alpha=0.3)

# Valeurs manquantes
ax3 = plt.subplot(1, 3, 3)
missing = df.isnull().sum().sort_values(ascending=False).head(10)
missing_pct = (missing / len(df) * 100)
bars = ax3.barh(range(len(missing)), missing_pct.values, color='coral', edgecolor='black')
ax3.set_yticks(range(len(missing)))
ax3.set_yticklabels(missing.index)
ax3.set_xlabel('Pourcentage (%)')
ax3.set_title('Top 10 Valeurs Manquantes')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('EDA_04_Quality.png', dpi=300, bbox_inches='tight')
print("✓ Sauvegardé: EDA_04_Quality.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. INFRASTRUCTURE & TECHNIQUE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  6. ASPECTS TECHNIQUES")
print("="*80 + "\n")

print(f"🔧 Infrastructure:")
print(f"  • Sites: {df['site'].nunique()}")
print(f"  • Appareils: {df['device'].nunique()}")
print(f"  • Infirmières: {df['nurse'].nunique()}")
print()

fig = plt.figure(figsize=(18, 8))

# Distribution des sites
ax1 = plt.subplot(1, 3, 1)
site_counts = df['site'].value_counts().head(10)
ax1.bar(range(len(site_counts)), site_counts.values, color='coral', edgecolor='black')
ax1.set_xticks(range(len(site_counts)))
ax1.set_xticklabels([f'Site {int(s)}' for s in site_counts.index], rotation=45, ha='right')
ax1.set_ylabel('Nombre d\'enregistrements')
ax1.set_title('Top 10 Sites')
ax1.grid(axis='y', alpha=0.3)

# Distribution des appareils
ax2 = plt.subplot(1, 3, 2)
device_counts = df['device'].value_counts().head(8)
bars = ax2.barh(range(len(device_counts)), device_counts.values, color='skyblue', edgecolor='black')
ax2.set_yticks(range(len(device_counts)))
ax2.set_yticklabels(device_counts.index)
ax2.set_xlabel('Nombre d\'enregistrements')
ax2.set_title('Top 8 Appareils')
ax2.grid(axis='x', alpha=0.3)

# Distribution des folds
ax3 = plt.subplot(1, 3, 3)
fold_counts = df['strat_fold'].value_counts().sort_index()
colors_fold = plt.cm.viridis(np.linspace(0, 1, len(fold_counts)))
ax3.bar(fold_counts.index, fold_counts.values, color=colors_fold, edgecolor='black')
ax3.set_xlabel('Fold')
ax3.set_ylabel('Nombre d\'enregistrements')
ax3.set_title('Stratification (Folds)')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDA_05_Technical.png', dpi=300, bbox_inches='tight')
print("✓ Sauvegardé: EDA_05_Technical.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RAPPORT RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  7. GÉNÉRATION DU RAPPORT RÉSUMÉ")
print("="*80 + "\n")

report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      RAPPORT D'ANALYSE EXPLORATOIRE                           ║
║                     Dataset PTB-XL ECG Database v1.0.3                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 STATISTIQUES GÉNÉRALES
─────────────────────────────────────────────────────────────────────────────────
• Nombre d'enregistrements:                  {df.shape[0]:>10,}
• Nombre de patients uniques:                {df['patient_id'].nunique():>10,}
• Nombre de variables:                       {df.shape[1]:>10}
• Période:                                   {df['recording_date'].min().strftime('%Y-%m-%d')} à {df['recording_date'].max().strftime('%Y-%m-%d')}

👥 DÉMOGRAPHIE
─────────────────────────────────────────────────────────────────────────────────
• Âge moyen:                                 {df['age'].mean():>10.1f} ans
• Âge médian:                                {df['age'].median():>10.1f} ans
• Femmes:                                    {(df['sex']==0).sum():>10,} ({(df['sex']==0).sum()/len(df)*100:.1f}%)
• Hommes:                                    {(df['sex']==1).sum():>10,} ({(df['sex']==1).sum()/len(df)*100:.1f}%)

🏥 DIAGNOSTICS
─────────────────────────────────────────────────────────────────────────────────
• Codes SCP uniques:                         {len(scp_counter):>10}
• Code le plus fréquent:                     {scp_counter.most_common(1)[0][0]:>10} ({scp_counter.most_common(1)[0][1]:,})
• Moyenne codes/enregistrement:              {np.mean(codes_per_record):>10.2f}

🎯 QUALITÉ
─────────────────────────────────────────────────────────────────────────────────
• Score moyen:                               {df['quality_score'].mean():>10.2f}/6
• Validés par humain:                        {df['validated_by_human'].sum():>10,} ({df['validated_by_human'].sum()/len(df)*100:.1f}%)
• Baseline Drift:                            {df['baseline_drift'].sum():>10,} ({df['baseline_drift'].sum()/len(df)*100:.1f}%)
• Static Noise:                              {df['static_noise'].sum():>10,} ({df['static_noise'].sum()/len(df)*100:.1f}%)
• Burst Noise:                               {df['burst_noise'].sum():>10,} ({df['burst_noise'].sum()/len(df)*100:.1f}%)

🔧 INFRASTRUCTURE
─────────────────────────────────────────────────────────────────────────────────
• Nombre de sites:                           {df['site'].nunique():>10}
• Nombre d'appareils:                        {df['device'].nunique():>10}
• Nombre d'infirmières:                      {df['nurse'].nunique():>10}

═══════════════════════════════════════════════════════════════════════════════
RECOMMANDATIONS
═══════════════════════════════════════════════════════════════════════════════

✓ Points Forts:
  • Large dataset ({df.shape[0]:,} enregistrements)
  • Bonne qualité générale (score: {df['quality_score'].mean():.2f}/6)
  • Validation humaine étendue ({df['validated_by_human'].sum()/len(df)*100:.1f}%)
  • Stratification intégrée pour ML

⚠️  Points d'Attention:
  • Valeurs manquantes pour height ({df['height'].isna().sum()/len(df)*100:.1f}%) et weight ({df['weight'].isna().sum()/len(df)*100:.1f}%)
  • Quelques problèmes de qualité du signal
  • Distribution temporelle non uniforme

💡 Applications Suggérées:
  1. Classification automatique des ECG par deep learning
  2. Détection d'anomalies cardiovasculaires
  3. Analyse de séries temporelles médicales
  4. Recherche clinique sur pathologies cardiaques

═══════════════════════════════════════════════════════════════════════════════
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════════════
"""

print(report)

with open('PTB_XL_EDA_Report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ Rapport sauvegardé: PTB_XL_EDA_Report.txt")

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*80)
print("█" + "  ✓ ANALYSE TERMINÉE AVEC SUCCÈS".center(78) + "█")
print("█" + " "*78 + "█")
print("█" + "  Fichiers générés:".ljust(78) + "█")
print("█" + "    • EDA_01_Demographics.png".ljust(78) + "█")
print("█" + "    • EDA_02_Diagnostics.png".ljust(78) + "█")
print("█" + "    • EDA_03_Temporal.png".ljust(78) + "█")
print("█" + "    • EDA_04_Quality.png".ljust(78) + "█")
print("█" + "    • EDA_05_Technical.png".ljust(78) + "█")
print("█" + "    • PTB_XL_EDA_Report.txt".ljust(78) + "█")
print("█" + " "*78 + "█")
print("█"*80)
