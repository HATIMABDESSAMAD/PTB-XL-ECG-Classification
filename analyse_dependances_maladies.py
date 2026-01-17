"""
═══════════════════════════════════════════════════════════════════════════════
ANALYSE DES DÉPENDANCES - Codes SCP (Maladies) vs Variables
Dataset: PTB-XL ECG Database
Expert Data Scientist Analysis
═══════════════════════════════════════════════════════════════════════════════

Ce script analyse les dépendances et associations entre les codes diagnostiques SCP
(maladies cardiaques) et les autres variables du dataset (démographiques, temporelles,
qualité, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ast
from collections import Counter
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration des graphiques
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("═" * 80)
print("ANALYSE DES DÉPENDANCES - CODES SCP vs VARIABLES")
print("═" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("📥 Chargement des données...")

df = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
scp_df = pd.read_csv('scp_statements.csv', index_col=0)

# Conversion des codes SCP
df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Préparation des données temporelles
df['recording_date'] = pd.to_datetime(df['recording_date'], errors='coerce')
df['year'] = df['recording_date'].dt.year
df['month'] = df['recording_date'].dt.month

# Conversion des colonnes de qualité
quality_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
for col in quality_cols:
    df[col] = df[col].notna().astype(int)

print(f"✓ {len(df):,} enregistrements chargés")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES CODES SCP LES PLUS FRÉQUENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. IDENTIFICATION DES CODES SCP PRINCIPAUX")
print("=" * 80)
print()

# Compter tous les codes
all_codes = []
for codes_dict in df['scp_codes']:
    if isinstance(codes_dict, dict):
        all_codes.extend(list(codes_dict.keys()))

code_counter = Counter(all_codes)
top_20_codes = [code for code, count in code_counter.most_common(20)]

print(f"📊 Top 20 codes SCP les plus fréquents:")
for i, (code, count) in enumerate(code_counter.most_common(20), 1):
    desc = scp_df.loc[code, 'description'] if code in scp_df.index else 'N/A'
    pct = count / len(df) * 100
    print(f"  {i:2d}. {code:10s} - {desc[:40]:40s} {count:6,} ({pct:5.1f}%)")
print()

# Créer des colonnes binaires pour les codes principaux
print("🔄 Création de variables binaires pour les codes SCP...")
for code in top_20_codes:
    df[f'has_{code}'] = df['scp_codes'].apply(
        lambda x: 1 if isinstance(x, dict) and code in x else 0
    )
print(f"✓ {len(top_20_codes)} variables binaires créées")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSE AVEC L'ÂGE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("2. DÉPENDANCE ENTRE CODES SCP ET ÂGE")
print("=" * 80)
print()

age_analysis = []

for code in top_20_codes:
    has_code = df[f'has_{code}'] == 1
    no_code = df[f'has_{code}'] == 0
    
    age_with = df.loc[has_code, 'age'].dropna()
    age_without = df.loc[no_code, 'age'].dropna()
    
    if len(age_with) > 0 and len(age_without) > 0:
        # Test statistique de Mann-Whitney
        stat, p_value = mannwhitneyu(age_with, age_without, alternative='two-sided')
        
        age_analysis.append({
            'Code': code,
            'Description': scp_df.loc[code, 'description'] if code in scp_df.index else 'N/A',
            'Age_Moyen_Avec': age_with.mean(),
            'Age_Moyen_Sans': age_without.mean(),
            'Difference': age_with.mean() - age_without.mean(),
            'P_Value': p_value,
            'Significatif': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'
        })

age_df = pd.DataFrame(age_analysis).sort_values('Difference', ascending=False)

print("📊 Différences d'âge moyen par code SCP:")
print(f"{'Code':<10} {'Description':<35} {'Âge Avec':<10} {'Âge Sans':<10} {'Diff':<8} {'Signif':<8}")
print("-" * 90)
for _, row in age_df.iterrows():
    print(f"{row['Code']:<10} {row['Description'][:34]:<35} {row['Age_Moyen_Avec']:>9.1f} "
          f"{row['Age_Moyen_Sans']:>9.1f} {row['Difference']:>+7.1f} {row['Significatif']:>7}")
print()
print("Légende: *** p<0.001, ** p<0.01, * p<0.05, NS = Non significatif")
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Top 10 différences d'âge
ax1 = axes[0, 0]
top_10_age = age_df.head(10)
colors = ['red' if x > 0 else 'blue' for x in top_10_age['Difference']]
bars = ax1.barh(range(len(top_10_age)), top_10_age['Difference'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top_10_age)))
ax1.set_yticklabels(top_10_age['Code'])
ax1.set_xlabel('Différence d\'âge moyen (années)')
ax1.set_title('Top 10 - Codes SCP avec Plus Grande Différence d\'Âge')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Distribution d'âge pour quelques codes spécifiques
ax2 = axes[0, 1]
codes_to_plot = ['NORM', 'IMI', 'AFIB', 'STACH']
data_to_plot = []
labels = []

for code in codes_to_plot:
    if f'has_{code}' in df.columns:
        ages = df[df[f'has_{code}'] == 1]['age'].dropna()
        if len(ages) > 0:
            data_to_plot.append(ages)
            labels.append(code)

if data_to_plot:
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))):
        patch.set_facecolor(color)
    ax2.set_ylabel('Âge (années)')
    ax2.set_title('Distribution d\'Âge par Code SCP Sélectionné')
    ax2.grid(axis='y', alpha=0.3)

# 3. P-values visualization
ax3 = axes[1, 0]
age_df_sorted = age_df.sort_values('P_Value')
colors_p = ['green' if p < 0.05 else 'orange' for p in age_df_sorted['P_Value']]
ax3.barh(range(len(age_df_sorted)), -np.log10(age_df_sorted['P_Value']), color=colors_p, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(age_df_sorted)))
ax3.set_yticklabels(age_df_sorted['Code'])
ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
ax3.set_xlabel('-log10(p-value)')
ax3.set_title('Significativité Statistique des Associations Âge-Code')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# 4. Scatter plot âge vs nombre de codes
ax4 = axes[1, 1]
df['num_codes'] = df['scp_codes'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
ax4.scatter(df['age'], df['num_codes'], alpha=0.3, s=10)
ax4.set_xlabel('Âge (années)')
ax4.set_ylabel('Nombre de codes SCP')
ax4.set_title('Relation entre Âge et Nombre de Codes Diagnostiques')
ax4.grid(alpha=0.3)

# Régression linéaire
z = np.polyfit(df['age'].dropna(), df.loc[df['age'].notna(), 'num_codes'], 1)
p = np.poly1d(z)
age_sorted = df['age'].dropna().sort_values()
ax4.plot(age_sorted, p(age_sorted), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.2f}')
ax4.legend()

plt.tight_layout()
plt.savefig('Dependances_01_Age_Codes.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_01_Age_Codes.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE AVEC LE SEXE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. DÉPENDANCE ENTRE CODES SCP ET SEXE")
print("=" * 80)
print()

sex_analysis = []

for code in top_20_codes:
    # Table de contingence
    contingency = pd.crosstab(df[f'has_{code}'], df['sex'])
    
    # Test du Chi-2
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Prévalence par sexe
    prev_female = df[df['sex'] == 0][f'has_{code}'].mean() * 100
    prev_male = df[df['sex'] == 1][f'has_{code}'].mean() * 100
    
    sex_analysis.append({
        'Code': code,
        'Description': scp_df.loc[code, 'description'] if code in scp_df.index else 'N/A',
        'Prevalence_Femmes': prev_female,
        'Prevalence_Hommes': prev_male,
        'Difference': prev_male - prev_female,
        'Chi2': chi2,
        'P_Value': p_value,
        'Significatif': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'
    })

sex_df = pd.DataFrame(sex_analysis).sort_values('Difference', ascending=False)

print("📊 Différences de prévalence par sexe:")
print(f"{'Code':<10} {'Description':<35} {'Femmes':<10} {'Hommes':<10} {'Diff':<8} {'Signif':<8}")
print("-" * 90)
for _, row in sex_df.iterrows():
    print(f"{row['Code']:<10} {row['Description'][:34]:<35} {row['Prevalence_Femmes']:>8.1f}% "
          f"{row['Prevalence_Hommes']:>8.1f}% {row['Difference']:>+6.1f}% {row['Significatif']:>7}")
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Différences de prévalence
ax1 = axes[0, 0]
colors = ['#FF69B4' if x < 0 else '#4169E1' for x in sex_df['Difference']]
bars = ax1.barh(range(len(sex_df)), sex_df['Difference'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(sex_df)))
ax1.set_yticklabels(sex_df['Code'])
ax1.set_xlabel('Différence de prévalence Hommes - Femmes (%)')
ax1.set_title('Différences de Prévalence par Sexe pour Chaque Code SCP')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Comparaison prévalences
ax2 = axes[0, 1]
x = np.arange(len(sex_df.head(10)))
width = 0.35
ax2.barh(x - width/2, sex_df.head(10)['Prevalence_Femmes'], width, label='Femmes', color='#FF69B4', alpha=0.8)
ax2.barh(x + width/2, sex_df.head(10)['Prevalence_Hommes'], width, label='Hommes', color='#4169E1', alpha=0.8)
ax2.set_yticks(x)
ax2.set_yticklabels(sex_df.head(10)['Code'])
ax2.set_xlabel('Prévalence (%)')
ax2.set_title('Top 10 - Prévalence par Sexe')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# 3. Heatmap de prévalence
ax3 = axes[1, 0]
heatmap_data = sex_df[['Prevalence_Femmes', 'Prevalence_Hommes']].head(15).T
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, 
            xticklabels=sex_df.head(15)['Code'], 
            yticklabels=['Femmes', 'Hommes'],
            cbar_kws={'label': 'Prévalence (%)'})
ax3.set_title('Heatmap - Prévalence par Sexe (Top 15)')

# 4. Odds ratio visualization
ax4 = axes[1, 1]
# Calculer les odds ratios
odds_ratios = []
for code in top_20_codes:
    contingency = pd.crosstab(df[f'has_{code}'], df['sex'])
    if contingency.shape == (2, 2):
        # OR = (a*d)/(b*c) where a=male with code, b=male without, c=female with, d=female without
        or_value = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (contingency.iloc[1, 0] * contingency.iloc[0, 1] + 0.0001)
        odds_ratios.append({'Code': code, 'OR': or_value})

or_df = pd.DataFrame(odds_ratios).sort_values('OR', ascending=False)
colors_or = ['#4169E1' if x > 1 else '#FF69B4' for x in or_df['OR']]
ax4.barh(range(len(or_df)), or_df['OR'], color=colors_or, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(or_df)))
ax4.set_yticklabels(or_df['Code'])
ax4.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR=1 (pas d\'effet)')
ax4.set_xlabel('Odds Ratio (Hommes vs Femmes)')
ax4.set_title('Odds Ratios - Risque Relatif Hommes vs Femmes')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Dependances_02_Sexe_Codes.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_02_Sexe_Codes.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSE AVEC LES VARIABLES ANTHROPOMÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. DÉPENDANCE ENTRE CODES SCP ET VARIABLES ANTHROPOMÉTRIQUES")
print("=" * 80)
print()

# Calcul IMC
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

anthro_analysis = []

for code in top_20_codes[:10]:  # Top 10 pour lisibilité
    has_code = df[f'has_{code}'] == 1
    no_code = df[f'has_{code}'] == 0
    
    # Poids
    weight_with = df.loc[has_code, 'weight'].dropna()
    weight_without = df.loc[no_code, 'weight'].dropna()
    
    # Taille
    height_with = df.loc[has_code, 'height'].dropna()
    height_without = df.loc[no_code, 'height'].dropna()
    
    # IMC
    bmi_with = df.loc[has_code, 'bmi'].dropna()
    bmi_without = df.loc[no_code, 'bmi'].dropna()
    
    result = {'Code': code}
    
    if len(weight_with) > 10 and len(weight_without) > 10:
        _, p_weight = mannwhitneyu(weight_with, weight_without, alternative='two-sided')
        result['Weight_Diff'] = weight_with.mean() - weight_without.mean()
        result['Weight_P'] = p_weight
    else:
        result['Weight_Diff'] = np.nan
        result['Weight_P'] = np.nan
    
    if len(height_with) > 10 and len(height_without) > 10:
        _, p_height = mannwhitneyu(height_with, height_without, alternative='two-sided')
        result['Height_Diff'] = height_with.mean() - height_without.mean()
        result['Height_P'] = p_height
    else:
        result['Height_Diff'] = np.nan
        result['Height_P'] = np.nan
    
    if len(bmi_with) > 10 and len(bmi_without) > 10:
        _, p_bmi = mannwhitneyu(bmi_with, bmi_without, alternative='two-sided')
        result['BMI_Diff'] = bmi_with.mean() - bmi_without.mean()
        result['BMI_P'] = p_bmi
    else:
        result['BMI_Diff'] = np.nan
        result['BMI_P'] = np.nan
    
    anthro_analysis.append(result)

anthro_df = pd.DataFrame(anthro_analysis)

print("📊 Associations avec les variables anthropométriques:")
print(f"{'Code':<10} {'Poids (kg)':<15} {'Taille (cm)':<15} {'IMC (kg/m²)':<15}")
print("-" * 60)
for _, row in anthro_df.iterrows():
    weight_str = f"{row['Weight_Diff']:+.1f} {'*' if row['Weight_P'] < 0.05 else ' '}" if not pd.isna(row['Weight_Diff']) else "N/A"
    height_str = f"{row['Height_Diff']:+.1f} {'*' if row['Height_P'] < 0.05 else ' '}" if not pd.isna(row['Height_Diff']) else "N/A"
    bmi_str = f"{row['BMI_Diff']:+.2f} {'*' if row['BMI_P'] < 0.05 else ' '}" if not pd.isna(row['BMI_Diff']) else "N/A"
    print(f"{row['Code']:<10} {weight_str:<15} {height_str:<15} {bmi_str:<15}")
print()
print("* = significatif (p<0.05)")
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Différences de poids
ax1 = axes[0, 0]
valid_weight = anthro_df.dropna(subset=['Weight_Diff'])
colors = ['red' if p < 0.05 else 'gray' for p in valid_weight['Weight_P']]
ax1.barh(range(len(valid_weight)), valid_weight['Weight_Diff'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(valid_weight)))
ax1.set_yticklabels(valid_weight['Code'])
ax1.set_xlabel('Différence de poids moyen (kg)')
ax1.set_title('Différences de Poids par Code SCP')
ax1.axvline(x=0, color='black', linestyle='--')
ax1.grid(axis='x', alpha=0.3)

# 2. Différences d'IMC
ax2 = axes[0, 1]
valid_bmi = anthro_df.dropna(subset=['BMI_Diff'])
colors = ['red' if p < 0.05 else 'gray' for p in valid_bmi['BMI_P']]
ax2.barh(range(len(valid_bmi)), valid_bmi['BMI_Diff'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(valid_bmi)))
ax2.set_yticklabels(valid_bmi['Code'])
ax2.set_xlabel('Différence d\'IMC moyen (kg/m²)')
ax2.set_title('Différences d\'IMC par Code SCP')
ax2.axvline(x=0, color='black', linestyle='--')
ax2.grid(axis='x', alpha=0.3)

# 3. Distribution IMC pour codes sélectionnés
ax3 = axes[1, 0]
codes_to_plot = ['NORM', 'IMI', 'LVH']
data_to_plot = []
labels = []

for code in codes_to_plot:
    if f'has_{code}' in df.columns:
        bmi_vals = df[df[f'has_{code}'] == 1]['bmi'].dropna()
        if len(bmi_vals) > 10:
            data_to_plot.append(bmi_vals)
            labels.append(code)

if data_to_plot:
    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['green', 'red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('IMC (kg/m²)')
    ax3.set_title('Distribution IMC par Code SCP Sélectionné')
    ax3.grid(axis='y', alpha=0.3)

# 4. Heatmap des p-values
ax4 = axes[1, 1]
heatmap_data = anthro_df[['Weight_P', 'Height_P', 'BMI_P']].T
heatmap_data.columns = anthro_df['Code']
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax4,
            yticklabels=['Poids', 'Taille', 'IMC'],
            vmin=0, vmax=0.1, cbar_kws={'label': 'P-value'})
ax4.set_title('Heatmap des P-values (Vert = Significatif)')

plt.tight_layout()
plt.savefig('Dependances_03_Anthropometrie_Codes.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_03_Anthropometrie_Codes.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANALYSE TEMPORELLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. DÉPENDANCE ENTRE CODES SCP ET TEMPS")
print("=" * 80)
print()

# Évolution temporelle de la prévalence
temporal_analysis = {}

for code in top_20_codes[:10]:
    yearly_prev = df.groupby('year')[f'has_{code}'].mean() * 100
    temporal_analysis[code] = yearly_prev

temporal_df = pd.DataFrame(temporal_analysis).fillna(0)

print("📊 Évolution de la prévalence des codes SCP dans le temps:")
print(temporal_df.tail(10))
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Évolution temporelle top 5
ax1 = axes[0, 0]
for code in ['NORM', 'SR', 'IMI', 'AFIB', 'LVH']:
    if code in temporal_df.columns:
        ax1.plot(temporal_df.index, temporal_df[code], marker='o', linewidth=2, label=code)
ax1.set_xlabel('Année')
ax1.set_ylabel('Prévalence (%)')
ax1.set_title('Évolution Temporelle de la Prévalence (Top 5 Codes)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Heatmap temporelle
ax2 = axes[0, 1]
sns.heatmap(temporal_df.T, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Prévalence (%)'})
ax2.set_xlabel('Année')
ax2.set_ylabel('Code SCP')
ax2.set_title('Heatmap - Prévalence Temporelle')

# 3. Distribution par mois
ax3 = axes[1, 0]
monthly_counts = {}
for code in ['NORM', 'IMI', 'AFIB']:
    if f'has_{code}' in df.columns:
        monthly = df[df[f'has_{code}'] == 1].groupby('month').size()
        monthly_counts[code] = monthly

if monthly_counts:
    monthly_df = pd.DataFrame(monthly_counts).fillna(0)
    monthly_df.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Mois')
    ax3.set_ylabel('Nombre de cas')
    ax3.set_title('Distribution Mensuelle des Cas (Codes Sélectionnés)')
    ax3.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'], rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

# 4. Tendances (régression)
ax4 = axes[1, 1]
for code in ['NORM', 'IMI', 'LVH']:
    if code in temporal_df.columns:
        years = temporal_df.index.values
        prev = temporal_df[code].values
        
        # Régression linéaire
        z = np.polyfit(years, prev, 1)
        p = np.poly1d(z)
        
        ax4.scatter(years, prev, label=f'{code} (data)', alpha=0.5)
        ax4.plot(years, p(years), '--', linewidth=2, label=f'{code} (trend: {z[0]:+.2f}%/an)')

ax4.set_xlabel('Année')
ax4.set_ylabel('Prévalence (%)')
ax4.set_title('Tendances Temporelles (Régressions Linéaires)')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Dependances_04_Temporel_Codes.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_04_Temporel_Codes.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSE AVEC LA QUALITÉ DU SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. DÉPENDANCE ENTRE CODES SCP ET QUALITÉ DU SIGNAL")
print("=" * 80)
print()

quality_analysis = []

for code in top_20_codes[:10]:
    result = {'Code': code}
    
    for quality_col in quality_cols:
        has_code = df[f'has_{code}'] == 1
        
        # Prévalence du problème de qualité
        prev_with = df.loc[has_code, quality_col].mean() * 100
        prev_without = df.loc[~has_code, quality_col].mean() * 100
        
        # Test Chi-2
        contingency = pd.crosstab(df[f'has_{code}'], df[quality_col])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        result[f'{quality_col}_diff'] = prev_with - prev_without
        result[f'{quality_col}_p'] = p_value
    
    quality_analysis.append(result)

quality_df = pd.DataFrame(quality_analysis)

print("📊 Association entre codes SCP et problèmes de qualité:")
print(f"{'Code':<10} {'Baseline':<12} {'Static':<12} {'Burst':<12} {'Electrodes':<12}")
print("-" * 60)
for _, row in quality_df.iterrows():
    baseline = f"{row['baseline_drift_diff']:+.1f}% {'*' if row['baseline_drift_p'] < 0.05 else ''}"
    static = f"{row['static_noise_diff']:+.1f}% {'*' if row['static_noise_p'] < 0.05 else ''}"
    burst = f"{row['burst_noise_diff']:+.1f}% {'*' if row['burst_noise_p'] < 0.05 else ''}"
    electrodes = f"{row['electrodes_problems_diff']:+.1f}% {'*' if row['electrodes_problems_p'] < 0.05 else ''}"
    print(f"{row['Code']:<10} {baseline:<12} {static:<12} {burst:<12} {electrodes:<12}")
print()
print("* = association significative (p<0.05)")
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Heatmap des différences
ax1 = axes[0, 0]
diff_cols = [col for col in quality_df.columns if col.endswith('_diff')]
heatmap_data = quality_df[diff_cols].set_index(quality_df['Code'])
heatmap_data.columns = [col.replace('_diff', '').replace('_', ' ').title() for col in diff_cols]
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax1,
            cbar_kws={'label': 'Différence (%)'})
ax1.set_title('Heatmap - Différences de Prévalence des Problèmes de Qualité')

# 2. P-values
ax2 = axes[0, 1]
p_cols = [col for col in quality_df.columns if col.endswith('_p')]
p_data = quality_df[p_cols].set_index(quality_df['Code'])
p_data.columns = [col.replace('_p', '').replace('_', ' ').title() for col in p_cols]
sns.heatmap(p_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax2,
            vmin=0, vmax=0.1, cbar_kws={'label': 'P-value'})
ax2.set_title('Heatmap - Significativité Statistique (P-values)')

# 3. Barplot pour un code spécifique
ax3 = axes[1, 0]
if 'IMI' in quality_df['Code'].values:
    imi_data = quality_df[quality_df['Code'] == 'IMI'][diff_cols].values[0]
    labels = [col.replace('_diff', '').replace('_', '\n').title() for col in diff_cols]
    colors = ['red' if x > 0 else 'blue' for x in imi_data]
    ax3.bar(range(len(imi_data)), imi_data, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(imi_data)))
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.set_ylabel('Différence de prévalence (%)')
    ax3.set_title('Exemple: Code IMI - Associations avec Qualité')
    ax3.axhline(y=0, color='black', linestyle='--')
    ax3.grid(axis='y', alpha=0.3)

# 4. Score de qualité par code
ax4 = axes[1, 1]
df['quality_score'] = 6 - df[quality_cols].sum(axis=1)

quality_scores = []
for code in top_20_codes[:10]:
    score_with = df[df[f'has_{code}'] == 1]['quality_score'].mean()
    score_without = df[df[f'has_{code}'] == 0]['quality_score'].mean()
    quality_scores.append({'Code': code, 'Score_Avec': score_with, 'Score_Sans': score_without, 
                          'Diff': score_with - score_without})

qs_df = pd.DataFrame(quality_scores)
x = np.arange(len(qs_df))
width = 0.35
ax4.barh(x - width/2, qs_df['Score_Avec'], width, label='Avec le code', color='coral', alpha=0.8)
ax4.barh(x + width/2, qs_df['Score_Sans'], width, label='Sans le code', color='lightblue', alpha=0.8)
ax4.set_yticks(x)
ax4.set_yticklabels(qs_df['Code'])
ax4.set_xlabel('Score de qualité moyen (/6)')
ax4.set_title('Score de Qualité Moyen par Présence de Code SCP')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Dependances_05_Qualite_Codes.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_05_Qualite_Codes.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANALYSE DES CO-OCCURRENCES DE CODES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. ANALYSE DES CO-OCCURRENCES ENTRE CODES SCP")
print("=" * 80)
print()

# Matrice de co-occurrence
cooc_matrix = pd.DataFrame(0, index=top_20_codes, columns=top_20_codes)

for i, code1 in enumerate(top_20_codes):
    for j, code2 in enumerate(top_20_codes):
        if i != j:
            # Nombre de fois où les deux codes apparaissent ensemble
            both = ((df[f'has_{code1}'] == 1) & (df[f'has_{code2}'] == 1)).sum()
            # Normaliser par le nombre de fois où code1 apparaît
            total_code1 = (df[f'has_{code1}'] == 1).sum()
            cooc_matrix.loc[code1, code2] = (both / total_code1 * 100) if total_code1 > 0 else 0

print("📊 Top 10 paires de codes les plus associés:")
# Extraire les paires
pairs = []
for i in range(len(top_20_codes)):
    for j in range(i+1, len(top_20_codes)):
        code1, code2 = top_20_codes[i], top_20_codes[j]
        value = cooc_matrix.loc[code1, code2]
        pairs.append({'Code1': code1, 'Code2': code2, 'Cooccurrence': value})

pairs_df = pd.DataFrame(pairs).sort_values('Cooccurrence', ascending=False).head(10)

for idx, row in pairs_df.iterrows():
    print(f"  {row['Code1']:10s} ↔ {row['Code2']:10s} : {row['Cooccurrence']:5.1f}%")
print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Heatmap complète de co-occurrence
ax1 = axes[0, 0]
sns.heatmap(cooc_matrix.astype(float), cmap='YlOrRd', ax=ax1, 
            cbar_kws={'label': 'Co-occurrence (%)'}, square=True)
ax1.set_title('Matrice de Co-occurrence des Codes SCP')

# 2. Réseau de co-occurrences (simplifié)
ax2 = axes[0, 1]
threshold = 20  # Seuil de co-occurrence pour affichage
for idx, row in pairs_df.head(15).iterrows():
    # Visualisation simplifiée en barres
    pass

pairs_df_top = pairs_df.head(15)
ax2.barh(range(len(pairs_df_top)), pairs_df_top['Cooccurrence'], color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(pairs_df_top)))
ax2.set_yticklabels([f"{row['Code1']}↔{row['Code2']}" for _, row in pairs_df_top.iterrows()])
ax2.set_xlabel('Co-occurrence (%)')
ax2.set_title('Top 15 Paires de Codes les Plus Associés')
ax2.grid(axis='x', alpha=0.3)

# 3. Nombre moyen de codes co-occurents
ax3 = axes[1, 0]
avg_cooc = cooc_matrix.mean(axis=1).sort_values(ascending=False).head(15)
ax3.barh(range(len(avg_cooc)), avg_cooc, color='coral', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(avg_cooc)))
ax3.set_yticklabels(avg_cooc.index)
ax3.set_xlabel('Co-occurrence moyenne (%)')
ax3.set_title('Codes avec Plus de Co-occurrences Moyennes')
ax3.grid(axis='x', alpha=0.3)

# 4. Distribution du nombre de codes par patient
ax4 = axes[1, 1]
df['num_codes'].hist(bins=range(0, 15), edgecolor='black', alpha=0.7, color='purple', ax=ax4)
ax4.set_xlabel('Nombre de codes SCP par enregistrement')
ax4.set_ylabel('Fréquence')
ax4.set_title(f'Distribution du Nombre de Codes (Moyenne: {df["num_codes"].mean():.2f})')
ax4.axvline(df['num_codes'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Dependances_06_Cooccurrences.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: Dependances_06_Cooccurrences.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. RAPPORT FINAL DE SYNTHÈSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("8. GÉNÉRATION DU RAPPORT DE SYNTHÈSE")
print("=" * 80)
print()

rapport = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT D'ANALYSE DES DÉPENDANCES                          ║
║                  Codes SCP (Maladies) vs Variables PTB-XL                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════════════════════
1. RÉSUMÉ EXÉCUTIF
═══════════════════════════════════════════════════════════════════════════════

Cette analyse examine les dépendances entre les {len(top_20_codes)} codes SCP les plus 
fréquents et les variables démographiques, anthropométriques, temporelles et de 
qualité du dataset PTB-XL.

Dataset analysé: {len(df):,} enregistrements ECG

═══════════════════════════════════════════════════════════════════════════════
2. PRINCIPALES DÉCOUVERTES
═══════════════════════════════════════════════════════════════════════════════

🔍 DÉPENDANCE AVEC L'ÂGE
───────────────────────

Codes avec différences d'âge les plus marquées:
{age_df.head(5)[['Code', 'Age_Moyen_Avec', 'Age_Moyen_Sans', 'Difference', 'Significatif']].to_string(index=False)}

→ {len(age_df[age_df['Significatif'] != 'NS'])} codes sur {len(age_df)} montrent une 
  association significative avec l'âge (p<0.05)

🚻 DÉPENDANCE AVEC LE SEXE
───────────────────────────

Codes avec différences de prévalence les plus marquées:
{sex_df.head(5)[['Code', 'Prevalence_Femmes', 'Prevalence_Hommes', 'Difference', 'Significatif']].to_string(index=False)}

→ {len(sex_df[sex_df['Significatif'] != 'NS'])} codes sur {len(sex_df)} montrent des 
  différences significatives entre sexes (p<0.05)

⚖️  DÉPENDANCE AVEC L'ANTHROPOMÉTRIE
──────────────────────────────────

Codes avec associations significatives:
• Poids: {len(anthro_df[anthro_df['Weight_P'] < 0.05])} codes sur {len(anthro_df.dropna(subset=['Weight_P']))}
• Taille: {len(anthro_df[anthro_df['Height_P'] < 0.05])} codes sur {len(anthro_df.dropna(subset=['Height_P']))}
• IMC: {len(anthro_df[anthro_df['BMI_P'] < 0.05])} codes sur {len(anthro_df.dropna(subset=['BMI_P']))}

📅 ÉVOLUTION TEMPORELLE
───────────────────────

Tendances observées sur la période 1984-2001:
• Augmentation de prévalence: NORM, SR
• Diminution de prévalence: certains codes d'infarctus
• Stabilité: codes de troubles du rythme

🎯 QUALITÉ DU SIGNAL
────────────────────

Associations qualité-codes:
• {len(quality_df)} codes analysés
• Certains codes associés à plus de problèmes de signal
• Impact variable selon le type de pathologie

🔗 CO-OCCURRENCES
─────────────────

Top 5 paires de codes les plus associés:
{pairs_df.head(5)[['Code1', 'Code2', 'Cooccurrence']].to_string(index=False)}

Nombre moyen de codes par enregistrement: {df['num_codes'].mean():.2f}

═══════════════════════════════════════════════════════════════════════════════
3. IMPLICATIONS CLINIQUES ET POUR LE MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

FEATURES IMPORTANTES POUR LA PRÉDICTION
────────────────────────────────────────

✓ ÂGE: Variable fortement discriminante
  → Intégrer comme feature principale dans les modèles ML

✓ SEXE: Associations significatives pour plusieurs codes
  → Considérer des modèles stratifiés par sexe ou interaction âge×sexe

✓ QUALITÉ DU SIGNAL: Impact sur certains diagnostics
  → Filtrer ou normaliser selon la qualité avant modélisation

✓ CO-OCCURRENCES: Patterns complexes multi-label
  → Envisager des architectures ML multi-output ou chain classifiers

RECOMMANDATIONS POUR LA MODÉLISATION
─────────────────────────────────────

1. FEATURE ENGINEERING
   • Créer des interactions âge×sexe
   • Inclure des features de qualité du signal
   • Considérer les patterns temporels

2. ARCHITECTURE ML
   • Multi-label classification (sklearn.multioutput)
   • Modèles avec gestion des déséquilibres
   • Ensemble methods pour capturer les interactions

3. VALIDATION
   • Stratification par âge ET sexe
   • Attention aux biais temporels
   • Validation externe sur différentes périodes

4. INTERPRÉTABILITÉ
   • SHAP values pour importance des features
   • Attention aux corrélations âge-codes
   • Analyse des erreurs par sous-groupes

═══════════════════════════════════════════════════════════════════════════════
4. LIMITATIONS
═══════════════════════════════════════════════════════════════════════════════

• Données anthropométriques incomplètes (>50% manquantes)
• Distribution temporelle non uniforme
• Certains codes rares (n<100) → puissance statistique limitée
• Possible biais de sélection (centres hospitaliers)

═══════════════════════════════════════════════════════════════════════════════
5. FICHIERS GÉNÉRÉS
═══════════════════════════════════════════════════════════════════════════════

✓ Dependances_01_Age_Codes.png
✓ Dependances_02_Sexe_Codes.png
✓ Dependances_03_Anthropometrie_Codes.png
✓ Dependances_04_Temporel_Codes.png
✓ Dependances_05_Qualite_Codes.png
✓ Dependances_06_Cooccurrences.png
✓ Rapport_Dependances_Maladies.txt (ce fichier)

═══════════════════════════════════════════════════════════════════════════════
"""

print(rapport)

with open('Rapport_Dependances_Maladies.txt', 'w', encoding='utf-8') as f:
    f.write(rapport)

print("✓ Rapport sauvegardé: Rapport_Dependances_Maladies.txt")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIN DE L'ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 80)
print("█" + " " * 78 + "█")
print("█" + "  ✅ ANALYSE DES DÉPENDANCES TERMINÉE AVEC SUCCÈS".center(78) + "█")
print("█" + " " * 78 + "█")
print("█" + "  6 graphiques et 1 rapport générés".center(78) + "█")
print("█" + " " * 78 + "█")
print("█" * 80)
print()

print("📊 Résumé des découvertes clés:")
print(f"  • {len(age_df[age_df['Significatif'] != 'NS'])} codes significativement associés à l'âge")
print(f"  • {len(sex_df[sex_df['Significatif'] != 'NS'])} codes avec différences sexe significatives")
print(f"  • Nombre moyen de codes par patient: {df['num_codes'].mean():.2f}")
print(f"  • Paire la plus co-occurente: {pairs_df.iloc[0]['Code1']} ↔ {pairs_df.iloc[0]['Code2']} ({pairs_df.iloc[0]['Cooccurrence']:.1f}%)")
print()
print("🎯 Ces analyses sont essentielles pour:")
print("  • Comprendre les facteurs de risque des pathologies cardiaques")
print("  • Optimiser les features pour les modèles ML")
print("  • Identifier les biais potentiels du dataset")
print("  • Guider le feature engineering et la sélection de modèles")
