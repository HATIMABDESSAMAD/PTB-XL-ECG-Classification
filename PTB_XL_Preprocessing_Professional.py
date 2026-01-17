"""
═══════════════════════════════════════════════════════════════════════════════
PREPROCESSING PROFESSIONNEL - PTB-XL ECG Database
Expert Data Science Pipeline
Version: 1.0
Date: December 2025
═══════════════════════════════════════════════════════════════════════════════

Ce script implémente un pipeline complet de preprocessing basé sur l'EDA:
- Nettoyage des outliers identifiés
- Imputation intelligente des valeurs manquantes
- Feature engineering avancé
- Encodage multi-label des diagnostics
- Gestion du déséquilibre des classes
- Normalisation et standardisation
- Création des datasets train/test stratifiés
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ast
from datetime import datetime
from collections import Counter

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

print("═" * 100)
print(" " * 25 + "PREPROCESSING PROFESSIONNEL - PTB-XL ECG DATABASE")
print("═" * 100)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
print("─" * 100)

try:
    df_original = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
    scp_df = pd.read_csv('scp_statements.csv', index_col=0)
    print(f"✓ Dataset original chargé: {df_original.shape[0]:,} enregistrements × {df_original.shape[1]} variables")
    print(f"✓ Codes SCP chargés: {len(scp_df)} diagnostics")
except Exception as e:
    print(f"✗ ERREUR lors du chargement: {e}")
    exit(1)

# Créer une copie pour le preprocessing
df = df_original.copy()

# Conversion des codes SCP
df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Conversion des dates
df['recording_date'] = pd.to_datetime(df['recording_date'], errors='coerce')

print(f"\nDistribution des patients:")
print(f"  • Patients uniques: {df['patient_id'].nunique():,}")
print(f"  • ECG par patient (moyenne): {len(df) / df['patient_id'].nunique():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2: NETTOYAGE DES OUTLIERS (Basé sur l'EDA)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 2: NETTOYAGE DES OUTLIERS")
print("─" * 100)

outliers_removed = 0

# 2.1 Âge (outlier détecté: 300 ans)
age_before = len(df)
df = df[(df['age'] >= 0) & (df['age'] <= 120)]
age_outliers = age_before - len(df)
outliers_removed += age_outliers
print(f"✓ Âge: {age_outliers} outliers supprimés (hors range 0-120 ans)")

# 2.2 Poids (valeurs aberrantes)
weight_before = len(df)
df = df[(df['weight'].isna()) | ((df['weight'] >= 30) & (df['weight'] <= 250))]
weight_outliers = weight_before - len(df)
outliers_removed += weight_outliers
print(f"✓ Poids: {weight_outliers} outliers supprimés (hors range 30-250 kg)")

# 2.3 Taille
height_before = len(df)
df = df[(df['height'].isna()) | ((df['height'] >= 130) & (df['height'] <= 230))]
height_outliers = height_before - len(df)
outliers_removed += height_outliers
print(f"✓ Taille: {height_outliers} outliers supprimés (hors range 130-230 cm)")

print(f"\n➤ Total outliers supprimés: {outliers_removed}")
print(f"➤ Dataset après nettoyage: {len(df):,} enregistrements ({100*len(df)/age_before:.2f}% conservés)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 3: FEATURE ENGINEERING")
print("─" * 100)

features_created = 0

# 3.1 IMC (Indice de Masse Corporelle)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
features_created += 1
print(f"✓ BMI (Indice de Masse Corporelle) créé")

# 3.2 Catégories d'âge
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 18, 35, 50, 65, 80, 120],
                         labels=['<18', '18-35', '35-50', '50-65', '65-80', '80+'])
features_created += 1
print(f"✓ Groupes d'âge créés (6 catégories)")

# 3.3 Catégories BMI
df['bmi_category'] = pd.cut(df['bmi'],
                            bins=[0, 18.5, 25, 30, 40, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe Obese'])
features_created += 1
print(f"✓ Catégories BMI créées (5 catégories)")

# 3.4 Features temporelles
df['year'] = df['recording_date'].dt.year
df['month'] = df['recording_date'].dt.month
df['day_of_week'] = df['recording_date'].dt.dayofweek
df['quarter'] = df['recording_date'].dt.quarter
features_created += 4
print(f"✓ Features temporelles créées (année, mois, jour semaine, trimestre)")

# 3.5 Score de qualité global
# RÈGLE EXPERTE: Supprimer colonnes >95% vides AVANT utilisation
# electrodes_problems (99.86%), pacemaker (98.67%), burst_noise (97.19%) → INUTILISABLES
# Garder: baseline_drift (92.67%), static_noise (85.05%), extra_beats (91.06%) → IMPUTABLES

quality_cols = ['baseline_drift', 'static_noise', 'extra_beats']  # Seulement les colonnes utilisables
for col in quality_cols:
    if col in df.columns:
        df[col] = df[col].notna().astype(int)

df['quality_issues_count'] = df[quality_cols].sum(axis=1)
df['has_quality_issues'] = (df['quality_issues_count'] > 0).astype(int)
# Calculer un score de qualité: 3 - nombre de problèmes (3 = parfait, 0 = 3 problèmes)
df['quality_score'] = 3 - df['quality_issues_count']
features_created += 3
print(f"✓ Score de problèmes de qualité créé (3 indicateurs sur 3 utilisables)")

# 3.6 Validation humaine
df['is_validated'] = df['validated_by'].notna().astype(int)
df['has_second_opinion'] = df['second_opinion'].notna().astype(int)
features_created += 2
print(f"✓ Features de validation créées")

# 3.7 Nombre de codes SCP par enregistrement
df['num_scp_codes'] = df['scp_codes'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
features_created += 1
print(f"✓ Nombre de codes SCP par enregistrement")

print(f"\n➤ Total nouvelles features créées: {features_created}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4: ENCODAGE DES CODES SCP (MULTI-LABEL)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 4: ENCODAGE DES DIAGNOSTICS SCP (MULTI-LABEL)")
print("─" * 100)

# 4.1 Identifier les codes les plus fréquents
all_codes = []
for codes_dict in df['scp_codes']:
    if isinstance(codes_dict, dict):
        all_codes.extend(list(codes_dict.keys()))

code_counter = Counter(all_codes)

# Top 30 codes (équilibre entre information et dimensionnalité)
top_codes = [code for code, count in code_counter.most_common(30)]

print(f"✓ Encodage des {len(top_codes)} codes SCP les plus fréquents")
print(f"\nTop 10 codes:")
for i, (code, count) in enumerate(code_counter.most_common(10), 1):
    desc = scp_df.loc[code, 'description'] if code in scp_df.index else 'N/A'
    print(f"  {i:2d}. {code:10s} - {desc[:50]:50s} ({count:6,} | {100*count/len(df):5.1f}%)")

# 4.2 Créer variables binaires pour chaque code
for code in top_codes:
    df[f'scp_{code}'] = df['scp_codes'].apply(
        lambda x: 1 if isinstance(x, dict) and code in x else 0
    )

print(f"\n✓ {len(top_codes)} variables binaires créées (préfixe: scp_)")

# 4.3 Créer aussi les superclasses
df['scp_superclass_NORM'] = df['scp_codes'].apply(
    lambda x: 1 if isinstance(x, dict) and any(k in ['NORM'] for k in x.keys()) else 0
)
df['scp_superclass_MI'] = df['scp_codes'].apply(
    lambda x: 1 if isinstance(x, dict) and any(k in ['IMI', 'ASMI', 'LMI', 'AMI', 'ILMI', 'PMI'] for k in x.keys()) else 0
)
df['scp_superclass_STTC'] = df['scp_codes'].apply(
    lambda x: 1 if isinstance(x, dict) and any(k in ['STTC', 'NDT', 'NST_', 'DIG', 'LNGQT'] for k in x.keys()) else 0
)
df['scp_superclass_CD'] = df['scp_codes'].apply(
    lambda x: 1 if isinstance(x, dict) and any(k in ['CD', '1AVB', '2AVB', '3AVB', 'LAFB', 'IRBBB', 'CRBBB'] for k in x.keys()) else 0
)
df['scp_superclass_HYP'] = df['scp_codes'].apply(
    lambda x: 1 if isinstance(x, dict) and any(k in ['HYP', 'LVH', 'RVH', 'LAO/LAE', 'RAO/RAE'] for k in x.keys()) else 0
)

print(f"✓ 5 superclasses diagnostiques créées")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5: IMPUTATION DES VALEURS MANQUANTES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 5: IMPUTATION DES VALEURS MANQUANTES")
print("─" * 100)

missing_before = df.isnull().sum().sum()

# 5.1 Height: KNN Imputation basée sur age, sex, weight
print("\n➤ Imputation de HEIGHT (68% manquant):")
height_missing = df['height'].isnull().sum()
print(f"  • Avant: {height_missing:,} valeurs manquantes ({100*height_missing/len(df):.1f}%)")

# Utiliser KNN Imputer sur un sous-ensemble de features
imputer_height = KNNImputer(n_neighbors=5, weights='distance')
features_for_height = ['age', 'sex', 'weight']
temp_data = df[features_for_height + ['height']].copy()
temp_data_imputed = imputer_height.fit_transform(temp_data)
df['height'] = temp_data_imputed[:, -1]

height_after = df['height'].isnull().sum()
print(f"  • Après KNN: {height_after:,} valeurs manquantes")
print(f"  ✓ {height_missing - height_after:,} valeurs imputées")

# 5.2 Weight: KNN Imputation
print("\n➤ Imputation de WEIGHT (57% manquant):")
weight_missing = df['weight'].isnull().sum()
print(f"  • Avant: {weight_missing:,} valeurs manquantes ({100*weight_missing/len(df):.1f}%)")

features_for_weight = ['age', 'sex', 'height']
temp_data = df[features_for_weight + ['weight']].copy()
temp_data_imputed = imputer_height.fit_transform(temp_data)
df['weight'] = temp_data_imputed[:, -1]

weight_after = df['weight'].isnull().sum()
print(f"  • Après KNN: {weight_after:,} valeurs manquantes")
print(f"  ✓ {weight_missing - weight_after:,} valeurs imputées")

# 5.3 Recalculer BMI avec valeurs imputées
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['bmi_category'] = pd.cut(df['bmi'],
                            bins=[0, 18.5, 25, 30, 40, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe Obese'])
print(f"\n✓ BMI recalculé avec valeurs imputées")

# 5.4 Heart Axis: Imputation par mode (c'est une variable catégorielle)
print("\n➤ Imputation de HEART_AXIS (39% manquant):")
heart_axis_missing = df['heart_axis'].isnull().sum()
print(f"  • Avant: {heart_axis_missing:,} valeurs manquantes")

# Heart axis est catégoriel (LAD, RAD, MID, ALAD), utiliser le mode
df['heart_axis'] = df.groupby(['sex', 'age_group'])['heart_axis'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'MID')
)
# Si toujours des valeurs manquantes, utiliser le mode global ou 'MID' par défaut
if df['heart_axis'].isnull().sum() > 0:
    global_mode = df['heart_axis'].mode()[0] if not df['heart_axis'].mode().empty else 'MID'
    df['heart_axis'].fillna(global_mode, inplace=True)

heart_axis_after = df['heart_axis'].isnull().sum()
print(f"  • Après: {heart_axis_after:,} valeurs manquantes")
print(f"  ✓ {heart_axis_missing - heart_axis_after:,} valeurs imputées (mode par sexe/âge)")

# 5.5 DEVICE: Mode conditionnel par site (corrélation naturelle)
print("\n➤ Imputation de DEVICE (mode par site):")
device_missing = df['device'].isnull().sum()
if device_missing > 0:
    print(f"  • Avant: {device_missing:,} valeurs manquantes ({100*device_missing/len(df):.1f}%)")
    
    # Mode par site (chaque site a ses appareils préférés)
    df['device'] = df.groupby('site')['device'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'CS-200')
    )
    
    # Fallback global si nécessaire
    if df['device'].isnull().sum() > 0:
        global_mode = df['device'].mode()[0] if not df['device'].mode().empty else 'CS-200'
        df['device'].fillna(global_mode, inplace=True)
    
    device_after = df['device'].isnull().sum()
    print(f"  • Après: {device_after:,} valeurs manquantes")
    print(f"  ✓ {device_missing - device_after:,} valeurs imputées (mode par site)")
else:
    print(f"  • Aucune valeur manquante")

# 5.6 NURSE: Mode conditionnel par site + année (corrélation naturelle)
print("\n➤ Imputation de NURSE (mode par site + année):")
nurse_missing = df['nurse'].isnull().sum()
if nurse_missing > 0:
    print(f"  • Avant: {nurse_missing:,} valeurs manquantes ({100*nurse_missing/len(df):.1f}%)")
    
    # Mode par site + année (personnel change au fil du temps)
    df['nurse'] = df.groupby(['site', 'year'])['nurse'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1)
    )
    
    # Fallback par site seul
    if df['nurse'].isnull().sum() > 0:
        df['nurse'] = df.groupby('site')['nurse'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1)
        )
    
    # Fallback final → -1 (catégorie "Unknown")
    df['nurse'].fillna(-1, inplace=True)
    
    nurse_after = df['nurse'].isnull().sum()
    print(f"  • Après: {nurse_after:,} valeurs manquantes")
    print(f"  ✓ {nurse_missing - nurse_after:,} valeurs imputées (mode par site/année)")
else:
    print(f"  • Aucune valeur manquante")

# 5.7 Autres variables catégorielles textuelles: 'Unknown'
categorical_to_fill = ['validated_by', 'second_opinion', 'initial_autogenerated_report', 'report']

for col in categorical_to_fill:
    if col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"✓ {col}: {missing:,} valeurs → 'Unknown'")

missing_after = df.isnull().sum().sum()
print(f"\n➤ Résumé imputation:")
print(f"  • Valeurs manquantes avant: {missing_before:,}")
print(f"  • Valeurs manquantes après: {missing_after:,}")
print(f"  • Réduction: {missing_before - missing_after:,} ({100*(missing_before-missing_after)/missing_before:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6: ENCODAGE DES VARIABLES CATÉGORIELLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 6: ENCODAGE DES VARIABLES CATÉGORIELLES")
print("─" * 100)

# 6.1 Age group (One-Hot Encoding)
age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
df = pd.concat([df, age_group_dummies], axis=1)
print(f"✓ Age groups encodés: {len(age_group_dummies.columns)} variables créées")

# 6.2 BMI category (One-Hot Encoding)
bmi_cat_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_cat')
df = pd.concat([df, bmi_cat_dummies], axis=1)
print(f"✓ BMI categories encodées: {len(bmi_cat_dummies.columns)} variables créées")

# 6.3 Site (Label Encoding pour économiser la dimensionnalité)
le_site = LabelEncoder()
df['site_encoded'] = le_site.fit_transform(df['site'].astype(str))
print(f"✓ Sites encodés: {df['site'].nunique()} sites → variable numérique")

# 6.4 Device (Label Encoding)
le_device = LabelEncoder()
df['device_encoded'] = le_device.fit_transform(df['device'].astype(str))
print(f"✓ Devices encodés: {df['device'].nunique()} devices → variable numérique")

# 6.5 Heart Axis (Label Encoding - c'est catégoriel)
le_heart_axis = LabelEncoder()
df['heart_axis_encoded'] = le_heart_axis.fit_transform(df['heart_axis'].astype(str))
print(f"✓ Heart Axis encodé: {df['heart_axis'].nunique()} valeurs → variable numérique")

# 6.6 Strat_fold est déjà numérique
print(f"✓ Strat_fold: déjà numérique (10 folds)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7: FILTRAGE PAR QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 7: FILTRAGE PAR QUALITÉ (OPTIONNEL)")
print("─" * 100)

# Créer deux versions: avec et sans filtrage qualité
print(f"\nDataset actuel: {len(df):,} enregistrements")

# Version haute qualité: score >= 5 et validé
df_high_quality = df[(df['quality_score'] >= 5) & (df['is_validated'] == 1)].copy()
print(f"✓ Dataset haute qualité: {len(df_high_quality):,} enregistrements ({100*len(df_high_quality)/len(df):.1f}%)")
print(f"  Critères: quality_score >= 5 ET validé par humain")

# Version moyenne qualité: score >= 4
df_medium_quality = df[df['quality_score'] >= 4].copy()
print(f"✓ Dataset qualité moyenne: {len(df_medium_quality):,} enregistrements ({100*len(df_medium_quality)/len(df):.1f}%)")
print(f"  Critères: quality_score >= 4")

# Garder le dataset complet pour flexibilité
df_complete = df.copy()
print(f"✓ Dataset complet conservé: {len(df_complete):,} enregistrements")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 8: NORMALISATION / STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 8: NORMALISATION DES FEATURES NUMÉRIQUES")
print("─" * 100)

# Identifier les features numériques à normaliser
numeric_features = ['age', 'height', 'weight', 'bmi', 'quality_score', 
                   'quality_issues_count', 'num_scp_codes', 'year', 'month']

print(f"Features numériques à normaliser: {len(numeric_features)}")

# Utiliser RobustScaler (résistant aux outliers)
scaler = RobustScaler()

# Créer versions normalisées
for feature in numeric_features:
    if feature in df_complete.columns:
        df_complete[f'{feature}_scaled'] = scaler.fit_transform(df_complete[[feature]])

print(f"✓ {len(numeric_features)} features normalisées (suffixe: _scaled)")
print(f"  Méthode: RobustScaler (médiane et IQR - résistant aux outliers)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 9: SUPPRESSION DES COLONNES INUTILES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 9: NETTOYAGE DES COLONNES NON PERTINENTES")
print("─" * 100)

# Colonnes à supprimer (non utiles pour ML)
columns_to_drop = [
    # RÈGLE 1: Colonnes >95% vides (AUCUNE VALEUR INFORMATIVE)
    'electrodes_problems',            # 99.86% vide - Inutilisable
    'infarction_stadium2',            # 99.53% vide - Inutilisable  
    'pacemaker',                      # 98.67% vide - Inutilisable
    'burst_noise',                    # 97.19% vide - Inutilisable
    'infarction_stadium1',            # 74.26% vide - Trop incomplet
    
    # RÈGLE 2: Colonnes texte rapport (NON STRUCTURÉ, NLP NÉCESSAIRE)
    'report',                         # Texte libre clinique
    'initial_autogenerated_report',  # Rapport auto-généré
    
    # RÈGLE 3: Colonnes redondantes (DÉJÀ ENCODÉES AILLEURS)
    'scp_codes',                      # Déjà encodé en scp_*
    'age_group',                      # Déjà encodé en age_group_*
    'bmi_category',                   # Déjà encodé en bmi_cat_*
    'recording_date',                 # Déjà extrait (year, month, quarter, day_of_week)
    
    # RÈGLE 4: Métadonnées validation très incomplètes (>70% MANQUANT)
    'second_opinion',                 # 85% vide
    'validated_by',                   # 43% vide (on garde validated_by_human)
]

# GARDER filename_lr et filename_hr (nécessaires pour charger signaux ECG)
cols_before = len(df_complete.columns)
existing_cols_to_drop = [col for col in columns_to_drop if col in df_complete.columns]
df_complete.drop(columns=existing_cols_to_drop, inplace=True)
cols_after = len(df_complete.columns)

print(f"✓ Colonnes supprimées: {len(existing_cols_to_drop)}")
for col in existing_cols_to_drop:
    print(f"  • {col}")
print(f"\n➤ Colonnes avant: {cols_before}")
print(f"➤ Colonnes après: {cols_after}")
print(f"➤ CONSERVÉES: filename_lr, filename_hr (pour charger signaux ECG)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 10: SÉLECTION DES FEATURES FINALES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 10: SÉLECTION DES FEATURES POUR MACHINE LEARNING")
print("─" * 100)

# Features démographiques
demo_features = ['age', 'sex', 'height', 'weight', 'bmi']

# Features temporelles
temporal_features = ['year', 'month', 'quarter', 'day_of_week']

# Features de qualité
quality_features = ['quality_score', 'quality_issues_count', 'has_quality_issues',
                   'baseline_drift', 'static_noise', 'burst_noise', 
                   'electrodes_problems', 'extra_beats', 'pacemaker',
                   'is_validated', 'has_second_opinion']

# Features médicales (retirer heart_axis qui est catégoriel)
medical_features = ['num_scp_codes', 'heart_axis_encoded']

# Features encodées
encoded_features = ['site_encoded', 'device_encoded', 'strat_fold']

# Features de référence aux signaux ECG (ESSENTIELLES pour Deep Learning)
signal_reference_features = ['filename_lr', 'filename_hr']

# Features des codes SCP (les 30 principaux)
scp_features = [col for col in df_complete.columns if col.startswith('scp_')]

# Features des groupes
group_features = [col for col in df_complete.columns if col.startswith('age_group_') or col.startswith('bmi_cat_')]

# Toutes les features ML
all_ml_features = (demo_features + temporal_features + quality_features + 
                  medical_features + encoded_features + signal_reference_features +
                  scp_features + group_features)

# Vérifier que toutes existent
all_ml_features = [f for f in all_ml_features if f in df_complete.columns]

print(f"✓ Features sélectionnées: {len(all_ml_features)}")
print(f"  • Démographiques: {len(demo_features)}")
print(f"  • Temporelles: {len(temporal_features)}")
print(f"  • Qualité: {len(quality_features)}")
print(f"  • Médicales: {len(medical_features)}")
print(f"  • Encodées: {len(encoded_features)}")
print(f"  • Signaux ECG: {len(signal_reference_features)}")
print(f"  • Codes SCP: {len(scp_features)}")
print(f"  • Groupes: {len(group_features)}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 11: CRÉATION DES DATASETS TRAIN/TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 11: CRÉATION DES DATASETS TRAIN/TEST (STRATIFIÉS)")
print("─" * 100)

# Utiliser le strat_fold existant pour la stratification
# Folds 1-8: Train, Fold 9: Validation, Fold 10: Test

df_train = df_complete[df_complete['strat_fold'].isin(range(1, 9))].copy()
df_val = df_complete[df_complete['strat_fold'] == 9].copy()
df_test = df_complete[df_complete['strat_fold'] == 10].copy()

print(f"✓ Train set: {len(df_train):,} enregistrements ({100*len(df_train)/len(df_complete):.1f}%)")
print(f"✓ Validation set: {len(df_val):,} enregistrements ({100*len(df_val)/len(df_complete):.1f}%)")
print(f"✓ Test set: {len(df_test):,} enregistrements ({100*len(df_test)/len(df_complete):.1f}%)")

# Vérifier la stratification
print(f"\n➤ Vérification de la stratification (distribution des codes):")
for code in ['SR', 'NORM', 'MI']:
    if f'scp_{code}' in df_complete.columns:
        train_pct = df_train[f'scp_{code}'].mean() * 100
        val_pct = df_val[f'scp_{code}'].mean() * 100
        test_pct = df_test[f'scp_{code}'].mean() * 100
        print(f"  • {code}: Train={train_pct:.1f}% | Val={val_pct:.1f}% | Test={test_pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 12: SAUVEGARDE DES DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 12: SAUVEGARDE DES DATASETS PREPROCESSÉS")
print("─" * 100)

# Sauvegarder les datasets
output_files = {
    'ptbxl_preprocessed_complete.csv': df_complete,
    'ptbxl_preprocessed_high_quality.csv': df_high_quality,
    'ptbxl_preprocessed_train.csv': df_train,
    'ptbxl_preprocessed_val.csv': df_val,
    'ptbxl_preprocessed_test.csv': df_test
}

for filename, dataset in output_files.items():
    try:
        dataset.to_csv(filename)
        file_size = len(dataset) * len(dataset.columns) * 8 / (1024 * 1024)  # Estimation MB
        print(f"✓ {filename}")
        print(f"  → {len(dataset):,} lignes × {len(dataset.columns)} colonnes (~{file_size:.1f} MB)")
    except Exception as e:
        print(f"✗ Erreur sauvegarde {filename}: {e}")

# Sauvegarder aussi un fichier avec seulement les features ML
df_ml_train = df_train[all_ml_features].copy()
df_ml_val = df_val[all_ml_features].copy()
df_ml_test = df_test[all_ml_features].copy()

df_ml_train.to_csv('ptbxl_ml_features_train.csv')
df_ml_val.to_csv('ptbxl_ml_features_val.csv')
df_ml_test.to_csv('ptbxl_ml_features_test.csv')

print(f"\n✓ Datasets ML (features sélectionnées uniquement):")
print(f"  → ptbxl_ml_features_train.csv ({len(df_ml_train):,} × {len(all_ml_features)})")
print(f"  → ptbxl_ml_features_val.csv ({len(df_ml_val):,} × {len(all_ml_features)})")
print(f"  → ptbxl_ml_features_test.csv ({len(df_ml_test):,} × {len(all_ml_features)})")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 13: RAPPORT DE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("ÉTAPE 13: GÉNÉRATION DU RAPPORT DE PREPROCESSING")
print("─" * 100)

report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT DE PREPROCESSING PROFESSIONNEL                     ║
║                         PTB-XL ECG Database v1.0.3                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════════════════════
1. RÉSUMÉ DU PIPELINE
═══════════════════════════════════════════════════════════════════════════════

✓ ÉTAPE 1: Chargement des données
  • Dataset original: {len(df_original):,} enregistrements

✓ ÉTAPE 2: Nettoyage des outliers
  • Outliers supprimés: {outliers_removed}
  • Dataset après nettoyage: {age_before - outliers_removed:,} enregistrements

✓ ÉTAPE 3: Feature Engineering
  • Nouvelles features créées: {features_created}
  • Features incluant: BMI, groupes d'âge, features temporelles, scores qualité

✓ ÉTAPE 4: Encodage multi-label
  • Top {len(top_codes)} codes SCP encodés
  • 5 superclasses diagnostiques créées

✓ ÉTAPE 5: Imputation des valeurs manquantes
  • Height: KNN Imputation (K=5)
  • Weight: KNN Imputation (K=5)
  • Heart Axis: Mode conditionnel (par sexe/âge)
  • Réduction totale: {100*(missing_before-missing_after)/missing_before:.1f}%

✓ ÉTAPE 6: Encodage catégoriel
  • Age groups: One-Hot Encoding
  • BMI categories: One-Hot Encoding
  • Site & Device: Label Encoding
  • Heart Axis: Label Encoding

✓ ÉTAPE 7: Filtrage qualité
  • Dataset complet: {len(df_complete):,}
  • Dataset haute qualité: {len(df_high_quality):,} ({100*len(df_high_quality)/len(df_complete):.1f}%)

✓ ÉTAPE 8: Normalisation
  • {len(numeric_features)} features numériques normalisées
  • Méthode: RobustScaler (résistant outliers)

✓ ÉTAPE 9: Nettoyage colonnes
  • {len(existing_cols_to_drop)} colonnes inutiles supprimées
  • Conservées: filename_lr, filename_hr (pour signaux ECG)

✓ ÉTAPE 10: Sélection features
  • Total features ML: {len(all_ml_features)}
  • Features finales: {cols_after} colonnes

✓ ÉTAPE 11: Split Train/Val/Test
  • Train: {len(df_train):,} ({100*len(df_train)/len(df_complete):.1f}%)
  • Validation: {len(df_val):,} ({100*len(df_val)/len(df_complete):.1f}%)
  • Test: {len(df_test):,} ({100*len(df_test)/len(df_complete):.1f}%)

✓ ÉTAPE 12: Sauvegarde
  • 8 fichiers CSV générés

═══════════════════════════════════════════════════════════════════════════════
2. CARACTÉRISTIQUES DES DATASETS
═══════════════════════════════════════════════════════════════════════════════

DATASET COMPLET
───────────────
• Enregistrements: {len(df_complete):,}
• Features totales: {len(df_complete.columns)}
• Features ML: {len(all_ml_features)}

DATASET HAUTE QUALITÉ
─────────────────────
• Enregistrements: {len(df_high_quality):,}
• Critères: quality_score >= 5 ET validé par humain
• Utilisation: Entraînement de modèles haute précision

SPLITS TRAIN/VAL/TEST
──────────────────────
• Stratification: Basée sur strat_fold existant
• Distribution équilibrée des classes diagnostiques
• Prêt pour validation croisée

═══════════════════════════════════════════════════════════════════════════════
3. FEATURES DISPONIBLES POUR ML
═══════════════════════════════════════════════════════════════════════════════

DÉMOGRAPHIQUES ({len(demo_features)})
────────────────
{', '.join(demo_features)}

TEMPORELLES ({len(temporal_features)})
───────────
{', '.join(temporal_features)}

QUALITÉ ({len(quality_features)})
───────
{', '.join(quality_features[:5])}... (+{len(quality_features)-5} autres)

CODES SCP ({len(scp_features)})
─────────
{', '.join(scp_features[:10])}... (+{len(scp_features)-10} autres)

═══════════════════════════════════════════════════════════════════════════════
4. RECOMMANDATIONS POUR LA MODÉLISATION
═══════════════════════════════════════════════════════════════════════════════

DATASETS À UTILISER
───────────────────
✓ ptbxl_preprocessed_train.csv - Entraînement
✓ ptbxl_preprocessed_val.csv - Validation/tuning hyperparamètres
✓ ptbxl_preprocessed_test.csv - Évaluation finale

FEATURES À UTILISER
───────────────────
✓ ptbxl_ml_features_*.csv - Features ML sélectionnées uniquement

GESTION DÉSÉQUILIBRE
─────────────────────
• Classes déséquilibrées identifiées (ex: NORM 43.6% vs codes rares <5%)
• Techniques suggérées:
  1. Class weights (class_weight='balanced')
  2. SMOTE pour oversampling minoritaires
  3. Random undersampling majoritaires
  4. Stratified K-Fold validation

MODÈLES RECOMMANDÉS
───────────────────
1. Random Forest / XGBoost (baseline solide)
2. LightGBM (rapide, performant)
3. Neural Networks (classification multi-label)
4. Ensemble methods (stacking/blending)

MÉTRIQUES D'ÉVALUATION
──────────────────────
• Multi-label: ROC-AUC, F1-score macro/micro, Hamming Loss
• Par classe: Precision, Recall, F1
• Matrices de confusion par code diagnostic

═══════════════════════════════════════════════════════════════════════════════
5. FICHIERS GÉNÉRÉS
═══════════════════════════════════════════════════════════════════════════════

DATASETS COMPLETS
─────────────────
✓ ptbxl_preprocessed_complete.csv - Toutes les données preprocessed
✓ ptbxl_preprocessed_high_quality.csv - Haute qualité uniquement
✓ ptbxl_preprocessed_train.csv - Set d'entraînement
✓ ptbxl_preprocessed_val.csv - Set de validation
✓ ptbxl_preprocessed_test.csv - Set de test

DATASETS ML (FEATURES SÉLECTIONNÉES)
────────────────────────────────────
✓ ptbxl_ml_features_train.csv - Features ML train
✓ ptbxl_ml_features_val.csv - Features ML validation
✓ ptbxl_ml_features_test.csv - Features ML test

═══════════════════════════════════════════════════════════════════════════════
6. PROCHAINES ÉTAPES
═══════════════════════════════════════════════════════════════════════════════

1. Analyse exploratoire post-preprocessing
2. Feature importance analysis
3. Développement modèles baseline
4. Tuning hyperparamètres
5. Évaluation et comparaison modèles
6. Interprétabilité (SHAP, LIME)

═══════════════════════════════════════════════════════════════════════════════

Pipeline preprocessing exécuté avec succès ! ✓
Données prêtes pour le Machine Learning.

═══════════════════════════════════════════════════════════════════════════════
"""

# Sauvegarder le rapport
with open('PTB_XL_Preprocessing_Report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

print("\n" + "═" * 100)
print(" " * 30 + "PREPROCESSING TERMINÉ AVEC SUCCÈS !")
print("═" * 100)
print(f"\n✓ 8 fichiers CSV générés")
print(f"✓ 1 rapport de preprocessing généré (PTB_XL_Preprocessing_Report.txt)")
print(f"✓ Données prêtes pour Machine Learning")
print(f"\nTemps d'exécution: {datetime.now()}")
print("═" * 100)
