"""
═══════════════════════════════════════════════════════════════════════════════
Analyse Exploratoire de Données (EDA) Professionnelle
Dataset: PTB-XL ECG Database
Version: 1.0.3
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
import wfdb

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Configuration des figures
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class PTBXLExplorer:
    """
    Classe principale pour l'analyse exploratoire du dataset PTB-XL
    """
    
    def __init__(self, database_path, scp_statements_path):
        """
        Initialisation de l'explorateur de données
        
        Args:
            database_path: Chemin vers ptbxl_database.csv
            scp_statements_path: Chemin vers scp_statements.csv
        """
        print("═" * 80)
        print("CHARGEMENT DES DONNÉES PTB-XL")
        print("═" * 80)
        
        # Chargement des données
        self.df = pd.read_csv(database_path, index_col='ecg_id')
        self.scp_df = pd.read_csv(scp_statements_path, index_col=0)
        
        # Conversion des codes SCP
        self.df['scp_codes'] = self.df['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        print(f"✓ Dataset principal chargé: {self.df.shape[0]:,} enregistrements ECG")
        print(f"✓ Dictionnaire SCP chargé: {self.scp_df.shape[0]} codes diagnostiques")
        print()
        
    def section_header(self, title):
        """Affiche un en-tête de section formaté"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    
    def overview(self):
        """Vue d'ensemble du dataset"""
        self.section_header("1. VUE D'ENSEMBLE DU DATASET")
        
        print("📊 Dimensions du dataset:")
        print(f"   • Nombre d'enregistrements: {self.df.shape[0]:,}")
        print(f"   • Nombre de variables: {self.df.shape[1]}")
        print(f"   • Nombre de patients uniques: {self.df['patient_id'].nunique():,}")
        print()
        
        print("📋 Structure des données:")
        print(self.df.info())
        print()
        
        print("🔍 Aperçu des premières lignes:")
        print(self.df.head())
        print()
        
        print("📈 Statistiques descriptives (variables numériques):")
        print(self.df.describe())
        print()
        
    def missing_values_analysis(self):
        """Analyse des valeurs manquantes"""
        self.section_header("2. ANALYSE DES VALEURS MANQUANTES")
        
        missing = pd.DataFrame({
            'Total_Missing': self.df.isnull().sum(),
            'Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        }).sort_values('Total_Missing', ascending=False)
        
        missing = missing[missing['Total_Missing'] > 0]
        
        print("🔴 Valeurs manquantes par variable:")
        print(missing)
        print()
        
        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Graphique en barres
        missing.sort_values('Percentage', ascending=True).plot(
            kind='barh', y='Percentage', ax=axes[0], color='coral', legend=False
        )
        axes[0].set_xlabel('Pourcentage de valeurs manquantes (%)')
        axes[0].set_title('Valeurs Manquantes par Variable')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Heatmap des valeurs manquantes
        sns.heatmap(
            self.df.isnull().transpose(), 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Valeur Manquante'},
            ax=axes[1],
            yticklabels=True,
            xticklabels=False
        )
        axes[1].set_title('Carte des Valeurs Manquantes')
        axes[1].set_xlabel('Index des enregistrements')
        
        plt.tight_layout()
        plt.savefig('01_missing_values_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 01_missing_values_analysis.png")
        plt.show()
        
    def demographic_analysis(self):
        """Analyse démographique"""
        self.section_header("3. ANALYSE DÉMOGRAPHIQUE")
        
        print("👥 Distribution par Sexe:")
        sex_dist = self.df['sex'].value_counts()
        sex_dist.index = sex_dist.index.map({0: 'Femme', 1: 'Homme'})
        print(sex_dist)
        print(f"\n   Ratio Homme/Femme: {sex_dist['Homme']/sex_dist['Femme']:.2f}")
        print()
        
        print("📅 Statistiques d'Âge:")
        print(f"   • Âge moyen: {self.df['age'].mean():.1f} ans")
        print(f"   • Âge médian: {self.df['age'].median():.1f} ans")
        print(f"   • Écart-type: {self.df['age'].std():.1f} ans")
        print(f"   • Étendue: [{self.df['age'].min():.0f} - {self.df['age'].max():.0f}] ans")
        print()
        
        # Visualisations
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Distribution de l'âge
        ax1 = fig.add_subplot(gs[0, :2])
        self.df['age'].hist(bins=50, edgecolor='black', alpha=0.7, ax=ax1, color='steelblue')
        ax1.axvline(self.df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {self.df["age"].mean():.1f}')
        ax1.axvline(self.df['age'].median(), color='green', linestyle='--', linewidth=2, label=f'Médiane: {self.df["age"].median():.1f}')
        ax1.set_xlabel('Âge (années)')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution de l\'Âge des Patients')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Distribution par sexe
        ax2 = fig.add_subplot(gs[0, 2])
        sex_counts = self.df['sex'].value_counts()
        colors_sex = ['#FF69B4', '#4169E1']
        labels_sex = ['Femme' if x == 0 else 'Homme' for x in sex_counts.index]
        wedges, texts, autotexts = ax2.pie(
            sex_counts.values, 
            labels=labels_sex, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_sex,
            explode=(0.05, 0.05)
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        ax2.set_title('Distribution par Sexe')
        
        # 3. Âge par sexe (Boxplot)
        ax3 = fig.add_subplot(gs[1, :2])
        df_temp = self.df.copy()
        df_temp['sex'] = df_temp['sex'].map({0: 'Femme', 1: 'Homme'})
        sns.boxplot(data=df_temp, x='sex', y='age', ax=ax3, palette=['#FF69B4', '#4169E1'])
        sns.stripplot(data=df_temp, x='sex', y='age', ax=ax3, color='black', alpha=0.2, size=2)
        ax3.set_ylabel('Âge (années)')
        ax3.set_xlabel('Sexe')
        ax3.set_title('Distribution de l\'Âge par Sexe')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Âge par sexe (Violin plot)
        ax4 = fig.add_subplot(gs[1, 2])
        sns.violinplot(data=df_temp, x='sex', y='age', ax=ax4, palette=['#FF69B4', '#4169E1'])
        ax4.set_ylabel('Âge (années)')
        ax4.set_xlabel('Sexe')
        ax4.set_title('Densité de l\'Âge par Sexe')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Distribution du poids
        ax5 = fig.add_subplot(gs[2, 0])
        self.df['weight'].dropna().hist(bins=40, edgecolor='black', alpha=0.7, ax=ax5, color='orange')
        ax5.set_xlabel('Poids (kg)')
        ax5.set_ylabel('Fréquence')
        ax5.set_title('Distribution du Poids')
        ax5.grid(alpha=0.3)
        
        # 6. Distribution de la taille
        ax6 = fig.add_subplot(gs[2, 1])
        self.df['height'].dropna().hist(bins=40, edgecolor='black', alpha=0.7, ax=ax6, color='green')
        ax6.set_xlabel('Taille (cm)')
        ax6.set_ylabel('Fréquence')
        ax6.set_title('Distribution de la Taille')
        ax6.grid(alpha=0.3)
        
        # 7. IMC (Body Mass Index)
        ax7 = fig.add_subplot(gs[2, 2])
        df_bmi = self.df[['height', 'weight']].dropna()
        df_bmi['bmi'] = df_bmi['weight'] / ((df_bmi['height'] / 100) ** 2)
        df_bmi['bmi'].hist(bins=40, edgecolor='black', alpha=0.7, ax=ax7, color='purple')
        ax7.axvline(25, color='orange', linestyle='--', linewidth=2, label='Surpoids (25)')
        ax7.axvline(30, color='red', linestyle='--', linewidth=2, label='Obésité (30)')
        ax7.set_xlabel('IMC (kg/m²)')
        ax7.set_ylabel('Fréquence')
        ax7.set_title('Distribution de l\'IMC')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        plt.savefig('02_demographic_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 02_demographic_analysis.png")
        plt.show()
        
    def diagnostic_analysis(self):
        """Analyse des diagnostics"""
        self.section_header("4. ANALYSE DES DIAGNOSTICS")
        
        # Extraction de tous les codes SCP
        all_scp_codes = []
        for codes_dict in self.df['scp_codes']:
            if isinstance(codes_dict, dict):
                all_scp_codes.extend(list(codes_dict.keys()))
        
        scp_counter = Counter(all_scp_codes)
        
        print(f"📊 Statistiques des Codes SCP:")
        print(f"   • Nombre total de codes SCP uniques: {len(scp_counter)}")
        print(f"   • Code le plus fréquent: {scp_counter.most_common(1)[0][0]} ({scp_counter.most_common(1)[0][1]:,} occurrences)")
        print()
        
        # Top 20 des diagnostics les plus fréquents
        print("🔝 Top 20 des Codes SCP les plus fréquents:")
        top_20_scp = scp_counter.most_common(20)
        for i, (code, count) in enumerate(top_20_scp, 1):
            description = self.scp_df.loc[code, 'description'] if code in self.scp_df.index else 'Description non disponible'
            print(f"   {i:2d}. {code:10s} - {description:50s} ({count:5,} occurrences)")
        print()
        
        # Classes diagnostiques
        print("🏥 Distribution par Classe Diagnostique:")
        diag_classes = self.scp_df[self.scp_df['diagnostic'] == 1.0]['diagnostic_class'].value_counts()
        for cls, count in diag_classes.items():
            print(f"   • {cls}: {count} codes")
        print()
        
        # Visualisations
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Top 15 codes SCP
        ax1 = fig.add_subplot(gs[0, :])
        top_15_codes = scp_counter.most_common(15)
        codes, counts = zip(*top_15_codes)
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(codes)))
        bars = ax1.barh(range(len(codes)), counts, color=colors_gradient)
        ax1.set_yticks(range(len(codes)))
        ax1.set_yticklabels(codes)
        ax1.set_xlabel('Nombre d\'occurrences')
        ax1.set_title('Top 15 des Codes SCP les plus fréquents')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count, i, f' {count:,}', va='center', fontweight='bold')
        
        # 2. Distribution des classes diagnostiques
        ax2 = fig.add_subplot(gs[1, 0])
        diag_counts = []
        diag_labels = []
        for cls in diag_classes.index:
            # Compter les enregistrements avec cette classe
            count = sum(1 for codes_dict in self.df['scp_codes'] 
                       if isinstance(codes_dict, dict) and 
                       any(code in self.scp_df[self.scp_df['diagnostic_class'] == cls].index 
                           for code in codes_dict.keys()))
            diag_counts.append(count)
            diag_labels.append(cls)
        
        colors_diag = plt.cm.Set3(np.linspace(0, 1, len(diag_labels)))
        wedges, texts, autotexts = ax2.pie(
            diag_counts, 
            labels=diag_labels, 
            autopct='%1.1f%%',
            colors=colors_diag,
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(9)
        ax2.set_title('Distribution par Classe Diagnostique')
        
        # 3. Nombre de codes SCP par enregistrement
        ax3 = fig.add_subplot(gs[1, 1])
        codes_per_record = [len(codes) if isinstance(codes, dict) else 0 for codes in self.df['scp_codes']]
        ax3.hist(codes_per_record, bins=range(0, max(codes_per_record)+2), edgecolor='black', alpha=0.7, color='teal')
        ax3.set_xlabel('Nombre de codes SCP par enregistrement')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution du Nombre de Codes par Enregistrement')
        ax3.axvline(np.mean(codes_per_record), color='red', linestyle='--', linewidth=2, 
                   label=f'Moyenne: {np.mean(codes_per_record):.2f}')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Catégories de déclarations SCP
        ax4 = fig.add_subplot(gs[2, :])
        statement_cats = self.scp_df['Statement Category'].value_counts()
        bars = ax4.bar(range(len(statement_cats)), statement_cats.values, color='skyblue', edgecolor='black')
        ax4.set_xticks(range(len(statement_cats)))
        ax4.set_xticklabels(statement_cats.index, rotation=45, ha='right')
        ax4.set_ylabel('Nombre de codes')
        ax4.set_title('Distribution par Catégorie de Déclaration SCP')
        ax4.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, count in zip(bars, statement_cats.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        plt.savefig('03_diagnostic_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 03_diagnostic_analysis.png")
        plt.show()
        
    def temporal_analysis(self):
        """Analyse temporelle"""
        self.section_header("5. ANALYSE TEMPORELLE")
        
        # Conversion des dates
        self.df['recording_date'] = pd.to_datetime(self.df['recording_date'], errors='coerce')
        self.df['year'] = self.df['recording_date'].dt.year
        self.df['month'] = self.df['recording_date'].dt.month
        self.df['day_of_week'] = self.df['recording_date'].dt.dayofweek
        
        print("📅 Période d'enregistrement:")
        print(f"   • Date la plus ancienne: {self.df['recording_date'].min()}")
        print(f"   • Date la plus récente: {self.df['recording_date'].max()}")
        print(f"   • Durée totale: {(self.df['recording_date'].max() - self.df['recording_date'].min()).days} jours")
        print()
        
        print("📊 Distribution par année:")
        yearly_dist = self.df['year'].value_counts().sort_index()
        print(yearly_dist)
        print()
        
        # Visualisations
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Évolution annuelle
        ax1 = fig.add_subplot(gs[0, :])
        yearly_counts = self.df['year'].value_counts().sort_index()
        ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='steelblue')
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Nombre d\'enregistrements')
        ax1.set_title('Évolution du Nombre d\'Enregistrements par Année')
        ax1.grid(alpha=0.3)
        
        # Ajouter les valeurs
        for x, y in zip(yearly_counts.index, yearly_counts.values):
            ax1.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Distribution mensuelle
        ax2 = fig.add_subplot(gs[1, 0])
        monthly_counts = self.df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                       'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        colors_months = plt.cm.Set3(np.linspace(0, 1, 12))
        bars = ax2.bar(monthly_counts.index, monthly_counts.values, color=colors_months, edgecolor='black')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_names, rotation=45, ha='right')
        ax2.set_ylabel('Nombre d\'enregistrements')
        ax2.set_title('Distribution par Mois')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Distribution par jour de la semaine
        ax3 = fig.add_subplot(gs[1, 1])
        dow_counts = self.df['day_of_week'].value_counts().sort_index()
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        colors_days = ['#FF6B6B' if i >= 5 else '#4ECDC4' for i in range(7)]
        bars = ax3.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], 
                      color=colors_days, edgecolor='black')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(day_names, rotation=45, ha='right')
        ax3.set_ylabel('Nombre d\'enregistrements')
        ax3.set_title('Distribution par Jour de la Semaine')
        ax3.grid(axis='y', alpha=0.3)
        
        plt.savefig('04_temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 04_temporal_analysis.png")
        plt.show()
        
    def technical_analysis(self):
        """Analyse des aspects techniques"""
        self.section_header("6. ANALYSE TECHNIQUE")
        
        print("🔧 Équipement et Infrastructure:")
        print(f"   • Nombre de sites d'enregistrement: {self.df['site'].nunique()}")
        print(f"   • Nombre d'appareils: {self.df['device'].nunique()}")
        print(f"   • Nombre d'infirmières: {self.df['nurse'].nunique()}")
        print()
        
        print("📊 Distribution des sites:")
        print(self.df['site'].value_counts())
        print()
        
        print("📱 Appareils utilisés:")
        print(self.df['device'].value_counts())
        print()
        
        print("📁 Stratification des plis (folds):")
        print(self.df['strat_fold'].value_counts().sort_index())
        print()
        
        # Visualisations
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Distribution des sites
        ax1 = fig.add_subplot(gs[0, 0])
        site_counts = self.df['site'].value_counts()
        ax1.bar(range(len(site_counts)), site_counts.values, color='coral', edgecolor='black')
        ax1.set_xlabel('Site')
        ax1.set_ylabel('Nombre d\'enregistrements')
        ax1.set_title('Distribution par Site d\'Enregistrement')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Distribution des appareils
        ax2 = fig.add_subplot(gs[0, 1:])
        device_counts = self.df['device'].value_counts()
        bars = ax2.barh(range(len(device_counts)), device_counts.values, color='skyblue', edgecolor='black')
        ax2.set_yticks(range(len(device_counts)))
        ax2.set_yticklabels(device_counts.index)
        ax2.set_xlabel('Nombre d\'enregistrements')
        ax2.set_title('Distribution par Appareil')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, device_counts.values)):
            ax2.text(count, i, f' {count:,}', va='center', fontweight='bold')
        
        # 3. Distribution des plis de stratification
        ax3 = fig.add_subplot(gs[1, 0])
        fold_counts = self.df['strat_fold'].value_counts().sort_index()
        colors_fold = plt.cm.viridis(np.linspace(0, 1, len(fold_counts)))
        bars = ax3.bar(fold_counts.index, fold_counts.values, color=colors_fold, edgecolor='black')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Nombre d\'enregistrements')
        ax3.set_title('Distribution par Fold de Stratification')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Qualité du signal - Baseline drift
        ax4 = fig.add_subplot(gs[1, 1])
        baseline_counts = self.df['baseline_drift'].value_counts()
        ax4.pie(baseline_counts.values, labels=['Non' if x == 0 else 'Oui' for x in baseline_counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
        ax4.set_title('Présence de Baseline Drift')
        
        # 5. Qualité du signal - Static noise
        ax5 = fig.add_subplot(gs[1, 2])
        static_counts = self.df['static_noise'].value_counts()
        ax5.pie(static_counts.values, labels=['Non' if x == 0 else 'Oui' for x in static_counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
        ax5.set_title('Présence de Static Noise')
        
        # 6. Qualité du signal - Burst noise
        ax6 = fig.add_subplot(gs[2, 0])
        burst_counts = self.df['burst_noise'].value_counts()
        ax6.pie(burst_counts.values, labels=['Non' if x == 0 else 'Oui' for x in burst_counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
        ax6.set_title('Présence de Burst Noise')
        
        # 7. Problèmes d'électrodes
        ax7 = fig.add_subplot(gs[2, 1])
        electrode_counts = self.df['electrodes_problems'].value_counts()
        ax7.pie(electrode_counts.values, labels=['Non' if x == 0 else 'Oui' for x in electrode_counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
        ax7.set_title('Problèmes d\'Électrodes')
        
        # 8. Validation humaine
        ax8 = fig.add_subplot(gs[2, 2])
        validated_counts = self.df['validated_by_human'].value_counts()
        ax8.pie(validated_counts.values, labels=['Non', 'Oui'], 
               autopct='%1.1f%%', startangle=90, colors=['#FFE4B5', '#98FB98'])
        ax8.set_title('Validation par un Humain')
        
        plt.savefig('05_technical_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 05_technical_analysis.png")
        plt.show()
        
    def quality_assessment(self):
        """Évaluation de la qualité des données"""
        self.section_header("7. ÉVALUATION DE LA QUALITÉ DES DONNÉES")
        
        # Conversion des colonnes de qualité en numérique (0/1)
        quality_cols = ['baseline_drift', 'static_noise', 'burst_noise', 
                       'electrodes_problems', 'extra_beats', 'pacemaker']
        
        for col in quality_cols:
            self.df[col] = self.df[col].notna().astype(int)
        
        # Calcul du score de qualité
        quality_issues = self.df[quality_cols].sum(axis=1)
        
        self.df['quality_score'] = 6 - quality_issues
        
        print("🎯 Score de Qualité des Enregistrements (0-6):")
        print(f"   • Score moyen: {self.df['quality_score'].mean():.2f}")
        print(f"   • Score médian: {self.df['quality_score'].median():.0f}")
        print()
        
        print("📊 Distribution des scores de qualité:")
        print(self.df['quality_score'].value_counts().sort_index(ascending=False))
        print()
        
        # Statistiques de validation
        print("✅ Statistiques de Validation:")
        print(f"   • Enregistrements validés par un humain: {self.df['validated_by_human'].sum():,} "
              f"({self.df['validated_by_human'].sum()/len(self.df)*100:.1f}%)")
        print(f"   • Enregistrements avec deuxième opinion: {self.df['second_opinion'].sum():,} "
              f"({self.df['second_opinion'].sum()/len(self.df)*100:.1f}%)")
        print()
        
        # Visualisations
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Distribution du score de qualité
        ax1 = fig.add_subplot(gs[0, :2])
        quality_counts = self.df['quality_score'].value_counts().sort_index()
        colors_quality = ['#FF0000', '#FF4500', '#FFA500', '#FFD700', '#ADFF2F', '#32CD32', '#228B22']
        bars = ax1.bar(quality_counts.index, quality_counts.values, 
                      color=[colors_quality[int(i)] for i in quality_counts.index], 
                      edgecolor='black')
        ax1.set_xlabel('Score de Qualité')
        ax1.set_ylabel('Nombre d\'enregistrements')
        ax1.set_title('Distribution du Score de Qualité (0=Mauvais, 6=Excellent)')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, quality_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/len(self.df)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Taux de problèmes de qualité
        ax2 = fig.add_subplot(gs[0, 2])
        quality_issues_pct = pd.Series({
            'baseline_drift': (self.df['baseline_drift'] == 1).sum(),
            'static_noise': (self.df['static_noise'] == 1).sum(),
            'burst_noise': (self.df['burst_noise'] == 1).sum(),
            'electrodes_problems': (self.df['electrodes_problems'] == 1).sum(),
            'extra_beats': (self.df['extra_beats'] == 1).sum(),
            'pacemaker': (self.df['pacemaker'] == 1).sum()
        }) / len(self.df) * 100
        quality_issues_pct = quality_issues_pct.sort_values(ascending=True)
        
        colors_issues = ['#90EE90' if x < 5 else '#FFD700' if x < 15 else '#FF6B6B' 
                        for x in quality_issues_pct.values]
        bars = ax2.barh(range(len(quality_issues_pct)), quality_issues_pct.values, 
                       color=colors_issues, edgecolor='black')
        ax2.set_yticks(range(len(quality_issues_pct)))
        ax2.set_yticklabels(['Baseline Drift' if x == 'baseline_drift' else
                            'Static Noise' if x == 'static_noise' else
                            'Burst Noise' if x == 'burst_noise' else
                            'Électrodes' if x == 'electrodes_problems' else
                            'Extra Beats' if x == 'extra_beats' else
                            'Pacemaker' for x in quality_issues_pct.index])
        ax2.set_xlabel('Pourcentage (%)')
        ax2.set_title('Taux de Problèmes de Qualité')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, quality_issues_pct.values)):
            ax2.text(val, i, f' {val:.1f}%', va='center', fontweight='bold')
        
        # 3. Matrice de corrélation des problèmes
        ax3 = fig.add_subplot(gs[1, :2])
        quality_cols = ['baseline_drift', 'static_noise', 'burst_noise', 
                       'electrodes_problems', 'extra_beats', 'pacemaker']
        
        # S'assurer que les colonnes sont numériques
        quality_data = self.df[quality_cols].copy()
        for col in quality_cols:
            quality_data[col] = pd.to_numeric(quality_data[col], errors='coerce').fillna(0)
        
        corr_matrix = quality_data.corr()
        
        labels_fr = ['Baseline\nDrift', 'Static\nNoise', 'Burst\nNoise', 
                    'Problèmes\nÉlectrodes', 'Extra\nBeats', 'Pacemaker']
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   xticklabels=labels_fr, yticklabels=labels_fr, ax=ax3)
        ax3.set_title('Matrice de Corrélation des Problèmes de Qualité')
        
        # 4. Comparaison qualité avec/sans validation humaine
        ax4 = fig.add_subplot(gs[1, 2])
        validated_quality = self.df[self.df['validated_by_human'] == True]['quality_score'].mean()
        not_validated_quality = self.df[self.df['validated_by_human'] == False]['quality_score'].mean()
        
        categories = ['Validé\npar humain', 'Non validé\npar humain']
        values = [validated_quality, not_validated_quality]
        colors_val = ['#32CD32', '#FFB6C1']
        
        bars = ax4.bar(categories, values, color=colors_val, edgecolor='black')
        ax4.set_ylabel('Score de Qualité Moyen')
        ax4.set_title('Score de Qualité selon Validation')
        ax4.set_ylim([0, 6])
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.savefig('06_quality_assessment.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 06_quality_assessment.png")
        plt.show()
        
    def correlation_analysis(self):
        """Analyse des corrélations"""
        self.section_header("8. ANALYSE DES CORRÉLATIONS")
        
        # Sélection des variables numériques
        numeric_cols = ['age', 'height', 'weight', 'site', 'nurse', 'strat_fold']
        numeric_data = self.df[numeric_cols].dropna()
        
        print("🔗 Matrice de Corrélation:")
        corr_matrix = numeric_data.corr()
        print(corr_matrix)
        print()
        
        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 1. Heatmap de corrélation
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                   ax=axes[0])
        axes[0].set_title('Matrice de Corrélation des Variables Numériques', fontsize=14, fontweight='bold')
        
        # 2. Scatter matrix pour variables clés
        ax2 = axes[1]
        # Relation âge vs poids
        valid_data = self.df[['age', 'weight']].dropna()
        scatter = ax2.scatter(valid_data['age'], valid_data['weight'], 
                             alpha=0.3, s=10, c=valid_data['age'], cmap='viridis')
        ax2.set_xlabel('Âge (années)')
        ax2.set_ylabel('Poids (kg)')
        ax2.set_title('Relation entre Âge et Poids')
        ax2.grid(alpha=0.3)
        
        # Ligne de régression
        z = np.polyfit(valid_data['age'], valid_data['weight'], 1)
        p = np.poly1d(z)
        ax2.plot(valid_data['age'].sort_values(), p(valid_data['age'].sort_values()), 
                "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        ax2.legend()
        
        plt.colorbar(scatter, ax=ax2, label='Âge')
        
        plt.tight_layout()
        plt.savefig('07_correlation_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: 07_correlation_analysis.png")
        plt.show()
        
    def generate_summary_report(self):
        """Génère un rapport résumé"""
        self.section_header("9. RAPPORT RÉSUMÉ")
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      RAPPORT D'ANALYSE EXPLORATOIRE                           ║
║                     Dataset PTB-XL ECG Database v1.0.3                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 STATISTIQUES GÉNÉRALES
─────────────────────────────────────────────────────────────────────────────────
• Nombre total d'enregistrements ECG:        {self.df.shape[0]:>10,}
• Nombre de patients uniques:                {self.df['patient_id'].nunique():>10,}
• Nombre de variables:                       {self.df.shape[1]:>10}
• Période d'enregistrement:                  {self.df['recording_date'].min().strftime('%Y-%m-%d')} à {self.df['recording_date'].max().strftime('%Y-%m-%d')}

👥 DÉMOGRAPHIE
─────────────────────────────────────────────────────────────────────────────────
• Âge moyen:                                 {self.df['age'].mean():>10.1f} ans
• Âge médian:                                {self.df['age'].median():>10.1f} ans
• Étendue d'âge:                             {self.df['age'].min():>10.0f} - {self.df['age'].max():.0f} ans
• Ratio Homme/Femme:                         {(self.df['sex']==1).sum()/(self.df['sex']==0).sum():>10.2f}:1

🏥 DIAGNOSTICS
─────────────────────────────────────────────────────────────────────────────────
• Nombre de codes SCP uniques:               {len(set(code for codes in self.df['scp_codes'] if isinstance(codes, dict) for code in codes.keys())):>10}
• Code le plus fréquent:                     {Counter([code for codes in self.df['scp_codes'] if isinstance(codes, dict) for code in codes.keys()]).most_common(1)[0][0]:>10}
• Moyenne de codes par enregistrement:       {np.mean([len(codes) if isinstance(codes, dict) else 0 for codes in self.df['scp_codes']]):>10.2f}

🎯 QUALITÉ DES DONNÉES
─────────────────────────────────────────────────────────────────────────────────
• Score de qualité moyen (0-6):              {self.df['quality_score'].mean():>10.2f}
• Enregistrements validés par humain:        {(self.df['validated_by_human']==True).sum():>10,} ({(self.df['validated_by_human']==True).sum()/len(self.df)*100:.1f}%)
• Enregistrements avec deuxième opinion:     {(self.df['second_opinion']==True).sum():>10,} ({(self.df['second_opinion']==True).sum()/len(self.df)*100:.1f}%)

⚠️  PROBLÈMES DE QUALITÉ (Taux d'occurrence)
─────────────────────────────────────────────────────────────────────────────────
• Baseline Drift:                            {(self.df['baseline_drift']==1).sum()/len(self.df)*100:>10.1f}%
• Static Noise:                              {(self.df['static_noise']==1).sum()/len(self.df)*100:>10.1f}%
• Burst Noise:                               {(self.df['burst_noise']==1).sum()/len(self.df)*100:>10.1f}%
• Problèmes d'électrodes:                    {(self.df['electrodes_problems']==1).sum()/len(self.df)*100:>10.1f}%
• Extra Beats:                               {(self.df['extra_beats']==1).sum()/len(self.df)*100:>10.1f}%
• Pacemaker:                                 {(self.df['pacemaker']==1).sum()/len(self.df)*100:>10.1f}%

🔧 INFRASTRUCTURE
─────────────────────────────────────────────────────────────────────────────────
• Nombre de sites:                           {self.df['site'].nunique():>10}
• Nombre d'appareils:                        {self.df['device'].nunique():>10}
• Nombre d'infirmières:                      {self.df['nurse'].nunique():>10}

📁 VALEURS MANQUANTES (Top 5)
─────────────────────────────────────────────────────────────────────────────────
"""
        # Top 5 valeurs manquantes
        missing = self.df.isnull().sum().sort_values(ascending=False).head()
        for col, count in missing.items():
            if count > 0:
                report += f"• {col:<40} {count:>10,} ({count/len(self.df)*100:>5.1f}%)\n"
        
        report += f"""
═══════════════════════════════════════════════════════════════════════════════
RECOMMANDATIONS
═══════════════════════════════════════════════════════════════════════════════

✓ Points Forts:
  • Dataset de grande taille avec {self.df.shape[0]:,} enregistrements
  • Bonne qualité globale (score moyen: {self.df['quality_score'].mean():.2f}/6)
  • Validation humaine pour {(self.df['validated_by_human']==True).sum()/len(self.df)*100:.1f}% des données
  • Stratification en folds pour validation croisée

⚠  Points d'Attention:
  • Certaines variables ont un taux élevé de valeurs manquantes
  • Distribution déséquilibrée entre hommes et femmes
  • Problèmes de qualité du signal présents dans une partie des données

💡 Suggestions pour la Suite:
  1. Imputation des valeurs manquantes pour height et weight
  2. Analyse de survie ou modélisation prédictive des diagnostics
  3. Traitement du signal ECG pour réduire le bruit
  4. Analyse approfondie des patterns ECG par classe diagnostique
  5. Développement de modèles de deep learning pour classification automatique

═══════════════════════════════════════════════════════════════════════════════
Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════════════
"""
        
        print(report)
        
        # Sauvegarde du rapport
        with open('PTB_XL_EDA_Summary_Report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✓ Rapport sauvegardé: PTB_XL_EDA_Summary_Report.txt")
        
    def run_complete_eda(self):
        """Exécute l'analyse complète"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  DÉBUT DE L'ANALYSE EXPLORATOIRE DE DONNÉES (EDA) PROFESSIONNELLE".center(78) + "█")
        print("█" + "  Dataset: PTB-XL ECG Database v1.0.3".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80 + "\n")
        
        # Exécution de toutes les analyses
        self.overview()
        self.missing_values_analysis()
        self.demographic_analysis()
        self.diagnostic_analysis()
        self.temporal_analysis()
        self.technical_analysis()
        self.quality_assessment()
        self.correlation_analysis()
        self.generate_summary_report()
        
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  ✓ ANALYSE EXPLORATOIRE TERMINÉE AVEC SUCCÈS".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" + "  Tous les graphiques et rapports ont été générés.".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Chemins des fichiers (à adapter selon votre configuration)
    DATABASE_PATH = 'ptbxl_database.csv'
    SCP_STATEMENTS_PATH = 'scp_statements.csv'
    
    # Création de l'explorateur et exécution de l'EDA complète
    explorer = PTBXLExplorer(DATABASE_PATH, SCP_STATEMENTS_PATH)
    explorer.run_complete_eda()
