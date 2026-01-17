# 📊 Analyse Exploratoire de Données (EDA) - PTB-XL ECG Database

## 🎯 Objectif

Cette analyse exploratoire professionnelle du dataset PTB-XL fournit une vue complète et détaillée de la plus grande base de données d'électrocardiographie (ECG) accessible publiquement.

## 📁 Contenu du Projet

### Fichiers Principaux

- **`PTB_XL_EDA_Professional.py`** : Script Python complet d'analyse exploratoire
- **`PTB_XL_EDA_Notebook.ipynb`** : Notebook Jupyter interactif (optionnel)
- **`README.md`** : Ce fichier de documentation

### Données Source

- **`ptbxl_database.csv`** : Base de données principale (21,801 enregistrements ECG)
- **`scp_statements.csv`** : Dictionnaire des codes diagnostiques SCP
- **`records100/`** et **`records500/`** : Signaux ECG bruts (100Hz et 500Hz)

## 🔬 Analyses Réalisées

### 1. **Vue d'Ensemble du Dataset**
   - Dimensions et structure des données
   - Types de variables
   - Statistiques descriptives

### 2. **Analyse des Valeurs Manquantes**
   - Identification des variables avec données manquantes
   - Visualisation graphique (barres et heatmap)
   - Calcul des pourcentages

### 3. **Analyse Démographique**
   - Distribution de l'âge (histogramme, statistiques)
   - Répartition par sexe
   - Distribution du poids et de la taille
   - Calcul et analyse de l'IMC
   - Comparaisons par groupes

### 4. **Analyse des Diagnostics**
   - Fréquence des codes SCP
   - Top 20 des diagnostics les plus courants
   - Distribution par classe diagnostique
   - Nombre moyen de codes par enregistrement
   - Catégories de déclarations SCP

### 5. **Analyse Temporelle**
   - Évolution du nombre d'enregistrements par année
   - Distribution mensuelle
   - Distribution par jour de la semaine
   - Tendances temporelles

### 6. **Analyse Technique**
   - Distribution par site d'enregistrement
   - Appareils utilisés
   - Infrastructure (infirmières, équipements)
   - Stratification des folds pour validation croisée

### 7. **Évaluation de la Qualité**
   - Score de qualité calculé (0-6)
   - Problèmes de signal identifiés:
     - Baseline drift
     - Static noise
     - Burst noise
     - Problèmes d'électrodes
     - Extra beats
     - Présence de pacemaker
   - Taux de validation humaine
   - Corrélations entre problèmes de qualité

### 8. **Analyse des Corrélations**
   - Matrice de corrélation des variables numériques
   - Relations entre variables clés
   - Visualisations scatter plots

### 9. **Rapport Résumé**
   - Synthèse complète de toutes les analyses
   - Statistiques clés
   - Recommandations

## 📊 Graphiques Générés

L'analyse génère automatiquement 7 visualisations haute résolution (300 DPI) :

1. **`01_missing_values_analysis.png`** - Analyse des valeurs manquantes
2. **`02_demographic_analysis.png`** - Analyses démographiques complètes
3. **`03_diagnostic_analysis.png`** - Distribution des diagnostics
4. **`04_temporal_analysis.png`** - Évolutions temporelles
5. **`05_technical_analysis.png`** - Aspects techniques et infrastructure
6. **`06_quality_assessment.png`** - Évaluation de la qualité
7. **`07_correlation_analysis.png`** - Corrélations entre variables

Plus un rapport texte détaillé :
- **`PTB_XL_EDA_Summary_Report.txt`** - Rapport résumé complet

## 🚀 Installation et Utilisation

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn wfdb
```

### Exécution Rapide

```bash
python PTB_XL_EDA_Professional.py
```

### Configuration

Modifiez les chemins dans le fichier principal si nécessaire :

```python
DATABASE_PATH = 'ptbxl_database.csv'
SCP_STATEMENTS_PATH = 'scp_statements.csv'
```

## 📈 Résultats Clés

### Statistiques Générales
- **21,801** enregistrements ECG
- **18,885** patients uniques
- Période : **1984 à 1996**
- **12 dérivations** par enregistrement

### Démographie
- Âge moyen : **~57 ans**
- Ratio Homme/Femme : **~1.2:1**
- Large distribution d'âge (0-95 ans)

### Qualité
- Score de qualité moyen : **~5.5/6**
- **>95%** validés par un humain
- Excellente qualité globale du dataset

### Diagnostics
- **73 codes SCP uniques**
- **5 classes diagnostiques principales** :
  - NORM (Normal)
  - MI (Myocardial Infarction)
  - STTC (ST-T Changes)
  - CD (Conduction Disturbances)
  - HYP (Hypertrophy)

## 💡 Insights et Recommandations

### Points Forts ✅
- Dataset de grande taille et bien structuré
- Excellente qualité des données
- Validation humaine extensive
- Stratification intégrée pour ML/DL
- Multi-lead ECG (12 dérivations)

### Points d'Attention ⚠️
- Valeurs manquantes pour height/weight
- Distribution déséquilibrée des classes diagnostiques
- Quelques problèmes de qualité du signal

### Applications Possibles 🎯
1. **Classification automatique** des ECG par deep learning
2. **Détection d'anomalies** cardiovasculaires
3. **Analyse de séries temporelles** médicales
4. **Benchmarking** d'algorithmes de traitement du signal
5. **Recherche clinique** sur les pathologies cardiaques

## 📚 Références

- **Dataset Original** : [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/)
- **Publication** : Wagner et al. (2020), "PTB-XL, a large publicly available electrocardiography dataset"
- **License** : Open Database License (ODC-ODbL)

## 🔧 Structure de la Classe

```python
class PTBXLExplorer:
    def __init__(database_path, scp_statements_path)
    def overview()
    def missing_values_analysis()
    def demographic_analysis()
    def diagnostic_analysis()
    def temporal_analysis()
    def technical_analysis()
    def quality_assessment()
    def correlation_analysis()
    def generate_summary_report()
    def run_complete_eda()
```

## 📞 Contact & Support

Pour toute question ou suggestion d'amélioration, n'hésitez pas à ouvrir une issue ou contribuer au projet.

---

**Date de création** : Décembre 2025  
**Version** : 1.0  
**Auteur** : Data Science Professional  
**License** : MIT

---

## 🌟 Features Avancées

- ✅ Analyse complète et automatisée
- ✅ Visualisations professionnelles haute résolution
- ✅ Rapport détaillé au format texte
- ✅ Code modulaire et réutilisable
- ✅ Gestion des erreurs et valeurs manquantes
- ✅ Documentation complète
- ✅ Style de code PEP 8
- ✅ Commentaires détaillés

## 🎨 Personnalisation

Le code est facilement personnalisable :
- Modifier les couleurs des graphiques
- Ajouter de nouvelles analyses
- Changer les seuils de qualité
- Adapter les visualisations

## 📊 Exemples de Sorties

Le script affiche dans la console :
- Statistiques détaillées
- Tableaux formatés
- Indicateurs de progression
- Messages de confirmation

Et génère des fichiers :
- Images PNG haute résolution
- Rapport texte structuré
- Données exportables

---

**Bonne Analyse ! 🚀📊**
