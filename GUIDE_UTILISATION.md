# 📊 Guide d'Utilisation - Analyse PTB-XL

## 🎯 Comment Utiliser Cette Analyse

### Option 1: Exécution Automatique (Recommandé)
```bash
python PTB_XL_EDA_Simple.py
```

### Option 2: Avec Vérification des Prérequis
```bash
python run_eda.py
```

### Option 3: Version Avancée (Plus Détaillée)
```bash
python PTB_XL_EDA_Professional.py
```

---

## 📁 Structure des Fichiers

### Scripts d'Analyse
- `PTB_XL_EDA_Simple.py` ⭐ **Version optimisée** (recommandée)
- `PTB_XL_EDA_Professional.py` - Version complète avancée
- `run_eda.py` - Launcher avec vérifications

### Documentation
- `RESULTATS_ANALYSE.md` ⭐ **Résultats détaillés** (à lire en premier)
- `README_EDA.md` - Documentation technique
- `PTB_XL_EDA_Report.txt` - Rapport textuel

### Visualisations Générées
1. `EDA_01_Demographics.png` - Analyses démographiques
2. `EDA_02_Diagnostics.png` - Diagnostics et codes SCP  
3. `EDA_03_Temporal.png` - Évolutions temporelles
4. `EDA_04_Quality.png` - Qualité des données
5. `EDA_05_Technical.png` - Infrastructure technique

---

## 🔍 Que Contient Chaque Visualisation ?

### 📊 EDA_01_Demographics.png
- Distribution de l'âge (histogramme avec statistiques)
- Répartition par sexe (camembert)
- Distribution du poids
- Boxplot âge par sexe
- Distribution de la taille
- Calcul et distribution de l'IMC

### 🏥 EDA_02_Diagnostics.png
- Top 15 codes SCP les plus fréquents
- Nombre de codes par enregistrement
- Distribution par classe diagnostique
- Catégories de déclarations SCP

### 📅 EDA_03_Temporal.png
- Évolution du nombre d'enregistrements par année
- Distribution mensuelle
- Distribution par jour de la semaine

### 🎯 EDA_04_Quality.png
- Distribution du score de qualité (0-6)
- Taux de problèmes de qualité
- Top 10 valeurs manquantes

### 🔧 EDA_05_Technical.png
- Distribution par site d'enregistrement
- Distribution par appareil ECG
- Stratification des folds

---

## 📈 Résumé des Découvertes

### 🟢 Points Positifs
- ✅ **21,799 enregistrements** de haute qualité
- ✅ **Score qualité 5.64/6** - Excellent !
- ✅ **73.7% validés par humain** - Fiabilité garantie
- ✅ **71 diagnostics différents** - Grande diversité
- ✅ **Stratification intégrée** - Prêt pour le ML

### 🟡 Points d'Attention
- ⚠️ Height manquant: 68%
- ⚠️ Weight manquant: 57%
- ⚠️ Bruit statique: 15% des cas
- ⚠️ Distribution temporelle non uniforme

---

## 💡 Applications Pratiques

### Pour le Machine Learning
```python
# Le dataset est déjà stratifié en 10 folds
# Utiliser la colonne 'strat_fold' pour validation croisée

train_data = df[df['strat_fold'] != 10]
test_data = df[df['strat_fold'] == 10]
```

### Pour la Classification
**Classes principales identifiées**:
- NORM (Normal) - 43.6%
- MI (Infarctus) - 25.2%
- STTC (ST-T Changes) - 24.1%
- CD (Conduction) - 21.4%
- HYP (Hypertrophie) - 14.4%

### Pour le Deep Learning
- Signaux disponibles à **100 Hz** et **500 Hz**
- **12 dérivations** par enregistrement
- Données dans `records100/` et `records500/`

---

## 🎓 Interprétation des Résultats

### Score de Qualité
```
6/6 = Parfait (aucun problème)
5/6 = Très bon (1 problème mineur)
4/6 = Bon (2 problèmes)
3/6 = Acceptable
<3 = Qualité douteuse
```
**Moyenne du dataset: 5.64/6** ⭐

### Codes SCP
- **SR** = Sinus Rhythm (rythme normal)
- **NORM** = Normal ECG
- **IMI** = Inferior Myocardial Infarction
- **ASMI** = Anteroseptal MI
- **LVH** = Left Ventricular Hypertrophy

---

## 🚀 Prochaines Étapes Suggérées

### 1. Nettoyage Avancé
```python
# Imputation des valeurs manquantes
# Filtrage des enregistrements de faible qualité
# Gestion des outliers (ex: âge = 300 ans)
```

### 2. Feature Engineering
```python
# Extraction de features ECG:
# - Heart Rate Variability (HRV)
# - QT interval
# - P-wave duration
# - QRS complex morphology
```

### 3. Modélisation
```python
# Modèles suggérés:
# - Random Forest (baseline)
# - XGBoost (performance)
# - CNN 1D (signal brut)
# - LSTM (séquences temporelles)
```

---

## 📚 Ressources Complémentaires

### Documentation Dataset
- PhysioNet: https://physionet.org/content/ptb-xl/
- Paper: Wagner et al., 2020
- Codes SCP: Standard international

### Tutoriels Recommandés
1. Chargement des signaux avec `wfdb`
2. Prétraitement ECG (filtrage, normalisation)
3. Classification multi-classes avec CNN
4. Interprétabilité (Grad-CAM, SHAP)

---

## ⚡ Tips & Astuces

### Performance
```python
# Charger uniquement les métadonnées d'abord
df = pd.read_csv('ptbxl_database.csv')

# Charger les signaux seulement si nécessaire
# (fichiers volumineux: 100Hz = 6.6GB, 500Hz = 31GB)
```

### Validation
```python
# Utiliser les folds intégrés
for fold in range(1, 11):
    train = df[df['strat_fold'] != fold]
    val = df[df['strat_fold'] == fold]
    # Train model...
```

### Qualité
```python
# Filtrer les enregistrements de haute qualité
high_quality = df[df['quality_score'] >= 5]
validated = df[df['validated_by_human'] == True]
```

---

## 🆘 Besoin d'Aide ?

### Problèmes Courants

**1. Erreur d'import**
```bash
pip install pandas numpy matplotlib seaborn wfdb
```

**2. Fichiers CSV introuvables**
Assurez-vous d'être dans le bon répertoire contenant `ptbxl_database.csv`

**3. Manque de mémoire**
Utilisez `PTB_XL_EDA_Simple.py` au lieu de la version complète

**4. Graphiques ne s'affichent pas**
Les images PNG sont sauvegardées automatiquement dans le dossier courant

---

## 📊 Checklist de l'Analyse

- ✅ Dataset chargé et exploré
- ✅ Valeurs manquantes identifiées
- ✅ Distributions analysées
- ✅ Qualité évaluée
- ✅ Visualisations générées
- ✅ Rapport créé
- ⬜ Nettoyage des données (à faire)
- ⬜ Feature engineering (à faire)
- ⬜ Modélisation ML (à faire)

---

## 🎉 Félicitations !

Vous disposez maintenant d'une **analyse exploratoire complète et professionnelle** du dataset PTB-XL !

Les visualisations et rapports générés vous permettent de:
- ✅ Comprendre la structure des données
- ✅ Identifier les opportunités et limites
- ✅ Prendre des décisions éclairées pour le ML
- ✅ Communiquer efficacement les résultats

---

**Bon courage pour vos modèles ML ! 🚀📊🏥**

---

*Dernière mise à jour: 29 Décembre 2025*
