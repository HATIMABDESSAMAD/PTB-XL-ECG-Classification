# PTB-XL Wide+Deep Pipeline - Installation & Exécution

## 📋 Vue d'ensemble

Pipeline complet pour classification ECG multi-label sur PTB-XL avec architecture **Wide+Deep**:
- **Deep Branch**: CNN1D + Transformer sur signaux 12 leads
- **Wide Branch**: MLP sur features cliniques (NeuroKit2) + metadata
- **Fusion**: Concaténation → FC Head → 71 codes SCP ou 5 superclasses

## 🚀 Installation rapide

```bash
# Dépendances principales
pip install pandas numpy scikit-learn
pip install wfdb neurokit2
pip install torch torchvision
pip install xgboost
pip install tqdm
```

## 📂 Structure du pipeline

```
PTB-XL Pipeline (7 steps)
│
├── STEP 1: Label Engineering
│   ├── Input : ptbxl_database.csv, scp_statements.csv
│   ├── Script: step1_label_engineering.py
│   └── Output: ptbxl_with_labels_expanded.csv (y__<CODE>, y_SUP__<CLASS>)
│
├── STEP 2: Signal Cleaning (NeuroKit2)
│   ├── Input : records100/ (WFDB)
│   ├── Script: step2_signal_cleaning.py
│   └── Output: cleaned_signals_100hz/*.npz (12×1000)
│
├── STEP 3: Wide Features Extraction
│   ├── Input : cleaned_signals_100hz/
│   ├── Script: step3_wide_features_extraction.py
│   └── Output: ptbxl_wide_features.csv (~42 features)
│
├── STEP 4: Wide Preprocessing
│   ├── Input : ptbxl_wide_features.csv
│   ├── Script: step4_wide_preprocessing.py
│   └── Output: preprocessed_wide/W_*.npz (Train/Val/Test)
│
├── STEP 5: Architecture PyTorch
│   ├── Script: step5_wide_deep_model.py (test)
│   └── Classes: WideDeepModel, DeepOnlyModel, WideOnlyModel
│
├── STEP 6: Training
│   ├── Script: step6_training.py (config éditable)
│   └── Output: models/best_model.pth + results/
│
└── STEP 7: Baselines & Comparaison
    ├── Script: step7_baselines.py
    └── Output: Tableau comparatif + analyse qualité
```

## ⚡ Exécution séquentielle

### STEP 1: Label Engineering (~1 minute)
```bash
python step1_label_engineering.py
```
**Sortie**: `ptbxl_with_labels_expanded.csv` avec colonnes:
- `y__<CODE>`: 71 labels binaires (ex: `y__NORM`, `y__MI`)
- `y_score__<CODE>`: scores originaux 0-100
- `y_SUP__<CLASS>`: 5 superclasses (NORM/MI/STTC/CD/HYP)

### STEP 2: Signal Cleaning (~20-30 minutes)
```bash
python step2_signal_cleaning.py
```
**Traitement**:
- Chargement WFDB (21,799 ECG)
- FIR bandpass 3-45 Hz par lead
- Z-score normalization
- Sauvegarde .npz compressé

**Sortie**: `cleaned_signals_100hz/X_clean_*.npz` (~250 MB total)

**⚠️ MODE TEST**: Pour test rapide, éditer ligne 77:
```python
SAMPLE_SIZE = 100  # Test sur 100 ECG seulement
```

### STEP 3: Wide Features Extraction (~10-15 minutes)
```bash
python step3_wide_features_extraction.py
```
**Features extraites (Lead II)**:
- R-peaks, HR, HRV (time domain)
- Intervalles P-QRS-T
- Entropies (sample, approximate)
- Qualité (`rpeaks_ok`, `delineation_ok`)
- Metadata (age, sex, device, etc.)

**Sortie**: `ptbxl_wide_features.csv` (~42 colonnes)

### STEP 4: Wide Preprocessing (~1 minute)
```bash
python step4_wide_preprocessing.py
```
**Preprocessing** (fit sur Train uniquement):
- Imputation: médiane (num), "Unknown" (cat)
- Encodage: one-hot (device/site/nurse), label (heart_axis)
- Scaling: z-score sur numériques

**Sortie**: `preprocessed_wide/W_train.npz`, `W_val.npz`, `W_test.npz`

### STEP 5: Test Architecture (~10 secondes)
```bash
python step5_wide_deep_model.py
```
**Vérification**: Forward pass OK, comptage paramètres

### STEP 6: Training (3 configurations)

#### Configuration A: Deep Only
```python
# Éditer step6_training.py:
class Config:
    model_type = 'deep_only'
    task_mode = '5superclass'  # ou '71codes'
    batch_size = 32
    num_epochs = 50
```
```bash
python step6_training.py
```

#### Configuration B: Wide Only (XGBoost)
```bash
python step7_baselines.py  # Entraîne XGBoost automatiquement
```

#### Configuration C: Wide+Deep ⭐ (RECOMMANDÉ)
```python
# Éditer step6_training.py:
class Config:
    model_type = 'wide_deep'
    task_mode = '5superclass'
```
```bash
python step6_training.py
```

**Durée**: 2-5 heures (GPU recommandé)

**Early stopping**: Patience 10 epochs sur Val AUC

### STEP 7: Comparaison & Analyse
```bash
python step7_baselines.py
```
**Résultats**:
- Tableau comparatif 3 baselines
- Analyse effet qualité signal (`RPeaks_ok`)

## 📊 Résultats attendus (CinC 2020)

| Modèle      | 5 Superclasses | 71 Codes SCP |
|-------------|----------------|--------------|
| Deep Only   | 0.85-0.88      | 0.78-0.82    |
| Wide Only   | 0.75-0.80      | 0.65-0.70    |
| **Wide+Deep** | **0.88-0.92** ⭐ | **0.80-0.85** ⭐ |

*AUC macro sur Test set*

## 🎯 Choix de la tâche

### 5 Superclasses (recommandé pour débuter)
- **NORM**: ECG normal
- **MI**: Myocardial Infarction
- **STTC**: ST/T Change
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

**Avantages**: Moins de déséquilibre, entraînement plus rapide

### 71 Codes SCP (avancé)
- Tous les codes diagnostiques PTB-XL
- Multi-label (plusieurs codes par ECG)

**Avantages**: Granularité fine, proche diagnostic clinique

## 🔧 Configuration GPU/CPU

```python
# step6_training.py
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Si GPU limité:
    batch_size = 16  # au lieu de 32
```

**Temps CPU vs GPU (epoch)**:
- CPU: ~45 minutes/epoch
- GPU (RTX 3080): ~5 minutes/epoch

## 📈 Monitoring training

```python
# Dans step6_training.py, epoch loop affiche:
Epoch 15/50 | Train Loss: 0.1234 | Val Loss: 0.1456 | Val AUC: 0.8765
  → Meilleur modèle sauvegardé (AUC: 0.8765)
```

## 🐛 Dépannage

### Erreur: "neurokit2 not found"
```bash
pip install neurokit2
```

### Erreur: "torch not found"
```bash
# CPU
pip install torch torchvision

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erreur: "Out of Memory" (GPU)
```python
# Réduire batch_size dans Config
batch_size = 16  # ou 8
```

### Signal cleaning trop lent
```python
# Mode test dans step2_signal_cleaning.py ligne 77
SAMPLE_SIZE = 100  # Test rapide
```

## 📦 Outputs finaux

```
ptb-xl-dataset/
├── ptbxl_with_labels_expanded.csv      (labels multi-label)
├── label_config.json                    (config pour modèle)
├── cleaned_signals_100hz/               (21,799 .npz ~250 MB)
├── ptbxl_wide_features.csv             (features tabulaires)
├── preprocessed_wide/                   (W_train/val/test.npz)
├── models/
│   └── best_model.pth                   (meilleur checkpoint)
└── results/
    └── results.json                     (métriques test)
```

## 📚 Références

1. **PTB-XL Dataset**: Wagner et al. (2020), Scientific Data
2. **CinC Challenge 2020**: Classification of 12-lead ECGs
3. **NeuroKit2**: Makowski et al. (2021)
4. **Wide & Deep**: Cheng et al. (2016), Google

## 🎓 Citation

```bibtex
@article{wagner2020ptb,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={154},
  year={2020}
}
```

## ✅ Checklist complète

- [ ] Step 1: Label Engineering exécuté
- [ ] Step 2: Signal Cleaning exécuté (21,799 .npz créés)
- [ ] Step 3: Wide Features Extraction exécuté
- [ ] Step 4: Wide Preprocessing exécuté
- [ ] Step 5: Architecture testée (forward pass OK)
- [ ] Step 6A: Deep Only entraîné
- [ ] Step 6B: Wide Only (XGBoost) entraîné
- [ ] Step 6C: Wide+Deep entraîné ⭐
- [ ] Step 7: Comparaison & analyse effectuée
- [ ] Résultats sauvegardés dans results/

## 🚀 Quick Start (résumé)

```bash
# 1. Labels
python step1_label_engineering.py

# 2-4. Preprocessing (30-40 min total)
python step2_signal_cleaning.py
python step3_wide_features_extraction.py
python step4_wide_preprocessing.py

# 5. Test architecture
python step5_wide_deep_model.py

# 6. Training Wide+Deep (éditer Config avant)
python step6_training.py

# 7. Comparaison
python step7_baselines.py
```

**Durée totale**: 3-6 heures (selon CPU/GPU)

---

**Questions?** Consultez `step7_baselines.py` pour le guide complet d'exécution.
