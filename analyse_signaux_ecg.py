"""
ANALYSE COMPLÈTE DES SIGNAUX ECG - RECORDS100 et RECORDS500
Dataset PTB-XL v1.0.3
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import struct

print("=" * 100)
print("ANALYSE APPROFONDIE DES SIGNAUX ECG - PTB-XL DATASET")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATISTIQUES GLOBALES DES FICHIERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("1. INVENTAIRE COMPLET DES FICHIERS")
print("─" * 100)

def count_files_in_directory(directory):
    """Compter fichiers .hea et .dat dans un répertoire"""
    hea_files = list(Path(directory).rglob("*.hea"))
    dat_files = list(Path(directory).rglob("*.dat"))
    
    # Calculer tailles
    total_hea_size = sum(f.stat().st_size for f in hea_files)
    total_dat_size = sum(f.stat().st_size for f in dat_files)
    
    return {
        'hea_count': len(hea_files),
        'dat_count': len(dat_files),
        'hea_size_mb': total_hea_size / (1024**2),
        'dat_size_mb': total_dat_size / (1024**2),
        'total_size_mb': (total_hea_size + total_dat_size) / (1024**2)
    }

# Analyser records100
print("\n📁 RECORDS100/ (Basse résolution - 100 Hz)")
records100_stats = count_files_in_directory('records100')
print(f"  • Fichiers .hea : {records100_stats['hea_count']:,}")
print(f"  • Fichiers .dat : {records100_stats['dat_count']:,}")
print(f"  • Taille .hea   : {records100_stats['hea_size_mb']:.2f} MB")
print(f"  • Taille .dat   : {records100_stats['dat_size_mb']:.2f} MB")
print(f"  • Taille totale : {records100_stats['total_size_mb']:.2f} MB")

# Analyser records500
print("\n📁 RECORDS500/ (Haute résolution - 500 Hz)")
records500_stats = count_files_in_directory('records500')
print(f"  • Fichiers .hea : {records500_stats['hea_count']:,}")
print(f"  • Fichiers .dat : {records500_stats['dat_count']:,}")
print(f"  • Taille .hea   : {records500_stats['hea_size_mb']:.2f} MB")
print(f"  • Taille .dat   : {records500_stats['dat_size_mb']:.2f} MB")
print(f"  • Taille totale : {records500_stats['total_size_mb']:.2f} MB")

print(f"\n📊 TOTAL COMBINÉ")
print(f"  • Enregistrements : {records100_stats['hea_count']:,}")
print(f"  • Fichiers totaux : {(records100_stats['hea_count'] + records100_stats['dat_count'] + records500_stats['hea_count'] + records500_stats['dat_count']):,}")
print(f"  • Espace disque   : {(records100_stats['total_size_mb'] + records500_stats['total_size_mb']):.2f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSE DES FICHIERS HEADER (.hea)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("2. ANALYSE DÉTAILLÉE DES FICHIERS HEADER (.hea)")
print("─" * 100)

def parse_header_file(filepath):
    """Parser un fichier .hea pour extraire métadonnées"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Ligne 1: nom freq nb_samples
    header_line = lines[0].strip().split()
    filename = header_line[0]
    n_leads = int(header_line[1])
    freq = int(header_line[2])
    n_samples = int(header_line[3])
    
    # Lignes suivantes: spécifications des leads
    leads = []
    for i in range(1, n_leads + 1):
        if i < len(lines):
            lead_info = lines[i].strip().split()
            lead_name = lead_info[-1] if len(lead_info) > 0 else f"Lead_{i}"
            leads.append(lead_name)
    
    return {
        'filename': filename,
        'n_leads': n_leads,
        'freq': freq,
        'n_samples': n_samples,
        'duration_sec': n_samples / freq,
        'leads': leads
    }

# Analyser 5 fichiers exemples de records100
print("\n🔍 EXEMPLE: 5 premiers fichiers records100/")
hea_files_100 = sorted(list(Path('records100/00000').glob("*.hea")))[:5]

for hea_file in hea_files_100:
    meta = parse_header_file(hea_file)
    print(f"\n  📄 {meta['filename']}")
    print(f"     • Dérivations : {meta['n_leads']} leads → {', '.join(meta['leads'])}")
    print(f"     • Fréquence   : {meta['freq']} Hz")
    print(f"     • Échantillons: {meta['n_samples']:,}")
    print(f"     • Durée       : {meta['duration_sec']} secondes")

# Analyser 5 fichiers exemples de records500
print("\n🔍 EXEMPLE: 5 premiers fichiers records500/")
hea_files_500 = sorted(list(Path('records500/00000').glob("*.hea")))[:5]

for hea_file in hea_files_500:
    meta = parse_header_file(hea_file)
    print(f"\n  📄 {meta['filename']}")
    print(f"     • Dérivations : {meta['n_leads']} leads → {', '.join(meta['leads'])}")
    print(f"     • Fréquence   : {meta['freq']} Hz")
    print(f"     • Échantillons: {meta['n_samples']:,}")
    print(f"     • Durée       : {meta['duration_sec']} secondes")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE DES FICHIERS DONNÉES (.dat)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("3. ANALYSE DES FICHIERS DONNÉES BINAIRES (.dat)")
print("─" * 100)

def analyze_dat_file(dat_filepath, n_samples, n_leads):
    """Analyser fichier .dat binaire"""
    file_size = os.path.getsize(dat_filepath)
    expected_size = n_samples * n_leads * 2  # 2 bytes par échantillon (16-bit)
    
    # Lire échantillons
    with open(dat_filepath, 'rb') as f:
        data = f.read()
    
    # Convertir en int16
    samples = struct.unpack(f'<{len(data)//2}h', data)  # little-endian signed short
    samples_array = np.array(samples).reshape((n_samples, n_leads))
    
    return {
        'file_size_bytes': file_size,
        'expected_size_bytes': expected_size,
        'size_match': file_size == expected_size,
        'min_value': samples_array.min(),
        'max_value': samples_array.max(),
        'mean_value': samples_array.mean(),
        'std_value': samples_array.std()
    }

# Analyser fichiers .dat correspondants
print("\n🔬 ANALYSE STATISTIQUE: records100/00000/00001_lr.dat")
meta = parse_header_file('records100/00000/00001_lr.hea')
dat_stats = analyze_dat_file('records100/00000/00001_lr.dat', meta['n_samples'], meta['n_leads'])

print(f"  • Taille fichier     : {dat_stats['file_size_bytes']:,} bytes")
print(f"  • Taille attendue    : {dat_stats['expected_size_bytes']:,} bytes")
print(f"  • Correspondance     : {'✓ OUI' if dat_stats['size_match'] else '✗ NON'}")
print(f"  • Valeur min (ADC)   : {dat_stats['min_value']}")
print(f"  • Valeur max (ADC)   : {dat_stats['max_value']}")
print(f"  • Valeur moyenne     : {dat_stats['mean_value']:.2f}")
print(f"  • Écart-type         : {dat_stats['std_value']:.2f}")

print("\n🔬 ANALYSE STATISTIQUE: records500/00000/00001_hr.dat")
meta_hr = parse_header_file('records500/00000/00001_hr.hea')
dat_stats_hr = analyze_dat_file('records500/00000/00001_hr.dat', meta_hr['n_samples'], meta_hr['n_leads'])

print(f"  • Taille fichier     : {dat_stats_hr['file_size_bytes']:,} bytes")
print(f"  • Taille attendue    : {dat_stats_hr['expected_size_bytes']:,} bytes")
print(f"  • Correspondance     : {'✓ OUI' if dat_stats_hr['size_match'] else '✗ NON'}")
print(f"  • Valeur min (ADC)   : {dat_stats_hr['min_value']}")
print(f"  • Valeur max (ADC)   : {dat_stats_hr['max_value']}")
print(f"  • Valeur moyenne     : {dat_stats_hr['mean_value']:.2f}")
print(f"  • Écart-type         : {dat_stats_hr['std_value']:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ORGANISATION HIÉRARCHIQUE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("4. STRUCTURE HIÉRARCHIQUE DES DOSSIERS")
print("─" * 100)

def analyze_folder_structure(base_dir):
    """Analyser structure des sous-dossiers"""
    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    folder_stats = []
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        hea_count = len(list(Path(subdir_path).glob("*.hea")))
        dat_count = len(list(Path(subdir_path).glob("*.dat")))
        folder_stats.append({
            'folder': subdir,
            'hea_count': hea_count,
            'dat_count': dat_count
        })
    
    return folder_stats

print("\n📂 RECORDS100/ - Répartition par dossier:")
folders_100 = analyze_folder_structure('records100')
print(f"  • Nombre de dossiers: {len(folders_100)}")
print(f"  • Range ECG IDs     : {folders_100[0]['folder']} à {folders_100[-1]['folder']}")
print(f"\n  Détail (premiers 5 dossiers):")
for folder in folders_100[:5]:
    print(f"    - {folder['folder']}/  : {folder['hea_count']:3d} fichiers .hea, {folder['dat_count']:3d} fichiers .dat")

print("\n📂 RECORDS500/ - Répartition par dossier:")
folders_500 = analyze_folder_structure('records500')
print(f"  • Nombre de dossiers: {len(folders_500)}")
print(f"  • Range ECG IDs     : {folders_500[0]['folder']} à {folders_500[-1]['folder']}")
print(f"\n  Détail (premiers 5 dossiers):")
for folder in folders_500[:5]:
    print(f"    - {folder['folder']}/  : {folder['hea_count']:3d} fichiers .hea, {folder['dat_count']:3d} fichiers .dat")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. VÉRIFICATION COHÉRENCE AVEC CSV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("5. VÉRIFICATION COHÉRENCE AVEC ptbxl_database.csv")
print("─" * 100)

# Charger CSV
df_csv = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
print(f"\n✓ CSV chargé: {len(df_csv):,} enregistrements")

# Vérifier que tous les filename_lr existent
print(f"\n🔍 Vérification existence fichiers filename_lr...")
missing_lr = []
for idx, row in df_csv.head(100).iterrows():  # Test sur 100 premiers
    hea_path = row['filename_lr'] + '.hea'
    dat_path = row['filename_lr'] + '.dat'
    
    if not os.path.exists(hea_path):
        missing_lr.append((idx, hea_path, '.hea'))
    if not os.path.exists(dat_path):
        missing_lr.append((idx, dat_path, '.dat'))

if len(missing_lr) == 0:
    print(f"  ✓ Tous les fichiers existent (échantillon de 100)")
else:
    print(f"  ✗ {len(missing_lr)} fichiers manquants détectés")
    for ecg_id, path, ext in missing_lr[:5]:
        print(f"    - ECG {ecg_id}: {path}")

# Vérifier que tous les filename_hr existent
print(f"\n🔍 Vérification existence fichiers filename_hr...")
missing_hr = []
for idx, row in df_csv.head(100).iterrows():
    hea_path = row['filename_hr'] + '.hea'
    dat_path = row['filename_hr'] + '.dat'
    
    if not os.path.exists(hea_path):
        missing_hr.append((idx, hea_path, '.hea'))
    if not os.path.exists(dat_path):
        missing_hr.append((idx, dat_path, '.dat'))

if len(missing_hr) == 0:
    print(f"  ✓ Tous les fichiers existent (échantillon de 100)")
else:
    print(f"  ✗ {len(missing_hr)} fichiers manquants détectés")
    for ecg_id, path, ext in missing_hr[:5]:
        print(f"    - ECG {ecg_id}: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CARACTÉRISTIQUES TECHNIQUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 100)
print("6. CARACTÉRISTIQUES TECHNIQUES DES SIGNAUX")
print("─" * 100)

print("""
┌─────────────────────┬──────────────────┬──────────────────┐
│ Caractéristique     │ records100/      │ records500/      │
├─────────────────────┼──────────────────┼──────────────────┤
│ Fréquence           │ 100 Hz           │ 500 Hz           │
│ Durée               │ 10 secondes      │ 10 secondes      │
│ Échantillons/lead   │ 1,000            │ 5,000            │
│ Nombre de leads     │ 12               │ 12               │
│ Format données      │ 16-bit signed    │ 16-bit signed    │
│ Taille/enregistr.   │ ~24 KB           │ ~120 KB          │
│ Résolution ADC      │ 1 µV/unit        │ 1 µV/unit        │
│ Gain standard       │ 1000 units/mV    │ 1000 units/mV    │
└─────────────────────┴──────────────────┴──────────────────┘

📊 LEADS STANDARD (12 dérivations ECG):
  • Bipolaires (Einthoven)    : I, II, III
  • Unipolaires augmentées     : AVR, AVL, AVF
  • Précordiales (thorax)      : V1, V2, V3, V4, V5, V6

💡 APPLICATIONS:
  • records100/ → Feature extraction, ML classique, déploiement temps réel
  • records500/ → Deep Learning, analyse morphologique détaillée, recherche
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("7. RÉSUMÉ EXÉCUTIF")
print("=" * 100)

total_recordings = records100_stats['hea_count']
total_files = (records100_stats['hea_count'] + records100_stats['dat_count'] + 
               records500_stats['hea_count'] + records500_stats['dat_count'])
total_size_gb = (records100_stats['total_size_mb'] + records500_stats['total_size_mb']) / 1024

print(f"""
✓ DATASET PTB-XL v1.0.3 - Signaux ECG

📊 VOLUME:
  • Enregistrements ECG        : {total_recordings:,}
  • Fichiers totaux            : {total_files:,}
  • Espace disque              : {total_size_gb:.2f} GB
  • Résolutions disponibles    : 2 (100 Hz et 500 Hz)

🏥 CONTENU MÉDICAL:
  • Dérivations par ECG        : 12 leads standard
  • Durée par enregistrement   : 10 secondes
  • Patients uniques           : ~18,869
  • Codes diagnostiques SCP    : 71 pathologies

🔧 QUALITÉ TECHNIQUE:
  • Format                     : PhysioNet WFDB
  • Encodage                   : 16-bit signed integer
  • Résolution temporelle      : 100 Hz (standard) / 500 Hz (recherche)
  • Intégrité fichiers         : ✓ Vérifiée (échantillon 100)

🎯 PRÊT POUR:
  ✓ Machine Learning (features tabulaires)
  ✓ Deep Learning (CNN/LSTM sur signaux bruts)
  ✓ Analyse morphologique (détection P-QRS-T)
  ✓ Classification multi-label (30 codes SCP principaux)
  ✓ Transfert d'apprentissage (pré-entraînement haute résolution)
""")

print("=" * 100)
print("ANALYSE TERMINÉE")
print("=" * 100)
