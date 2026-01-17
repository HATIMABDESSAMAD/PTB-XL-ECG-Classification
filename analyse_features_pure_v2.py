import numpy as np
import pandas as pd

# Charger données
df_excel = pd.read_csv('ptbxl_from_excel_consolidated.csv')

# Colonnes EXCLUES (selon step4d)
EXCLUDE_COLS = [
    'ecg_id', 'patient_id', 'filename_lr', 'filename_hr',
    'report', 'validated_by', 'nurse', 'site', 'device',
    'recording_date', 'strat_fold'
] + [f'scp_{x}' for x in ['SR', 'NORM', 'ABQRS', 'IMI', 'ASMI', 'LVH', 'NDT', 'LAFB', 
                          'AFIB', 'ISC_', 'PVC', 'IRBBB', 'STD_', 'VCLVH', 'STACH', 
                          'IVCD', '1AVB', 'SARRH', 'NST_', 'ISCAL', 'SBRAD', 'CRBBB', 
                          'QWAVE', 'CLBBB', 'ILMI', 'LOWT', 'LAO/LAE', 'NT_', 'PAC', 'AMI']] \
  + [f'scp_superclass_{x}' for x in ['NORM', 'MI', 'STTC', 'CD', 'HYP']]

# Features Excel GARDÉES
excel_cols = [col for col in df_excel.columns if col not in EXCLUDE_COLS]

print("="*80)
print("🏆 ANALYSE COMPLÈTE DU MODÈLE PURE (94.29% AUC MACRO)")
print("="*80)

print(f"\n📋 NOMBRE DE FEATURES: {len(excel_cols)} features Excel")
print("\nLISTE COMPLÈTE DES 32 FEATURES EXCEL:")
print("-"*80)

# Grouper par catégorie
categories = {
    '📊 DÉMOGRAPHIQUES': ['age', 'sex', 'height', 'weight', 'bmi'],
    '📅 TEMPORELLES': ['year', 'month', 'quarter', 'day_of_week'],
    '✅ QUALITÉ SIGNAL': ['quality_score', 'quality_issues_count', 'has_quality_issues', 
                          'baseline_drift', 'static_noise', 'extra_beats'],
    '🔧 MÉTADONNÉES': ['is_validated', 'has_second_opinion', 'num_scp_codes'],
    '🏥 ENCODAGES': ['heart_axis_encoded', 'site_encoded', 'device_encoded'],
    '📂 CATÉGORIES DÉRIVÉES': ['age_group_<18', 'age_group_18-35', 'age_group_35-50',
                                'age_group_50-65', 'age_group_65-80', 'age_group_80+',
                                'bmi_cat_Underweight', 'bmi_cat_Normal', 'bmi_cat_Overweight',
                                'bmi_cat_Obese', 'bmi_cat_Severe Obese'],
    '🔀 AUTRES': ['split']
}

idx = 1
for cat_name, cat_cols in categories.items():
    cols_present = [c for c in cat_cols if c in excel_cols]
    if cols_present:
        print(f"\n{cat_name} ({len(cols_present)} features):")
        for col in cols_present:
            print(f"  {idx:2d}. {col}")
            idx += 1

print("\n" + "="*80)
print(f"✅ TOTAL VÉRIFIÉ: {len(excel_cols)} features")
print("="*80)
