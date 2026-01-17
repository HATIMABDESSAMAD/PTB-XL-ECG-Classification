"""
═══════════════════════════════════════════════════════════════════════════════
Script de Lancement Rapide - EDA PTB-XL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

def check_requirements():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'wfdb']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Packages manquants détectés:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Installation automatique...")
        os.system(f"pip install {' '.join(missing_packages)}")
        print("✅ Installation terminée!\n")
    else:
        print("✅ Toutes les dépendances sont installées!\n")

def check_data_files():
    """Vérifie que les fichiers de données sont présents"""
    required_files = ['ptbxl_database.csv', 'scp_statements.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Fichiers de données manquants:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n⚠️  Assurez-vous que les fichiers CSV sont dans le même répertoire.")
        return False
    else:
        print("✅ Tous les fichiers de données sont présents!\n")
        return True

def main():
    """Fonction principale"""
    print("═" * 80)
    print("  LANCEMENT DE L'ANALYSE EXPLORATOIRE PTB-XL")
    print("═" * 80 + "\n")
    
    # Vérification des prérequis
    print("🔍 Vérification des prérequis...\n")
    check_requirements()
    
    if not check_data_files():
        print("\n❌ Impossible de continuer sans les fichiers de données.")
        print("📥 Téléchargez le dataset depuis: https://physionet.org/content/ptb-xl/")
        sys.exit(1)
    
    # Import et exécution
    print("🚀 Lancement de l'analyse...\n")
    
    try:
        from PTB_XL_EDA_Professional import PTBXLExplorer
        
        # Chemins des fichiers
        DATABASE_PATH = 'ptbxl_database.csv'
        SCP_STATEMENTS_PATH = 'scp_statements.csv'
        
        # Création et exécution
        explorer = PTBXLExplorer(DATABASE_PATH, SCP_STATEMENTS_PATH)
        explorer.run_complete_eda()
        
        print("\n" + "═" * 80)
        print("  ✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("═" * 80)
        print("\n📁 Fichiers générés:")
        print("   • 01_missing_values_analysis.png")
        print("   • 02_demographic_analysis.png")
        print("   • 03_diagnostic_analysis.png")
        print("   • 04_temporal_analysis.png")
        print("   • 05_technical_analysis.png")
        print("   • 06_quality_assessment.png")
        print("   • 07_correlation_analysis.png")
        print("   • PTB_XL_EDA_Summary_Report.txt")
        print("\n🎉 Consultez les graphiques et le rapport pour les résultats!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {str(e)}")
        print("\n📝 Détails de l'erreur:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
