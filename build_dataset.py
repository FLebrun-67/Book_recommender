import sys
import time
import json
import os
import argparse
#from typing import Dict, List

# Importer votre classe améliorée
# (Assurez-vous que dataset_builder_improved.py soit dans le même dossier)
try:
    from dataset_builder import ImprovedDatasetBuilder
    print("✅ Module dataset_builder_improved importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Assurez-vous que dataset_builder_improved.py est dans le même dossier")
    sys.exit(1)

class ProgressTracker:
    """Classe pour suivre le progrès sans Streamlit"""
    
    def __init__(self, total_books: int):
        self.total_books = total_books
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, percentage: float, message: str):
        """Met à jour le progrès"""
        current_time = time.time()
        
        # Afficher seulement toutes les 5 secondes pour éviter le spam
        if current_time - self.last_update >= 5:
            elapsed = current_time - self.start_time
            
            if percentage > 0:
                estimated_total = elapsed / percentage
                remaining = estimated_total - elapsed
                
                print(f"\n📊 PROGRÈS: {percentage:.1%}")
                print(f"   📚 {message}")
                print(f"   ⏱️  Écoulé: {self._format_time(elapsed)}")
                print(f"   ⏳ Estimé restant: {self._format_time(remaining)}")
                print(f"   🎯 ETA: {self._format_time(estimated_total)}")
            else:
                print(f"\n🚀 DÉMARRAGE: {message}")
            
            self.last_update = current_time
    
    def _format_time(self, seconds: float) -> str:
        """Formate le temps en format lisible"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Construit un dataset de livres de haute qualité",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/enhanced_dataset_with_descriptions.json',
        help='Fichier de sortie pour le dataset'
    )
    
    parser.add_argument(
        '--no-descriptions',
        action='store_true',
        help='Désactiver la récupération des descriptions (plus rapide)'
    )
    
    parser.add_argument(
        '--genres',
        nargs='+',
        choices=['Fantasy', 'Science Fiction', 'Mystery', 'Romance', 'Historical Fiction', 'Thriller', 'Adventure', 'Horror'],
        help='Genres spécifiques à construire (par défaut: tous)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        help='Nombre total de livres cibles (remplace les objectifs par genre)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Mode test rapide (50 livres par genre)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Affichage détaillé'
    )
    
    return parser.parse_args()

def configure_builder(args) -> ImprovedDatasetBuilder:
    """Configure le builder selon les arguments"""
    
    # Créer le builder
    enable_descriptions = not args.no_descriptions
    builder = ImprovedDatasetBuilder(enable_descriptions=enable_descriptions)
    
    # Modifier les objectifs selon les arguments
    if args.quick_test:
        print("🧪 MODE TEST RAPIDE activé")
        builder.target_genres = {genre: 50 for genre in builder.target_genres.keys()}
    
    elif args.target_size:
        total_genres = len(builder.target_genres)
        books_per_genre = args.target_size // total_genres
        remainder = args.target_size % total_genres
        
        print(f"🎯 OBJECTIF PERSONNALISÉ: {args.target_size} livres")
        
        new_targets = {}
        for i, genre in enumerate(builder.target_genres.keys()):
            target = books_per_genre + (1 if i < remainder else 0)
            new_targets[genre] = target
        
        builder.target_genres = new_targets
    
    elif args.genres:
        print(f"🎯 GENRES SÉLECTIONNÉS: {', '.join(args.genres)}")
        # Garder seulement les genres demandés
        original_targets = builder.target_genres.copy()
        builder.target_genres = {
            genre: original_targets[genre] 
            for genre in args.genres 
            if genre in original_targets
        }
    
    return builder

def main():
    """Fonction principale"""
    print("🚀 DATASET BUILDER AUTONOME")
    print("=" * 50)
    
    # Parser les arguments
    args = parse_arguments()
    
    # Configuration
    enable_descriptions = not args.no_descriptions
    print(f"📝 Descriptions: {'✅ Activées' if enable_descriptions else '❌ Désactivées'}")
    print(f"📁 Fichier de sortie: {args.output}")
    
    # Créer le dossier de sortie si nécessaire
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Configurer le builder
        builder = configure_builder(args)
        
        # Afficher les objectifs
        print("\n🎯 OBJECTIFS PAR GENRE:")
        total_target = 0
        for genre, count in builder.target_genres.items():
            print(f"   {genre}: {count} livres")
            total_target += count
        
        print(f"\n📊 TOTAL CIBLE: {total_target} livres")
        
        if enable_descriptions:
            print(f"⏱️  TEMPS ESTIMÉ: {total_target * 2 // 60} minutes")
        else:
            print(f"⏱️  TEMPS ESTIMÉ: {total_target * 0.5 // 60} minutes")
        
        # Demander confirmation
        response = input("\n🤔 Continuer ? (y/N): ").lower().strip()
        if response not in ['y', 'yes', 'oui', 'o']:
            print("❌ Construction annulée")
            return
        
        # Créer le tracker de progrès
        progress_tracker = ProgressTracker(total_target)
        
        # Lancer la construction
        print("\n🏗️  DÉMARRAGE DE LA CONSTRUCTION...")
        print("=" * 50)
        
        start_time = time.time()
        
        final_stats = builder.build_improved_dataset (
            progress_callback=progress_tracker.update
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Sauvegarder
        print(f"\n💾 SAUVEGARDE dans {args.output}...")
        
        dataset_data = {
            "books_data": final_stats['books_data'],
            "stats": final_stats.get('quality_stats', {}),
            "timestamp": time.time(),
            "enable_descriptions": enable_descriptions,
            "target_genres": builder.target_genres,
            "version": "super_improved_v1",
            "build_duration_seconds": duration,
            "total_books_built": len(final_stats['books_data'])
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        
        # Rapport final
        print("=" * 50)
        print("🎉 CONSTRUCTION TERMINÉE AVEC SUCCÈS !")
        print("=" * 50)
        
        books_built = len(final_stats['books_data'])
        success_rate = (books_built / total_target) * 100
        
        print(f"📚 Livres construits: {books_built:,} / {total_target:,}")
        print(f"📈 Taux de réussite: {success_rate:.1f}%")
        print(f"⏱️  Durée totale: {progress_tracker._format_time(duration)}")
        print(f"🏎️  Vitesse: {books_built / (duration / 60):.1f} livres/minute")
        
        if 'quality_stats' in final_stats:
            quality_stats = final_stats['quality_stats']
            print("\n📊 QUALITÉ DES DONNÉES:")
            print(f"   🔍 Livres analysés: {quality_stats.get('total_found', 0):,}")
            print(f"   ✅ Qualité validée: {quality_stats.get('quality_passed', 0):,}")
            print(f"   🌟 Enrichis ISBN: {quality_stats.get('isbn_enriched', 0):,}")
            
            if quality_stats.get('total_found', 0) > 0:
                quality_rate = (quality_stats.get('quality_passed', 0) / quality_stats['total_found']) * 100
                print(f"   📈 Taux de qualité: {quality_rate:.1f}%")
        
        print(f"\n💾 Dataset sauvegardé: {args.output}")
        print("🚀 Prêt pour: streamlit run app.py")
        
        # Conseils d'utilisation
        if success_rate < 80:
            print("\n💡 CONSEILS POUR AMÉLIORER:")
            print("   - Réessayez avec --no-descriptions pour un dataset de base")
            print("   - Utilisez --quick-test pour tester la configuration")
            print("   - Vérifiez votre connexion internet")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTION PAR L'UTILISATEUR")
        print("💾 Construction interrompue. Fichiers partiels peuvent exister.")
        sys.exit(1)
        
    except Exception as e:
        print("\n❌ ERREUR CRITIQUE: {str(e)}")
        print("🔧 Détails techniques pour le debug:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()