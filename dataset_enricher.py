import pandas as pd
import requests
import time
import json
import os
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class OptimizedOpenLibraryEnricher:
    
    def __init__(self, api_delay=0.2, max_workers=3, cache_file="isbn_cache.json"):
        self.api_delay = api_delay
        self.max_workers = max_workers
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Charge le cache depuis le fichier JSON"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"📂 Cache chargé: {len(cache)} ISBN déjà traités")
                    return cache
            except Exception as e:
                print(f"⚠️ Erreur chargement cache: {e}")
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache dans le fichier JSON"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde cache: {e}")
    
    def get_metadata_batch(self, isbn_list: list) -> Dict:
        """Traite un batch d'ISBN avec parallélisme contrôlé"""
        
        # Filtrer les ISBN déjà en cache
        isbn_to_process = [isbn for isbn in isbn_list if str(isbn) not in self.cache]
        
        if not isbn_to_process:
            print("📋 Tous les ISBN sont déjà en cache")
            return {str(isbn): self.cache[str(isbn)] for isbn in isbn_list}
        
        print(f"🔄 Traitement de {len(isbn_to_process)} nouveaux ISBN...")
        
        # Traitement parallèle avec contrôle du nombre de threads
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumettre les tâches
            future_to_isbn = {
                executor.submit(self._get_single_metadata, isbn): isbn 
                for isbn in isbn_to_process
            }
            
            # Récupérer les résultats avec barre de progression
            for future in tqdm(as_completed(future_to_isbn), 
                             total=len(isbn_to_process), 
                             desc="📚 Enrichissement API"):
                
                isbn = future_to_isbn[future]
                try:
                    metadata = future.result()
                    results[str(isbn)] = metadata
                    self.cache[str(isbn)] = metadata
                    
                    # Sauvegarder le cache périodiquement
                    if len(results) % 100 == 0:
                        self._save_cache()
                        
                except Exception as e:
                    print(f"❌ Erreur pour ISBN {isbn}: {e}")
                    results[str(isbn)] = self._fallback_metadata(isbn)
        
        # Sauvegarder le cache final
        self._save_cache()
        
        # Retourner tous les résultats (cache + nouveaux)
        final_results = {}
        for isbn in isbn_list:
            final_results[str(isbn)] = self.cache.get(str(isbn), self._fallback_metadata(isbn))
        
        return final_results

    def _get_single_metadata(self, isbn: str) -> Dict:
        """Récupère les métadonnées d'un seul ISBN"""
        
        # Délai avant la requête
        time.sleep(self.api_delay)
        
        url = "https://openlibrary.org/api/books"
        params = {
            'bibkeys': f'ISBN:{isbn}',
            'jscmd': 'details',
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            book_data = data.get(f"ISBN:{isbn}", {}).get('details', {})
            
            if not book_data:
                return self._fallback_metadata(isbn)
            
            # Extraction des métadonnées (version propre - une seule colonne de chaque)
            # Extraire une seule fois chaque donnée
            title = book_data.get('title', f'Book_{isbn}')
            authors = self._extract_authors(book_data)
            year = self._extract_year(book_data)
            publisher = self._extract_publisher(book_data)
            cover_url = self._extract_cover(book_data, isbn)
            description = self._extract_description(book_data, isbn)
            subjects = self._extract_subjects(book_data)
            
            metadata = {
                'ISBN': isbn,
                'Book-Title': title,                    # Pour SVD (train.py)
                'Book-Author': authors,
                'api_first_publish_year': year,         # Convention enrichie
                'api_publisher_string': publisher,      # Convention enrichie
                'api_cover_url': cover_url,             # Convention enrichie
                'Description': description,
                'subject_string_final': subjects,       # Convention enrichie
                'api_enriched': True,
                'used_fallback': False
            }
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Erreur API pour ISBN {isbn}: {e}")
            return self._fallback_metadata(isbn)
    
    def _extract_authors(self, book_data: Dict) -> str:
        """Extrait les auteurs"""
        authors = []
        if 'authors' in book_data:
            for author in book_data['authors']:
                if 'name' in author:
                    authors.append(author['name'])
        return ', '.join(authors) if authors else 'Unknown Author'
    
    def _extract_year(self, book_data: Dict) -> Optional[int]:
        """Extrait l'année de publication"""
        for field in ['first_publish_date', 'publish_date', 'created']:
            if field in book_data:
                try:
                    year_str = str(book_data[field])
                    # Extraire les 4 premiers chiffres
                    year = int(year_str[:4])
                    if 1000 <= year <= 2030:  # Validation basique
                        return year
                except:
                    continue
        return None
    
    def _extract_publisher(self, book_data: Dict) -> str:
        """Extrait l'éditeur"""
        publishers = book_data.get('publishers', [])
        return ', '.join(publishers[:3]) if publishers else 'Unknown Publisher'
    
    def _extract_cover(self, book_data: Dict, isbn: str) -> str:
        """Extrait l'URL de couverture"""
        covers = book_data.get('covers', [])
        if covers:
            return f"https://covers.openlibrary.org/b/id/{covers[0]}-L.jpg"
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    
    def _extract_description(self, book_data: Dict, isbn: str) -> str:
        """Extrait la description"""
        desc = book_data.get('description', '')
        if isinstance(desc, dict) and 'value' in desc:
            description = desc['value'].strip()
            if description:
                return description
        elif isinstance(desc, str) and desc.strip():
            return desc.strip()
        
        try:
            works = book_data.get('works', [])
            if works:
                work_key = works[0]['key']  # Prendre le premier work
                description_from_work = self._get_work_description(work_key)
                if description_from_work:
                    return description_from_work
        except Exception as e:
            print(f"⚠️ Erreur récupération work pour ISBN {isbn}: {e}")
        
        subjects = self._extract_subjects(book_data)
        if subjects != 'Fiction':
            return f"Livre traitant de: {subjects}"
    
        return 'No description available'
    
    def _get_work_description(self, work_key: str) -> str:
        """
        Récupère la description depuis un Work OpenLibrary
        """
        try:
            # Petite pause pour éviter de surcharger l'API
            time.sleep(0.1)
        
            work_url = f"https://openlibrary.org{work_key}.json"
            response = requests.get(work_url, timeout=8)
            response.raise_for_status()
        
            work_data = response.json()
        
            # Extraire la description du work
            desc = work_data.get('description', '')
            if isinstance(desc, dict) and 'value' in desc:
                return desc['value'].strip()
            elif isinstance(desc, str) and desc.strip():
                return desc.strip()
        
            return ''
        
        except Exception as e:
            print(f"⚠️ Erreur récupération work {work_key}: {e}")
            return ''
    
    def _extract_subjects(self, book_data: Dict) -> str:
        """Extrait les sujets/genres"""
        subjects = book_data.get('subjects', [])
        return ', '.join(subjects[:10]) if subjects else 'Fiction'
    
    def _fallback_metadata(self, isbn: str) -> Dict:
        """Métadonnées de fallback quand l'API échoue (version propre)"""
        fallback_title = f'Book_{isbn}'
        fallback_cover = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        fallback_author = 'Unknown Author'
        fallback_publisher = 'Unknown Publisher'
        fallback_subjects = 'Fiction'
        
        return {
            'ISBN': isbn,
            'Book-Title': fallback_title,              # Pour SVD
            'Book-Author': fallback_author,
            'api_first_publish_year': None,            # Convention enrichie
            'api_publisher_string': fallback_publisher, # Convention enrichie
            'api_cover_url': fallback_cover,           # Convention enrichie
            'Description': 'No description available',
            'subject_string_final': fallback_subjects, # Convention enrichie
            'api_enriched': False,
            'used_fallback': True
        }


def enrich_ratings_optimized(ratings_csv: str, output_csv: str, 
                           batch_size: int = 1000, max_workers: int = 3):
    """
    Enrichit le dataset de ratings de manière optimisée
    
    Args:
        ratings_csv: Chemin vers le fichier CSV original
        output_csv: Chemin de sortie pour le fichier enrichi
        batch_size: Taille des batches pour le traitement
        max_workers: Nombre de threads parallèles
    """
    
    print("🚀 Démarrage de l'enrichissement optimisé...")
    
    # 1. Charger les données
    print("📂 Chargement des données...")
    df = pd.read_csv(ratings_csv)
    print(f"📊 Dataset chargé: {len(df):,} lignes")
    
    # 2. Extraire les ISBN uniques et convertir en liste
    unique_isbns = df['ISBN'].unique().tolist()  # Conversion directe
    print(f"📚 ISBN uniques à traiter: {len(unique_isbns):,}")
    
    # 3. Initialiser l'enrichisseur
    enricher = OptimizedOpenLibraryEnricher(
        api_delay=0.2, 
        max_workers=max_workers
    )
    
    # 4. Traitement par batches
    all_metadata = {}
    
    for i in range(0, len(unique_isbns), batch_size):
        batch = unique_isbns[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(unique_isbns) + batch_size - 1) // batch_size
        
        print(f"\n🔄 Traitement batch {batch_num}/{total_batches}")
        print(f"📋 ISBN dans ce batch: {len(batch)}")
        
        # Traiter le batch
        batch_metadata = enricher.get_metadata_batch(batch)
        all_metadata.update(batch_metadata)
        
        # Estimation du temps restant
        if batch_num > 1:
            avg_time_per_batch = (time.time() - start_time) / batch_num
            remaining_time = avg_time_per_batch * (total_batches - batch_num)
            print(f"⏱️ Temps restant estimé: {remaining_time/3600:.1f} heures")
    
    # 5. Créer le DataFrame des métadonnées
    print("\n🔗 Fusion des données...")
    metadata_df = pd.DataFrame([all_metadata[str(isbn)] for isbn in unique_isbns])
    metadata_df['ISBN'] = unique_isbns
    
    # 6. Fusionner avec le dataset original
    df_enriched = df.merge(metadata_df, on='ISBN', how='left')
    
    # 7. Sauvegarder
    print(f"💾 Sauvegarde vers {output_csv}...")
    df_enriched.to_csv(output_csv, index=False)
    
    # 8. Statistiques finales
    print("\n✅ Enrichissement terminé !")
    print(f"📊 Lignes finales: {len(df_enriched):,}")
    print(f"🌟 Livres enrichis avec API: {metadata_df['api_enriched'].sum():,}")
    print(f"🔄 Livres avec fallback: {metadata_df['used_fallback'].sum():,}")
    
    return df_enriched


# Script principal
if __name__ == "__main__":
    start_time = time.time()
    
    # Configuration
    input_file = "data/Ratings_enriched.csv"
    output_file = "data/Ratings_enriched_optimized.csv"
    
    # Lancer l'enrichissement
    enriched_df = enrich_ratings_optimized(
        ratings_csv=input_file,
        output_csv=output_file,
        batch_size=500,    # Batches plus petits pour plus de contrôle
        max_workers=3      # 3 threads parallèles
    )
    
    # Temps total
    total_time = time.time() - start_time
    print(f"\n⏱️ Temps total d'exécution: {total_time/3600:.1f} heures")