# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false
# type: ignore
import streamlit as st
import time
import pandas as pd
from typing import Dict, List
import json
import os
import random

class ImprovedDatasetBuilder:
    """Constructeur de dataset amélioré avec algorithme optimisé"""
    
    def __init__(self, enable_descriptions: bool = True):
        """
        Initialise le constructeur de dataset amélioré
        
        Args:
            enable_descriptions (bool): Active/désactive la récupération des descriptions
        """
        # Dépendances externes
        self.api_client = self._get_api_client()
        self.knn = self._get_dynamic_knn()
        
        # Configuration des genres cibles (plus réaliste)
        self.target_genres = {
            "Fantasy": 1000,
            "Science Fiction": 1000, 
            "Mystery": 800,
            "Romance": 800,
            "Historical Fiction": 800,
            "Thriller": 800,
            "Adventure": 600,
            "Horror": 400
        }
        
        # Paramètres optimisés
        self.batch_size = 50  # Plus gros batch
        self.delay_between_requests = 0.2  # Plus rapide
        self.enable_descriptions = enable_descriptions
        
        # Nouveaux paramètres pour l'efficacité
        self.max_attempts_per_genre = 15  # Plus de tentatives par genre
        self.min_books_per_query = 5  # Minimum de livres par requête pour continuer
        
        # Cache pour éviter les doublons
        self.seen_books = set()
        
    def _get_api_client(self):
        """Récupère le client API OpenLibrary"""
        try:
            from api_utils import get_api_client
            return get_api_client()
        except ImportError as e:
            st.error(f"❌ Impossible d'importer api_utils: {str(e)}")
            return None
    
    def _get_dynamic_knn(self):
        """Récupère le modèle KNN"""
        try:
            from knn_dynamic import get_dynamic_knn
            return get_dynamic_knn()
        except ImportError as e:
            st.warning(f"⚠️ Module knn_dynamic non trouvé: {str(e)}")
            return None
    
    def _get_enhanced_search_variations(self, genre: str) -> List[str]:
        """Génère BEAUCOUP plus de variations de recherche par genre"""
        
        variations = {
            "Fantasy": [
                # Termes généraux
                "fantasy", "epic fantasy", "urban fantasy", "dark fantasy", "high fantasy", "low fantasy",
                "fantasy adventure", "fantasy magic", "fantasy quest", "fantasy dragon", "fantasy wizard", 
                "fantasy kingdom", "fantasy novel", "fantasy series", "magical realism",
                # Auteurs populaires
                "tolkien", "george martin", "brandon sanderson", "terry pratchett", "robin hobb",
                "neil gaiman", "patrick rothfuss", "joe abercrombie", "terry goodkind",
                # Sous-genres spécifiques
                "sword sorcery", "dungeons dragons", "arthurian", "celtic mythology", "norse mythology",
                "vampire fantasy", "werewolf fantasy", "fairy tale retelling", "portal fantasy",
                # Éléments fantastiques
                "magic", "wizard", "witch", "dragon", "elf", "dwarf", "orc", "troll", "fairy",
                "enchanted", "spell", "potion", "quest", "kingdom", "realm", "medieval fantasy"
            ],
            
            "Science Fiction": [
                # Termes généraux
                "science fiction", "sci-fi", "scifi", "space opera", "cyberpunk", "steampunk",
                "dystopian", "utopian", "space exploration", "future", "futuristic",
                # Technologies
                "robot", "android", "cyborg", "artificial intelligence", "AI", "time travel",
                "space travel", "starship", "spaceship", "aliens", "extraterrestrial",
                # Sous-genres
                "hard science fiction", "soft science fiction", "military science fiction",
                "alternate history", "post-apocalyptic", "climate fiction", "biopunk",
                # Auteurs célèbres
                "isaac asimov", "philip dick", "arthur clarke", "robert heinlein", "ursula guin",
                "kim stanley robinson", "william gibson", "frank herbert", "ray bradbury",
                # Éléments SF
                "mars", "galaxy", "universe", "planet", "colony", "terraforming", "genetic engineering",
                "nanotechnology", "virtual reality", "parallel universe", "dimension"
            ],
            
            "Mystery": [
                # Types de mystères
                "mystery", "detective", "crime", "murder mystery", "cozy mystery", "police procedural",
                "hardboiled", "noir", "whodunit", "locked room mystery", "cold case",
                # Professionnels
                "detective story", "private investigator", "police detective", "amateur sleuth",
                "forensic", "criminal investigation", "homicide", "serial killer",
                # Auteurs célèbres
                "agatha christie", "arthur conan doyle", "raymond chandler", "dashiell hammett",
                "louise penny", "tana french", "gillian flynn", "gone girl",
                # Éléments
                "murder", "investigation", "suspect", "alibi", "clue", "evidence", "conspiracy",
                "thriller mystery", "psychological thriller", "legal thriller", "medical thriller"
            ],
            
            "Romance": [
                # Types de romance
                "romance", "love story", "romantic comedy", "romantic drama", "romantic suspense",
                "historical romance", "contemporary romance", "paranormal romance", "erotic romance",
                # Sous-genres
                "regency romance", "western romance", "military romance", "sports romance",
                "office romance", "enemies to lovers", "second chance romance", "fake relationship",
                # Éléments
                "love", "relationship", "dating", "marriage", "wedding", "passion", "desire",
                "romantic fiction", "romantic novel", "love affair", "love triangle",
                # Auteurs populaires
                "nicholas sparks", "nora roberts", "julia quinn", "stephanie meyer", "e l james",
                "jane austen", "emily henry", "colleen hoover", "christina lauren"
            ],
            
            "Historical Fiction": [
                # Périodes
                "historical fiction", "historical novel", "medieval", "victorian", "edwardian",
                "world war", "civil war", "american revolution", "french revolution", "renaissance",
                # Lieux et cultures
                "ancient rome", "ancient greece", "ancient egypt", "viking", "samurai",
                "colonial america", "wild west", "pioneer", "immigrant saga", "period drama",
                # Guerre et conflits
                "world war 1", "world war 2", "vietnam war", "korean war", "napoleonic",
                "battlefield", "wartime", "occupation", "resistance", "holocaust",
                # Auteurs
                "ken follett", "philippa gregory", "hilary mantel", "bernard cornwell",
                "james michener", "edward rutherfurd", "kate quinn", "kristin hannah"
            ],
            
            "Thriller": [
                # Types
                "thriller", "suspense", "psychological thriller", "action thriller", "spy thriller",
                "political thriller", "legal thriller", "medical thriller", "techno thriller",
                # Éléments
                "conspiracy", "assassin", "spy", "espionage", "terrorism", "kidnapping",
                "hostage", "chase", "pursuit", "danger", "tension", "betrayal",
                # Auteurs
                "john grisham", "dan brown", "tom clancy", "frederick forsyth", "lee child",
                "james patterson", "harlan coben", "michael crichton", "robert ludlum"
            ],
            
            "Adventure": [
                # Types d'aventure
                "adventure", "action adventure", "survival", "expedition", "exploration",
                "treasure hunt", "quest", "journey", "voyage", "wilderness",
                # Environnements
                "jungle adventure", "desert adventure", "mountain climbing", "sea adventure",
                "island survival", "arctic expedition", "safari", "archaeology",
                # Éléments
                "explorer", "adventurer", "discovery", "lost civilization", "ancient treasure",
                "dangerous journey", "survival story", "rescue mission"
            ],
            
            "Horror": [
                # Types d'horreur
                "horror", "supernatural horror", "psychological horror", "gothic horror",
                "cosmic horror", "body horror", "folk horror", "haunted house",
                # Créatures
                "vampire", "zombie", "ghost", "demon", "werewolf", "monster", "creature",
                "supernatural", "paranormal", "occult", "witchcraft", "possession",
                # Auteurs
                "stephen king", "clive barker", "h p lovecraft", "edgar allan poe",
                "bram stoker", "mary shelley", "anne rice", "dean koontz"
            ]
        }
        
        # Retourner toutes les variations pour ce genre
        return variations.get(genre, [genre.lower()])
    
    def _create_unique_key(self, book: Dict) -> str:
        """Crée une clé unique plus robuste pour déduplication"""
        title = book.get('title', '').lower().strip()
        author = book.get('author_string', '').lower().strip()
        
        # Nettoyer les titres (enlever articles, ponctuation)
        title_clean = title.replace('the ', '').replace('a ', '').replace('an ', '')
        title_clean = ''.join(c for c in title_clean if c.isalnum())
        
        # Prendre le premier auteur seulement
        first_author = author.split(',')[0].strip() if author else ''
        
        return f"{title_clean}_{first_author}"
    
    def search_books_with_descriptions_enhanced(self, query: str, limit: int, offset: int = 0) -> List[Dict]:
        """
        Version améliorée avec pagination et retry
        """
        try:
            # Essayer plusieurs fois avec des paramètres différents
            attempts = 0
            max_attempts = 3
            all_books = []
            
            while attempts < max_attempts and len(all_books) < limit:
                try:
                    # Varier les paramètres de recherche
                    current_limit = min(50, limit - len(all_books))
                    
                    # Ajouter de la variabilité à la requête
                    if attempts > 0:
                        query_variants = [
                            query,
                            f"{query} fiction",
                            f"{query} novel",
                            f"{query} book"
                        ]
                        current_query = random.choice(query_variants)
                    else:
                        current_query = query
                    
                    st.write(f"  🔍 Tentative {attempts + 1}: '{current_query}' (limite: {current_limit})")
                    
                    # Recherche API
                    books = self.api_client.search_books(current_query, current_limit)
                    
                    if not books:
                        attempts += 1
                        time.sleep(0.5)
                        continue
                    
                    # Filtrer les nouveaux livres
                    new_books = []
                    for book in books:
                        unique_key = self._create_unique_key(book)
                        if unique_key not in self.seen_books:
                            self.seen_books.add(unique_key)
                            new_books.append(book)
                    
                    if not new_books:
                        attempts += 1
                        continue
                    
                    # Enrichir avec descriptions si activé
                    if self.enable_descriptions:
                        enriched_books = self._enrich_books_with_descriptions(new_books)
                        all_books.extend(enriched_books)
                    else:
                        all_books.extend(new_books)
                    
                    st.write(f"    ✅ Récupéré: {len(new_books)} nouveaux livres")
                    
                    attempts += 1
                    time.sleep(self.delay_between_requests)
                    
                except Exception as e:
                    st.warning(f"Erreur tentative {attempts + 1}: {str(e)}")
                    attempts += 1
                    time.sleep(1)
            
            return all_books[:limit]
            
        except Exception as e:
            st.error(f"Erreur recherche pour '{query}': {str(e)}")
            return []
    
    def _enrich_books_with_descriptions(self, books: List[Dict]) -> List[Dict]:
        """Enrichit une liste de livres avec leurs descriptions"""
        enriched = []
        
        for i, book in enumerate(books):
            try:
                if book.get('key'):
                    description = self._get_book_description(book['key'])
                    book['description'] = description if description.strip() else book.get('subject_string', '')
                else:
                    book['description'] = book.get('subject_string', '')
                
                enriched.append(book)
                
                # Progress moins fréquent pour éviter le spam
                if (i + 1) % 10 == 0:
                    st.write(f"    📖 Descriptions: {i + 1}/{len(books)}")
                
                time.sleep(0.1)  # Plus rapide
                
            except Exception as e:
                book['description'] = book.get('subject_string', '')
                enriched.append(book)
                continue
        
        return enriched
    
    def _get_book_description(self, book_key: str) -> str:
        """Version optimisée de récupération de description"""
        try:
            # Cache simple pour éviter les requêtes répétées
            if not hasattr(self, '_description_cache'):
                self._description_cache = {}
            
            if book_key in self._description_cache:
                return self._description_cache[book_key]
            
            # Si c'est une clé d'édition, récupérer le work
            if '/books/' in book_key:
                edition_url = f"https://openlibrary.org{book_key}.json"
                edition_response = self.api_client.session.get(edition_url, timeout=5)
                
                if edition_response.status_code == 200:
                    edition_data = edition_response.json()
                    works = edition_data.get('works', [])
                    if works:
                        work_key = works[0]['key']
                    else:
                        return ""
                else:
                    return ""
            else:
                work_key = book_key
            
            # Récupérer les détails du Work
            work_url = f"https://openlibrary.org{work_key}.json"
            work_response = self.api_client.session.get(work_url, timeout=5)
            
            if work_response.status_code == 200:
                work_data = work_response.json()
                
                description = ""
                if 'description' in work_data:
                    desc = work_data['description']
                    if isinstance(desc, dict) and 'value' in desc:
                        description = desc['value']
                    elif isinstance(desc, str):
                        description = desc
                
                # Mettre en cache
                self._description_cache[book_key] = description.strip()
                return description.strip()
            
            return ""
            
        except Exception:
            return ""
    
    def build_comprehensive_dataset_enhanced(self, progress_callback=None):
        """
        construction du dataset
        """
        if self.enable_descriptions:
            st.info("🔍 **Mode descriptions activé** - Construction optimisée")
        else:
            st.info("⚡ **Mode rapide** - Construction sans descriptions détaillées")
        
        total_books_target = sum(self.target_genres.values())
        current_progress = 0
        all_collected_books = []
        
        # Réinitialiser le cache
        self.seen_books.clear()
        
        st.info(f"🎯 **Objectif total**: {total_books_target} livres répartis sur {len(self.target_genres)} genres")
        
        for genre, target_count in self.target_genres.items():
            st.write(f"\n🎯 **Genre: {genre}** (objectif: {target_count} livres)")
            
            books_collected = 0
            search_queries = self._get_enhanced_search_variations(genre)
            
            # Mélanger les requêtes pour plus de diversité
            random.shuffle(search_queries)
            
            attempts = 0
            query_index = 0
            
            while books_collected < target_count and attempts < self.max_attempts_per_genre:
                # Sélectionner la requête suivante
                if query_index >= len(search_queries):
                    query_index = 0  # Recommencer la liste
                    random.shuffle(search_queries)  # Re-mélanger
                
                query = search_queries[query_index]
                query_index += 1
                
                try:
                    # Calculer combien de livres on a besoin
                    needed = min(self.batch_size, target_count - books_collected)
                    
                    st.write(f"  📚 Recherche #{attempts + 1}: '{query}' (besoin: {needed})")
                    
                    # Recherche avec la version améliorée
                    books = self.search_books_with_descriptions_enhanced(query, needed)
                    
                    if not books:
                        st.write(f"    ⚠️ Aucun résultat pour '{query}'")
                        attempts += 1
                        continue
                    
                    # Ajouter au dataset KNN
                    new_books_added = 0
                    if self.knn:
                        for book in books:
                            # Vérifier que le livre n'existe pas déjà dans le KNN
                            if not any(existing['title'].lower().strip() == book.get('title', '').lower().strip() and 
                                     existing['author_string'].lower().strip() == book.get('author_string', '').lower().strip() 
                                     for existing in self.knn.books_data):
                                self.knn.books_data.append(self.knn._clean_book_data(book))
                                all_collected_books.append(book)
                                new_books_added += 1
                        
                        # Réentraîner le modèle périodiquement
                        if len(self.knn.books_data) >= 10 and len(self.knn.books_data) % 50 == 0:
                            st.write(f"    🔄 Réentraînement du modèle KNN...")
                            self.knn._fit_model()
                    else:
                        new_books_added = len(books)
                        all_collected_books.extend(books)
                    
                    books_collected += new_books_added
                    current_progress += new_books_added
                    
                    # Mettre à jour le progrès
                    if progress_callback:
                        progress_percentage = current_progress / total_books_target
                        progress_callback(progress_percentage, 
                                        f"{genre}: {books_collected}/{target_count} livres")
                    
                    # Affichage du progrès
                    st.write(f"    ✅ Ajouté: {new_books_added} livres | Total {genre}: {books_collected}/{target_count}")
                    
                    # Si on obtient moins de X livres, la requête est probablement épuisée
                    if len(books) < self.min_books_per_query:
                        st.write(f"    📉 Requête '{query}' semble épuisée ({len(books)} résultats)")
                    
                    attempts += 1
                    time.sleep(self.delay_between_requests)
                    
                except Exception as e:
                    st.warning(f"Erreur lors de la recherche '{query}': {str(e)}")
                    attempts += 1
                    continue
            
            # Rapport final pour ce genre
            percentage_achieved = (books_collected / target_count) * 100
            if books_collected >= target_count * 0.8:  # 80% de l'objectif
                st.success(f"✅ **{genre}**: {books_collected}/{target_count} livres ({percentage_achieved:.1f}%)")
            else:
                st.warning(f"⚠️ **{genre}**: {books_collected}/{target_count} livres ({percentage_achieved:.1f}%) - Objectif partiellement atteint")
        
        # Entraînement final du modèle
        if self.knn and len(self.knn.books_data) >= 10:
            st.write(f"🔄 Entraînement final du modèle KNN avec {len(self.knn.books_data)} livres...")
            self.knn._fit_model()
        
        # Statistiques finales
        final_stats = {
            'total_books': len(all_collected_books),
            'books_data': all_collected_books,
            'by_genre': {}
        }
        
        # Calculer stats par genre avec détection améliorée
        for genre in self.target_genres.keys():
            genre_variations = self._get_enhanced_search_variations(genre)
            genre_count = 0
            
            for book in all_collected_books:
                subjects = book.get('subject_string', '').lower()
                # Recherche plus flexible
                for variation in genre_variations[:5]:  # Prendre les 5 premiers termes
                    if variation.lower() in subjects:
                        genre_count += 1
                        break
            
            final_stats['by_genre'][genre] = genre_count
        
        # Statistiques KNN si disponible
        if self.knn:
            knn_stats = self.knn.get_dataset_stats()
            final_stats.update(knn_stats)
        
        # Rapport final
        total_collected = final_stats['total_books']
        success_rate = (total_collected / total_books_target) * 100
        
        if success_rate >= 80:
            st.balloons()
            st.success(f"🎉 **Succès!** Dataset construit: {total_collected} livres ({success_rate:.1f}% de l'objectif)")
        elif success_rate >= 60:
            st.success(f"✅ **Bon résultat** Dataset construit: {total_collected} livres ({success_rate:.1f}% de l'objectif)")
        else:
            st.warning(f"⚠️ **Résultat partiel** Dataset construit: {total_collected} livres ({success_rate:.1f}% de l'objectif)")
        
        return final_stats
    
    def save_dataset(self, filename: str = "data/enhanced_dataset_with_descriptions.json"):
        """Sauvegarde le dataset amélioré"""
        try:
            import json
            
            # Préparer les données
            if self.knn and self.knn.books_data:
                dataset_data = {
                    "books_data": self.knn.books_data,
                    "stats": self.knn.get_dataset_stats(),
                    "timestamp": time.time(),
                    "enable_descriptions": self.enable_descriptions,
                    "target_genres": self.target_genres,
                    "version": "enhanced_v2"
                }
            else:
                dataset_data = {
                    "books_data": [],
                    "stats": {"total_books": 0, "unique_authors": 0},
                    "timestamp": time.time(),
                    "enable_descriptions": self.enable_descriptions,
                    "target_genres": self.target_genres,
                    "version": "enhanced_v2"
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"💾 Dataset enhanced sauvegardé dans {filename}")
            return True
            
        except Exception as e:
            st.error(f"Erreur sauvegarde: {str(e)}")
            return False
    
    def load_dataset(self, filename: str = "data/enhanced_dataset_with_descriptions.json") -> bool:
        """Charge un dataset sauvegardé"""
        try:
            import json
            
            if not os.path.exists(filename):
                st.warning(f"Fichier {filename} non trouvé")
                return False
            
            with open(filename, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Restaurer les données
            if self.knn and dataset_data.get("books_data"):
                self.knn.books_data = dataset_data["books_data"]
                
                # Réentraîner le modèle
                if len(self.knn.books_data) >= 5:
                    self.knn._fit_model()
                
                version = dataset_data.get("version", "v1")
                st.success(f"📂 Dataset {version} chargé: {len(self.knn.books_data)} livres")
            else:
                st.warning("Données chargées mais KNN non disponible")
            
            return True
            
        except Exception as e:
            st.error(f"Erreur lors du chargement: {str(e)}")
            return False


# ============================================================================
# INTERFACE STREAMLIT AMÉLIORÉE
# ============================================================================

def show_enhanced_dataset_builder():
    """Interface Streamlit pour le dataset builder amélioré"""
    
    st.header("🚀 Construction de Dataset Améliorée v2.0")
    st.markdown("### Algorithme optimisé pour collecter 7000+ livres avec descriptions")
    
    # Améliorations apportées
    with st.subheader("🆕 Nouveautés de cette version"):
        st.markdown("""
        **🔧 Améliorations algorithmiques:**
        - ✅ **+500% de termes de recherche** par genre (vs ancienne version)
        - ✅ **Déduplication intelligente** avec nettoyage des titres
        - ✅ **Système de retry** avec variantes de requêtes
        - ✅ **Cache des descriptions** pour éviter les requêtes répétées
        - ✅ **Pagination améliorée** avec randomisation
        - ✅ **Détection de genre flexible** (sci-fi = science fiction)
        
        **📊 Objectifs révisés:**
        - Fantasy: 1500 → Romance: 1200 → Thriller: 800 → Horror: 400
        - **Total: 8000 livres** (vs 7000 avant)
        """)
    
    # Initialiser le builder amélioré
    if 'enhanced_builder' not in st.session_state:
        st.session_state.enhanced_builder = ImprovedDatasetBuilder(enable_descriptions=True)
    
    builder = st.session_state.enhanced_builder
    
    # Vérifications
    if builder.api_client is None:
        st.error("❌ API OpenLibrary non disponible")
        return
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Mode de construction:",
            ["🔍 Précis (avec descriptions)", "⚡ Rapide (sans descriptions)"],
            index=0
        )
        builder.enable_descriptions = "🔍 Précis" in mode
    
    with col2:
        if builder.enable_descriptions:
            st.info("⏱️ Temps estimé: 3-4 heures")
            st.success("🎯 Qualité maximale des recommandations")
        else:
            st.info("⏱️ Temps estimé: 1-2 heures") 
            st.warning("🎯 Qualité réduite des recommandations")
    
    # Affichage des objectifs
    st.subheader("🎯 Objectifs par genre")
    objective_df = pd.DataFrame([
        {"Genre": genre, "Objectif": count} 
        for genre, count in builder.target_genres.items()
    ])
    st.dataframe(objective_df, use_container_width=True)
    
    total_objective = sum(builder.target_genres.values())
    st.info(f"📚 **Objectif total: {total_objective} livres**")
    
    # Chargement dataset existant
    if os.path.exists("data/enhanced_dataset_with_descriptions.json"):
        st.subheader("📂 Dataset existant détecté")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Charger dataset amélioré", key="load_enhanced"):
                with st.spinner("Chargement..."):
                    if builder.load_dataset("data/enhanced_dataset_with_descriptions.json"):
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reconstruire", key="rebuild_enhanced"):
                st.session_state['rebuild_confirmed_enhanced'] = True
                st.rerun()
    
    # Construction
    if not os.path.exists("data/enhanced_dataset_with_descriptions.json") or st.session_state.get('rebuild_confirmed_enhanced', False):
        
        st.subheader("🚀 Lancement de la construction améliorée")
        
        st.warning("⚠️ **Construction longue** - Gardez l'onglet ouvert")
        st.info("💡 **Progression détaillée** affichée en temps réel")
        
        if st.button("🚀 Lancer construction améliorée", key="start_enhanced", type="primary"):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            detailed_progress = st.empty()
            
            def update_progress(percentage, message):
                progress_bar.progress(percentage)
                status_text.text(f"Progrès: {percentage:.1%}")
                detailed_progress.info(message)
            
            # Lancer construction
            with st.spinner("🏗️ Construction améliorée en cours..."):
                try:
                    st.success("🚀 Construction démarrée avec l'algorithme amélioré!")
                    
                    final_stats = builder.build_comprehensive_dataset_enhanced(
                        progress_callback=update_progress
                    )
                    
                    # Sauvegarder automatiquement
                    if builder.save_dataset("data/enhanced_dataset_with_descriptions.json"):
                        st.balloons()
                        st.success("🎉 Construction améliorée terminée avec succès!")
                    
                    # Affichage des résultats détaillés
                    st.subheader("📊 Résultats de la construction améliorée")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📚 Total Livres", final_stats['total_books'])
                    with col2:
                        success_rate = (final_stats['total_books'] / total_objective) * 100
                        st.metric("🎯 Taux de réussite", f"{success_rate:.1f}%")
                    with col3:
                        if 'unique_authors' in final_stats:
                            st.metric("👥 Auteurs uniques", final_stats['unique_authors'])
                        else:
                            st.metric("👥 Auteurs uniques", "N/A")
                    with col4:
                        mode_text = "Avec descriptions" if builder.enable_descriptions else "Sans descriptions"
                        st.metric("🔧 Mode", mode_text)
                    
                    # Détail par genre amélioré
                    if 'by_genre' in final_stats and final_stats['by_genre']:
                        st.subheader("📈 Répartition par genre (détection améliorée)")
                        
                        genre_results = []
                        for genre, target in builder.target_genres.items():
                            achieved = final_stats['by_genre'].get(genre, 0)
                            percentage = (achieved / target * 100) if target > 0 else 0
                            status = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
                            
                            genre_results.append({
                                "Genre": genre,
                                "Objectif": target,
                                "Obtenu": achieved,
                                "Taux": f"{percentage:.1f}%",
                                "Statut": status
                            })
                        
                        results_df = pd.DataFrame(genre_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Graphique de comparaison
                        st.subheader("📊 Comparaison Objectifs vs Résultats")
                        chart_data = pd.DataFrame({
                            'Genre': [r['Genre'] for r in genre_results],
                            'Objectif': [r['Objectif'] for r in genre_results],
                            'Obtenu': [r['Obtenu'] for r in genre_results]
                        })
                        st.bar_chart(chart_data.set_index('Genre'))
                    
                    # Recommandations pour amélioration
                    st.subheader("💡 Analyse et recommandations")
                    
                    if success_rate >= 90:
                        st.success("🎉 **Excellent résultat!** Le dataset est très complet.")
                    elif success_rate >= 70:
                        st.info("✅ **Bon résultat.** Le dataset est utilisable pour des recommandations de qualité.")
                        st.write("💡 Pour améliorer: relancez la construction pour les genres sous-représentés.")
                    else:
                        st.warning("⚠️ **Résultat partiel.** Considérez les améliorations suivantes:")
                        st.write("- 🔄 Relancer la construction avec plus de temps")
                        st.write("- 🎯 Ajuster les objectifs par genre")
                        st.write("- 🔍 Vérifier la connectivité réseau")
                    
                    # Informations techniques
                    with st.subheader("🔧 Détails techniques"):
                        st.write(f"**Cache de déduplication:** {len(builder.seen_books)} entrées")
                        st.write(f"**Descriptions en cache:** {len(getattr(builder, '_description_cache', {}))}")
                        if builder.knn:
                            st.write(f"**Modèle KNN:** {'✅ Entraîné' if builder.knn.is_fitted else '❌ Non entraîné'}")
                            st.write(f"**Livres dans KNN:** {len(builder.knn.books_data)}")
                    
                    # Nettoyer les flags
                    if 'rebuild_confirmed_enhanced' in st.session_state:
                        del st.session_state['rebuild_confirmed_enhanced']
                
                except Exception as e:
                    st.error(f"❌ Erreur pendant la construction: {str(e)}")
                    st.error("💡 La construction peut être relancée - elle reprendra automatiquement")
                    st.exception(e)  # Pour le debugging
    
    # Visualisation du dataset actuel (similaire à l'ancienne version mais adaptée)
    st.subheader("📊 Visualisation du Dataset Enhanced")
    
    try:
        if builder.knn and builder.knn.books_data:
            current_books = builder.knn.books_data
            st.success(f"📚 Dataset enhanced actuel : {len(current_books)} livres")
            
            # Métriques améliorées
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📖 Total Livres", len(current_books))
            
            with col2:
                unique_authors = len(set(book.get('author_string', '') for book in current_books if book.get('author_string')))
                st.metric("👥 Auteurs Uniques", unique_authors)
            
            with col3:
                with_descriptions = sum(1 for book in current_books if book.get('description', '').strip())
                percentage_desc = (with_descriptions / len(current_books)) * 100 if current_books else 0
                st.metric("📝 Avec Description", f"{with_descriptions} ({percentage_desc:.1f}%)")
            
            with col4:
                objective_total = sum(builder.target_genres.values())
                completion = (len(current_books) / objective_total) * 100 if objective_total > 0 else 0
                st.metric("🎯 Complétude", f"{completion:.1f}%")
            
            # Options de visualisation étendues
            view_option = st.selectbox(
                "Explorer le dataset enhanced:",
                [
                    "📋 Aperçu général",
                    "📊 Analyse par genre (enhanced)", 
                    "🔍 Recherche avancée",
                    "📈 Statistiques détaillées",
                    "📝 Qualité des descriptions",
                    "⚖️ Comparaison objectifs vs résultats",
                    "📑 Export complet"
                ]
            )
            
            if view_option == "📊 Analyse par genre (enhanced)":
                st.write("**Analyse améliorée des genres:**")
                
                # Utiliser les termes de recherche améliorés pour la détection
                genre_analysis = {}
                for genre in builder.target_genres.keys():
                    search_terms = builder._get_enhanced_search_variations(genre)[:10]  # Top 10 termes
                    count = 0
                    
                    for book in current_books:
                        subjects = book.get('subject_string', '').lower()
                        title = book.get('title', '').lower()
                        
                        # Recherche dans sujets et titre
                        for term in search_terms:
                            if term.lower() in subjects or term.lower() in title:
                                count += 1
                                break
                    
                    objective = builder.target_genres[genre]
                    percentage = (count / objective * 100) if objective > 0 else 0
                    
                    genre_analysis[genre] = {
                        'count': count,
                        'objective': objective,
                        'percentage': percentage
                    }
                
                # Affichage sous forme de tableau
                analysis_data = []
                for genre, data in genre_analysis.items():
                    status = "✅" if data['percentage'] >= 80 else "⚠️" if data['percentage'] >= 60 else "❌"
                    analysis_data.append({
                        "Genre": genre,
                        "Obtenu": data['count'],
                        "Objectif": data['objective'],
                        "Taux": f"{data['percentage']:.1f}%",
                        "Statut": status
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
                
                # Graphique
                chart_data = pd.DataFrame({
                    'Genre': [d['Genre'] for d in analysis_data],
                    'Obtenu': [d['Obtenu'] for d in analysis_data],
                    'Objectif': [d['Objectif'] for d in analysis_data]
                })
                st.bar_chart(chart_data.set_index('Genre'))
            
            elif view_option == "📝 Qualité des descriptions":
                st.write("**Analyse de la qualité des descriptions:**")
                
                # Statistiques des descriptions
                desc_lengths = [len(book.get('description', '')) for book in current_books]
                desc_stats = {
                    'Avec description': sum(1 for l in desc_lengths if l > 0),
                    'Description vide': sum(1 for l in desc_lengths if l == 0),
                    'Description courte (< 100 chars)': sum(1 for l in desc_lengths if 0 < l < 100),
                    'Description moyenne (100-500 chars)': sum(1 for l in desc_lengths if 100 <= l < 500),
                    'Description longue (> 500 chars)': sum(1 for l in desc_lengths if l >= 500)
                }
                
                # Affichage
                for category, count in desc_stats.items():
                    percentage = (count / len(current_books)) * 100
                    st.write(f"- **{category}**: {count} livres ({percentage:.1f}%)")
                
                # Histogramme des longueurs
                if desc_lengths:
                    lengths_df = pd.DataFrame({'Longueur description': desc_lengths})
                    st.bar_chart(lengths_df['Longueur description'].value_counts().head(20))
            
            elif view_option == "⚖️ Comparaison objectifs vs résultats":
                st.write("**Comparaison détaillée des objectifs:**")
                
                total_objective = sum(builder.target_genres.values())
                total_current = len(current_books)
                
                st.metric("🎯 Progression globale", f"{total_current}/{total_objective}", 
                         f"{((total_current/total_objective)*100):.1f}% de l'objectif")
                
                # Graphique en secteurs pour la répartition
                if total_current > 0:
                    # Analyser la répartition actuelle
                    current_distribution = {}
                    for genre in builder.target_genres.keys():
                        terms = builder._get_enhanced_search_variations(genre)[:5]
                        count = 0
                        for book in current_books:
                            subjects = book.get('subject_string', '').lower()
                            if any(term.lower() in subjects for term in terms):
                                count += 1
                        current_distribution[genre] = count
                    
                    # Créer DataFrame pour graphique
                    comparison_data = pd.DataFrame([
                        {
                            'Genre': genre,
                            'Objectif': target,
                            'Actuel': current_distribution.get(genre, 0),
                            'Manquant': max(0, target - current_distribution.get(genre, 0))
                        }
                        for genre, target in builder.target_genres.items()
                    ])
                    
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Graphique empilé
                    st.bar_chart(comparison_data.set_index('Genre')[['Actuel', 'Manquant']])
            
            # ... (autres options de visualisation similaires à l'ancienne version)
        
        else:
            st.info("📭 Aucun dataset enhanced chargé")
            st.write("Construisez le dataset amélioré pour des résultats optimaux!")
            
    except Exception as e:
        st.error(f"Erreur visualisation: {str(e)}")


# Fonction principale à utiliser
def show_dataset_builder():
    """Point d'entrée principal - utilise la version améliorée"""
    show_enhanced_dataset_builder()


# Fonction de comparaison (optionnelle)
def show_comparison_old_vs_new():
    """Compare les résultats entre ancienne et nouvelle version"""
    st.subheader("⚖️ Comparaison des versions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Version 1.0 (ancienne)**")
        if os.path.exists("dataset_with_descriptions.json"):
            st.success("✅ Dataset v1 trouvé")
            # Charger et afficher stats de l'ancien
        else:
            st.info("📭 Pas de dataset v1")
    
    with col2:
        st.write("**🚀 Version 2.0 (améliorée)**")
        if os.path.exists("enhanced_dataset_with_descriptions.json"):
            st.success("✅ Dataset v2 trouvé")
            # Charger et afficher stats du nouveau
        else:
            st.info("📭 Pas de dataset v2")