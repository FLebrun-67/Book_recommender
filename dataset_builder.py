import time
from typing import Dict, List
import random
import re

class ImprovedDatasetBuilder:
    """Dataset Builder avec améliorations pour plus de résultats"""
    
    def __init__(self, enable_descriptions: bool = True, verbose: bool = True):
        self.verbose = verbose
        
        # Dépendances externes
        self.api_client = self._get_api_client()
        
        # Configuration optimisée
        self.target_genres = {
            "Fantasy": 400,
            "Science Fiction": 400, 
            "Mystery": 400,
            "Romance": 400,
            "Historical Fiction": 400,
            "Thriller": 400,
            "Adventure": 400,
            "Horror": 400
        }
        
        # Paramètres optimisés pour plus de résultats
        self.batch_size = 50           # ← Augmenté de 30 à 50
        self.max_attempts_per_genre = 40  # ← Augmenté de 20 à 40
        self.delay_between_requests = 0.15  # ← Réduit de 0.2 à 0.15
        self.enable_descriptions = enable_descriptions
        
        # Filtre qualité adaptatif
        self.quality_threshold = 35    # ← Réduit de 50 à 35
        self.min_description_length = 15  # ← Réduit de 20 à 15
        
        # Cache et statistiques
        self.seen_books = set()
        self.rejected_books = set()
        self.quality_stats = self._init_quality_stats()
        
        # Nouveaux paramètres pour diversité
        self.min_books_per_query = 3   # Continue si au moins 3 livres trouvés
        self.query_rotation_limit = 3  # Nombre de rotations avant abandon
    
    def _init_quality_stats(self):
        """Initialise les statistiques de qualité"""
        return {
            'total_found': 0,
            'isbn_enriched': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'duplicate_rejected': 0,
            'queries_attempted': 0,
            'successful_queries': 0
        }
    
    def _get_api_client(self):
        """Récupère le client API OpenLibrary"""
        try:
            from api_utils import get_api_client
            client = get_api_client()  # ← Correction du bug avec parenthèses
            if client and hasattr(client, 'search_books'):
                self._log("✅ Client API initialisé avec succès", "INFO")
                return client
            else:
                self._log("❌ Client API invalide", "ERROR")
                return None
        
        except Exception as e:
            self._log(f"❌ Erreur API: {str(e)}", "ERROR")
            return None
    
    def _log(self, message: str, level: str = "INFO"):
        """Log amélioré avec timestamps"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def _get_enhanced_search_variations(self, genre: str) -> List[str]:
        """
        Variations de recherche MASSIVEMENT étendues
        ← Ici votre intuition était bonne !
        """
        variations = {
            "Fantasy": [
                # Termes généraux
                "fantasy", "epic fantasy", "urban fantasy", "dark fantasy", "high fantasy", 
                "low fantasy", "fantasy adventure", "fantasy magic", "fantasy quest", 
                "fantasy dragon", "fantasy wizard", "fantasy kingdom", "fantasy novel",
                "fantasy series", "magical realism", "fantasy fiction", "fantasy literature",
                
                # Auteurs populaires (TRÈS efficace)
                "tolkien", "j r r tolkien", "brandon sanderson", "terry pratchett", 
                "neil gaiman", "patrick rothfuss", "joe abercrombie", "terry goodkind",
                "robin hobb", "george r r martin", "game of thrones", "lord of the rings",
                "robert jordan", "wheel of time", "mistborn", "stormlight",
                
                # Sous-genres spécifiques
                "sword and sorcery", "sword sorcery", "dungeons dragons", "d&d fantasy",
                "arthurian legend", "arthurian", "celtic mythology", "norse mythology",
                "vampire fantasy", "werewolf fantasy", "fairy tale retelling", 
                "portal fantasy", "quest fantasy", "coming of age fantasy",
                
                # Éléments fantastiques
                "magic", "wizard", "witch", "sorcerer", "mage", "dragon", "dragons",
                "elf", "elves", "dwarf", "dwarves", "orc", "orcs", "troll", "fairy",
                "enchanted", "spell", "spells", "potion", "quest", "kingdom", "realm",
                "medieval fantasy", "epic quest", "chosen one", "magical world",
                
                # Séries et univers connus
                "harry potter", "narnia", "middle earth", "westeros", "wheel of time",
                "forgotten realms", "dragonlance", "warcraft", "warhammer fantasy"
            ],
            
            "Science Fiction": [
                # Termes généraux
                "science fiction", "sci-fi", "scifi", "space opera", "cyberpunk", 
                "steampunk", "biopunk", "dystopian", "utopian", "post-apocalyptic",
                "space exploration", "future", "futuristic", "sci fi", "science fantasy",
                
                # Technologies
                "robot", "robots", "android", "cyborg", "artificial intelligence", 
                "AI", "time travel", "time machine", "space travel", "starship", 
                "spaceship", "spacecraft", "aliens", "extraterrestrial", "alien invasion",
                "virtual reality", "nanotechnology", "genetic engineering", "clone",
                
                # Sous-genres
                "hard science fiction", "soft science fiction", "military science fiction",
                "space marine", "alternate history", "parallel universe", "dimension",
                "climate fiction", "cli-fi", "first contact", "galactic empire",
                
                # Auteurs célèbres
                "isaac asimov", "asimov", "philip k dick", "arthur c clarke", 
                "robert heinlein", "ursula k le guin", "kim stanley robinson", 
                "william gibson", "neuromancer", "frank herbert", "dune", 
                "ray bradbury", "fahrenheit 451", "foundation", "ender's game",
                "orson scott card", "douglas adams", "hitchhiker's guide",
                
                # Univers et séries
                "star trek", "star wars", "blade runner", "matrix", "terminator",
                "alien", "mars", "galaxy", "universe", "planet", "colony", 
                "terraforming", "warp drive", "hyperspace", "federation"
            ],
            
            "Mystery": [
                # Types de mystères
                "mystery", "detective", "crime", "murder mystery", "cozy mystery", 
                "police procedural", "hardboiled", "noir", "whodunit", "whodunnit",
                "locked room mystery", "cold case", "detective story", "crime fiction",
                "mystery novel", "detective novel", "mystery thriller",
                
                # Professionnels
                "private investigator", "private detective", "police detective", 
                "amateur sleuth", "sleuth", "investigator", "forensic", "forensics",
                "criminal investigation", "homicide", "murder", "serial killer",
                "crime scene", "evidence", "clues", "suspect", "alibi",
                
                # Auteurs célèbres
                "agatha christie", "christie", "hercule poirot", "miss marple",
                "arthur conan doyle", "sherlock holmes", "holmes", "watson",
                "raymond chandler", "dashiell hammett", "louise penny", "tana french",
                "gillian flynn", "gone girl", "john le carre", "patricia cornwell",
                "michael connelly", "james patterson", "harlan coben",
                
                # Sous-genres
                "cozy mystery", "british mystery", "nordic noir", "scandinavian crime",
                "psychological thriller", "legal thriller", "medical thriller",
                "historical mystery", "victorian mystery", "regency mystery"
            ],
            
            "Romance": [
                # Types généraux
                "romance", "love story", "romantic comedy", "romantic drama", 
                "romantic suspense", "romantic fiction", "love", "romance novel",
                "contemporary romance", "modern romance", "romance literature",
                
                # Sous-genres historiques
                "historical romance", "regency romance", "victorian romance",
                "medieval romance", "western romance", "highland romance",
                "scottish romance", "irish romance", "pirate romance",
                
                # Sous-genres contemporains
                "contemporary romance", "romantic comedy", "chick lit", "women's fiction",
                "new adult romance", "college romance", "workplace romance",
                "sports romance", "military romance", "medical romance",
                
                # Thèmes populaires
                "enemies to lovers", "friends to lovers", "second chance romance",
                "fake relationship", "marriage of convenience", "billionaire romance",
                "small town romance", "holiday romance", "summer romance",
                
                # Auteurs populaires
                "nicholas sparks", "nora roberts", "julia quinn", "stephanie meyer",
                "twilight", "e l james", "fifty shades", "jane austen", "pride prejudice",
                "emily henry", "colleen hoover", "christina lauren", "rainbow rowell",
                
                # Paranormal romance
                "paranormal romance", "vampire romance", "werewolf romance", 
                "shifter romance", "angel romance", "demon romance", "fae romance"
            ],
            
            "Historical Fiction": [
                # Termes généraux
                "historical fiction", "historical novel", "historical", "period drama",
                "historical saga", "historical epic", "historical adventure",
                
                # Périodes antiques
                "ancient rome", "roman empire", "ancient greece", "ancient egypt",
                "biblical fiction", "ancient world", "classical antiquity",
                
                # Moyen Âge
                "medieval", "middle ages", "medieval fiction", "crusades",
                "knights", "medieval romance", "arthurian legend",
                
                # Époques modernes
                "victorian", "victorian fiction", "edwardian", "georgian",
                "regency", "renaissance", "tudor", "elizabethan", "jacobean",
                
                # Guerres
                "world war", "world war 1", "world war 2", "wwi", "wwii",
                "civil war", "american civil war", "american revolution",
                "french revolution", "napoleonic", "vietnam war", "korean war",
                
                # Lieux et cultures
                "colonial america", "wild west", "frontier", "pioneer",
                "immigrant saga", "depression era", "prohibition",
                "viking", "samurai", "japanese historical", "chinese historical",
                
                # Auteurs célèbres
                "ken follett", "pillars of the earth", "philippa gregory",
                "hilary mantel", "wolf hall", "bernard cornwell", "sharpe",
                "james michener", "edward rutherfurd", "kate quinn",
                "kristin hannah", "nightingale", "all quiet western front"
            ],
            
            "Thriller": [
                # Types généraux
                "thriller", "suspense", "psychological thriller", "action thriller",
                "thriller novel", "suspense novel", "edge of seat",
                
                # Sous-genres
                "spy thriller", "espionage", "political thriller", "legal thriller",
                "medical thriller", "techno thriller", "military thriller",
                "conspiracy thriller", "domestic thriller", "international thriller",
                
                # Éléments
                "conspiracy", "assassin", "spy", "secret agent", "terrorism",
                "terrorist", "kidnapping", "hostage", "chase", "pursuit",
                "danger", "tension", "betrayal", "double agent", "cover up",
                
                # Auteurs populaires
                "john grisham", "dan brown", "da vinci code", "tom clancy",
                "jack ryan", "frederick forsyth", "lee child", "jack reacher",
                "james patterson", "alex cross", "harlan coben", "michael crichton",
                "robert ludlum", "jason bourne", "vince flynn", "brad thor",
                
                # Organisations
                "cia", "fbi", "mi6", "kgb", "mossad", "interpol", "nsa"
            ],
            
            "Adventure": [
                # Types d'aventure
                "adventure", "action adventure", "adventure novel", "adventure story",
                "survival", "survival story", "expedition", "exploration",
                "journey", "quest", "voyage", "travel adventure",
                
                # Environnements
                "jungle adventure", "desert adventure", "mountain climbing",
                "sea adventure", "ocean adventure", "island survival",
                "arctic expedition", "amazon", "sahara", "everest",
                
                # Thèmes
                "treasure hunt", "lost civilization", "ancient treasure",
                "explorer", "adventurer", "discovery", "archaeological",
                "treasure hunter", "pirates", "shipwreck", "castaway",
                
                # Auteurs et séries
                "indiana jones", "tomb raider", "national treasure",
                "robinson crusoe", "treasure island", "jules verne",
                "around the world", "journey center earth", "twenty thousand leagues",
                
                # Action
                "action", "high adventure", "swashbuckling", "daring rescue",
                "escape", "chase", "dangerous journey", "wilderness survival"
            ],
            
            "Horror": [
                # Types généraux
                "horror", "horror novel", "horror fiction", "scary", "frightening",
                "terrifying", "nightmare", "dark fiction", "macabre",
                
                # Sous-genres
                "supernatural horror", "psychological horror", "gothic horror",
                "cosmic horror", "lovecraftian", "body horror", "folk horror",
                "haunted house", "ghost story", "paranormal horror",
                
                # Créatures
                "vampire", "vampires", "zombie", "zombies", "ghost", "ghosts",
                "demon", "demons", "devil", "satan", "werewolf", "werewolves",
                "monster", "monsters", "creature", "beast", "undead",
                
                # Thèmes
                "supernatural", "paranormal", "occult", "witchcraft", "witch",
                "possession", "exorcism", "haunting", "haunted", "cursed",
                "evil", "darkness", "nightmare", "terror", "fear",
                
                # Auteurs célèbres
                "stephen king", "king", "it", "shining", "pet sematary",
                "clive barker", "hellraiser", "h p lovecraft", "lovecraft",
                "cthulhu", "edgar allan poe", "poe", "bram stoker", "dracula",
                "mary shelley", "frankenstein", "anne rice", "vampire chronicles",
                "dean koontz", "peter straub", "shirley jackson", "haunting hill house"
            ]
        }
        
        return variations.get(genre, [genre.lower()])
    
    def _validate_book_quality(self, book: Dict) -> Dict:
        """
        Validation de qualité ADAPTATIVE
        ← Ici aussi, filtre moins strict
        """
        validation = {
            'valid': False,
            'score': 0,
            'reasons': []
        }
        
        # Vérification des champs obligatoires
        title = book.get('title', '').strip()
        author = book.get('author_string', '').strip()
        
        if not title or len(title) < 3:
            validation['reasons'].append("Titre manquant ou trop court")
            return validation
        
        if not author or len(author) < 2:
            validation['reasons'].append("Auteur manquant ou invalide")
            return validation
        
        # Score de qualité adaptatif
        score = 25  # Base réduite de 30 à 25
        
        # Description (critère assoupli)
        description = book.get('description', '').strip()
        if description and len(description) >= self.min_description_length:
            score += 20  # Réduit de 25 à 20
        elif description and len(description) >= 5:
            score += 10  # Nouveau: même une description courte compte
        
        # Sujets/genres (critère assoupli)
        subjects = book.get('subject_string', '').strip()
        if not subjects:
            validation['reasons'].append("Aucun sujet/genre")
            return validation
        score += 15  # Réduit de 20 à 15
        
        # Couverture
        cover = book.get('cover_url', '')
        if cover and cover.startswith('http'):
            score += 10  # Réduit de 15 à 10
        
        # ISBN
        isbn = book.get('isbn', '')
        if isbn and self._validate_isbn(isbn):
            score += 10  # Maintenu à 10
        
        # Année de publication
        year = book.get('first_publish_year')
        if year and str(year).isdigit() and 1800 <= int(year) <= 2024:
            score += 5  # Nouveau critère
        
        # Seuil adaptatif basé sur la qualité générale trouvée
        validation['score'] = score
        validation['valid'] = score >= self.quality_threshold  # Seuil réduit
        
        return validation
    
    def _validate_isbn(self, isbn: str) -> bool:
        """Validation ISBN simplifiée"""
        if not isbn:
            return False
        cleaned = re.sub(r'[^0-9X]', '', str(isbn).upper())
        return len(cleaned) in [10, 13]
    
    def search_books_with_quality_control(self, query: str, limit: int) -> List[Dict]:
        """
        Recherche optimisée avec VÉRIFICATION CRITIQUE du client API
        """
        # VÉRIFICATION CRITIQUE : s'assurer que api_client n'est pas None
        if self.api_client is None:
            self._log("❌ ERREUR CRITIQUE: api_client est None dans search_books_with_quality_control", "ERROR")
            return []
        
        self.quality_stats['queries_attempted'] += 1
        
        try:
            # Recherche avec limite augmentée
            search_limit = min(limit * 3, 100)
            
            # APPEL PROTÉGÉ de la méthode search_books
            raw_books = self.api_client.search_books(query, search_limit)
            
            if not raw_books:
                return []
            
            self.quality_stats['successful_queries'] += 1
            quality_books = []
            
            for book in raw_books:
                if len(quality_books) >= limit:
                    break
                
                self.quality_stats['total_found'] += 1
                
                # Vérifier unicité
                unique_key = self._create_robust_unique_key(book)
                if unique_key in self.seen_books or unique_key in self.rejected_books:
                    self.quality_stats['duplicate_rejected'] += 1
                    continue
                
                # Enrichir via ISBN si possible
                if book.get('isbn') and self._validate_isbn(book.get('isbn', '')):
                    book = self._enrich_book_with_isbn(book)
                
                # Valider la qualité
                validation = self._validate_book_quality(book)
                
                if validation['valid']:
                    cleaned_book = self._finalize_book_data(book)
                    cleaned_book['quality_score'] = validation['score']
                    quality_books.append(cleaned_book)
                    self.seen_books.add(unique_key)
                    self.quality_stats['quality_passed'] += 1
                    
                    if self.verbose and len(quality_books) % 5 == 0:
                        self._log(f"    📚 Collectés: {len(quality_books)}/{limit}")
                else:
                    self.rejected_books.add(unique_key)
                    self.quality_stats['quality_failed'] += 1
                
                time.sleep(self.delay_between_requests)
            
            return quality_books
            
        except Exception as e:
            self._log(f"❌ Erreur recherche pour '{query}': {str(e)}", "ERROR")
            return []
    
    def _create_robust_unique_key(self, book: Dict) -> str:
        """Clé unique robuste pour déduplication"""
        title = book.get('title', '').lower().strip()
        author = book.get('author_string', '').lower().strip()
        
        # Nettoyer le titre
        title_clean = re.sub(r'\b(the|a|an)\b', '', title)
        title_clean = re.sub(r'[^\w\s]', '', title_clean)
        title_clean = re.sub(r'\s+', '', title_clean)
        
        # Premier auteur
        first_author = author.split(',')[0].strip().replace(' ', '') if author else ''
        
        return f"{title_clean}_{first_author}"
    
    def _enrich_book_with_isbn(self, book: Dict) -> Dict:
        """
        Enrichissement via ISBN - basé sur la logique qui marche dans l'API test
        """
        if not self.enable_descriptions:
            return book
    
        isbn = book.get('isbn', '').strip()
        if not isbn or not self._validate_isbn(isbn):
            return book
    
        try:
            import requests

            url = "https://openlibrary.org/api/books"
            params = {
                'bibkeys': f'ISBN:{isbn}',
                'jscmd': 'details',
                'format': 'json'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            book_data = data.get(f"ISBN:{isbn}", {}).get('details', {})

            if book_data:
                #extraction des descriptions
                description = book_data.get('description', '')
                if isinstance(description, dict) and 'value' in description:
                    description = description['value']
                elif isinstance(description, str):
                    description = description
                else:
                    description = ''
            
            #enrichissement si description trouvée
                if description and description != 'No description available':
                    book['description'] = description.strip()
                    book['isbn_enriched'] = True
                    self.quality_stats['isbn_enriched'] += 1

                    if self.verbose:
                        desc_preview = description[:50] + "..." if len(description) > 50 else description
                        self._log(f"📝 Description API: {desc_preview}")
                # Enrichir d'autres champs
                subjects = book_data.get('subjects', [])
                if subjects and not book.get('subject_string'):
                    book['subject_string'] = ', '.join(subjects[:10])
            
            time.sleep(self.delay_between_requests * 3)

        except Exception as e:
            self._log(f"❌ Erreur enrichissement ISBN {isbn}: {str(e)}", "WARNING")
    
        return book
    
    def _finalize_book_data(self, book: Dict) -> Dict:
        """Finalise les données du livre (inchangé)"""
        # ... (même code que votre version actuelle)
        return self._clean_book_data(book)
    
    def _clean_book_data(self, book: Dict) -> Dict:
        """Nettoyage des données (inchangé)"""
        return {
            'title': book.get('title', '').strip(),
            'author_string': book.get('author_string', '').strip(),
            'subject_string': book.get('subject_string', '').strip(),
            'description': book.get('description', '').strip(),
            'publisher_string': book.get('publisher_string', '').strip(),
            'first_publish_year': self._safe_int(book.get('first_publish_year')),
            'isbn': book.get('isbn', ''),
            'cover_url': book.get('cover_url', ''),
            'key': book.get('key', ''),
            'isbn_enriched': book.get('isbn_enriched', False),
            'quality_score': book.get('quality_score', 0)
        }
    
    def _safe_int(self, value) -> int:
        """Conversion sécurisée en int"""
        try:
            if value is None:
                return 2000
            return int(value)
        except (ValueError, TypeError):
            return 2000
    
    def build_improved_dataset(self, progress_callback=None):
        """
        Construction de dataset avec toutes les améliorations
        """
        self._log("🚀 Construction avec améliorations activées")
        
        total_books_target = sum(self.target_genres.values())
        current_progress = 0
        all_collected_books = []
        
        # Réinitialiser
        self.seen_books.clear()
        self.rejected_books.clear()
        self.quality_stats = self._init_quality_stats()
        
        for genre, target_count in self.target_genres.items():
            self._log(f"🎯 Genre: {genre} (objectif: {target_count})")
            
            books_collected = 0
            search_queries = self._get_enhanced_search_variations(genre)
            random.shuffle(search_queries)
            
            attempts = 0
            query_index = 0
            rotation_count = 0
            
            while (books_collected < target_count and 
                   attempts < self.max_attempts_per_genre and
                   rotation_count < self.query_rotation_limit):
                
                if query_index >= len(search_queries):
                    query_index = 0
                    rotation_count += 1
                    random.shuffle(search_queries)
                    self._log(f"    🔄 Rotation {rotation_count}/3 des requêtes")
                
                query = search_queries[query_index]
                query_index += 1
                
                try:
                    needed = min(self.batch_size, target_count - books_collected)
                    
                    # Recherche avec contrôle qualité amélioré
                    quality_books = self.search_books_with_quality_control(query, needed)
                    
                    if len(quality_books) < self.min_books_per_query:
                        attempts += 1
                        continue
                    
                    # Ajouter les livres
                    all_collected_books.extend(quality_books)
                    books_collected += len(quality_books)
                    current_progress += len(quality_books)
                    
                    # Callback de progrès
                    if progress_callback:
                        progress_percentage = current_progress / total_books_target
                        progress_callback(progress_percentage, 
                                        f"{genre}: {books_collected}/{target_count}")
                    
                    self._log(f"    📊 {genre}: {books_collected}/{target_count} "
                             f"({(books_collected/target_count)*100:.1f}%)")
                    
                    attempts += 1
                    
                except Exception as e:
                    self._log(f"Erreur {genre}: {str(e)}", "WARNING")
                    attempts += 1
                    time.sleep(1)
                    continue
            
            # Rapport par genre
            success_rate = (books_collected / target_count) * 100
            if success_rate >= 70:  # Seuil réduit de 80 à 70
                self._log(f"✅ {genre}: {books_collected}/{target_count} "
                         f"({success_rate:.1f}%)")
            else:
                self._log(f"⚠️ {genre}: {books_collected}/{target_count} "
                         f"({success_rate:.1f}%)")
        
        # Rapport final amélioré
        self._log("\n📊 RAPPORT FINAL AMÉLIORÉ:")
        self._log(f"   🎯 Objectif: {total_books_target:,} livres")
        self._log(f"   ✅ Obtenus: {len(all_collected_books):,} livres")
        self._log(f"   📈 Taux global: {(len(all_collected_books)/total_books_target)*100:.1f}%")
        self._log(f"   🔍 Requêtes tentées: {self.quality_stats['queries_attempted']}")
        self._log(f"   ✅ Requêtes réussies: {self.quality_stats['successful_queries']}")
        
        return {
            'total_books': len(all_collected_books),
            'books_data': all_collected_books,
            'quality_stats': self.quality_stats,
            'success_rate': (len(all_collected_books) / total_books_target) * 100
        }