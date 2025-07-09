import requests
import time
import streamlit as st
from typing import Dict, List, Optional
import pandas as pd

class OpenLibraryAPI:
    """Classe pour gérer les appels à l'API Open Library"""

    def __init__(self):
        self.base_url = "https://openlibrary.org"
        self.session = requests.Session()
        #ajout d'un délai entre les requests
        self.request_delay = 0.1
    
    def get_book_by_isbn(self, isbn:str) -> Optional[Dict]:
        """
        Récupère les informations d'un livre par son ISBN
        
        Args:
            isbn (str): Le numéro ISBN du livre
            
        Returns:
            Dict: Dictionnaire contenant les infos du livre ou None si non trouvé
        """
        try:
            #Nettoyage de l'ISBN (enlever tirets et espaces)
            clean_isbn = isbn.replace("-", "").replace(" ", "")

            #Url de l'API avec jscmd=data pour avoir toutes les infos
            url = f"{self.base_url}/api/books"
            params = {
                'bibkeys': f'ISBN:{clean_isbn}',
                'jscmd': 'data',
                'format':'json'
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            book_key = f'ISBN:{clean_isbn}'

            if book_key in data:
                return self._parse_book_data(data[book_key])
            else:
                return None
        
        except Exception as e:
            st.error(f"Erreur lors de la récupération du livre {isbn}: {str(e)}")
            return None
        
        finally:
            # Respecter le délai entre requêtes
            time.sleep(self.request_delay)

    def search_books(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Recherche des livres par titre, auteur, etc.
        
        Args:
            query (str): Terme de recherche
            limit (int): Nombre max de résultats
            
        Returns:
            List[Dict]: Liste des livres trouvés
        """
        try:
            url = f"{self.base_url}/search.json"
            params = {
                'q': query,
                'limit': limit,
                'fields': 'key,title,author_name,first_publish_year,subject,isbn,cover_i,publisher'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            books = []
            
            for doc in data.get('docs', []):
                book = self._parse_search_result(doc)
                if book:
                    books.append(book)
            
            return books
            
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {str(e)}")
            return []
        
        finally:
            time.sleep(self.request_delay)
    
    def _parse_book_data(self, book_data: Dict) -> Dict:
        """
        Parse les données détaillées d'un livre (API books)
        
        Args:
            book_data (Dict): Données brutes de l'API
            
        Returns:
            Dict: Données formatées pour notre système
        """
        # Extraire les auteurs
        authors = []
        if 'authors' in book_data:
            authors = [author.get('name', '') for author in book_data['authors']]
        
        # Extraire les sujets/genres
        subjects = []
        if 'subjects' in book_data:
            subjects = [subject.get('name', '') for subject in book_data['subjects']]
        
        # Extraire les éditeurs
        publishers = []
        if 'publishers' in book_data:
            publishers = [pub.get('name', '') for pub in book_data['publishers']]
        
        # Classification Dewey (très utile pour le KNN)
        dewey_classes = []
        if 'classifications' in book_data and 'dewey_decimal_class' in book_data['classifications']:
            dewey_classes = book_data['classifications']['dewey_decimal_class']
        
        # URLs des couvertures
        cover_urls = {}
        if 'cover' in book_data:
            cover_urls = book_data['cover']
        
        # Construire l'objet livre formaté
        formatted_book = {
            'title': book_data.get('title', ''),
            'authors': authors,
            'author_string': ', '.join(authors),  # Pour TF-IDF
            'subjects': subjects,
            'subject_string': ', '.join(subjects),  # Pour TF-IDF
            'publishers': publishers,
            'publisher_string': ', '.join(publishers),
            'publish_date': book_data.get('publish_date', ''),
            'number_of_pages': book_data.get('number_of_pages', 0),
            'dewey_classes': dewey_classes,
            'dewey_string': ', '.join(dewey_classes),
            'description': book_data.get('description', ''),
            'cover_urls': cover_urls,
            'url': book_data.get('url', ''),
            'identifiers': book_data.get('identifiers', {}),
        }
        
        return formatted_book
    
    def _parse_search_result(self, doc: Dict) -> Dict:
        """
        Parse les résultats de recherche
        
        Args:
            doc (Dict): Document de résultat de recherche
            
        Returns:
            Dict: Données formatées
        """
        # Récupérer le premier ISBN disponible
        isbn = None
        if 'isbn' in doc and doc['isbn']:
            isbn = doc['isbn'][0]
        
        # URL de la couverture
        cover_url = None
        if 'cover_i' in doc:
            cover_url = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-L.jpg"
        
        formatted_book = {
            'title': doc.get('title', ''),
            'authors': doc.get('author_name', []),
            'author_string': ', '.join(doc.get('author_name', [])),
            'subjects': doc.get('subject', [])[:10],  # Limiter à 10 sujets
            'subject_string': ', '.join(doc.get('subject', [])[:10]),
            'first_publish_year': doc.get('first_publish_year'),
            'isbn': isbn,
            'cover_url': cover_url,
            'publishers': doc.get('publisher', []),
            'key': doc.get('key', ''),
        }
        
        return formatted_book

# Fonction utilitaire pour créer une instance cachée
@st.cache_resource
def get_api_client():
    """Crée une instance cachée du client API"""
    return OpenLibraryAPI()

# Fonctions helper pour l'utilisation dans Streamlit
def search_books_streamlit(query: str, limit: int = 10) -> pd.DataFrame:
    """
    Recherche des livres et retourne un DataFrame pandas
    
    Args:
        query (str): Terme de recherche
        limit (int): Nombre de résultats
        
    Returns:
        pd.DataFrame: DataFrame contenant les livres
    """
    api = get_api_client()
    books = api.search_books(query, limit)
    
    if books:
        return pd.DataFrame(books)
    else:
        return pd.DataFrame()

def get_book_details_streamlit(isbn: str) -> Optional[Dict]:
    """
    Récupère les détails d'un livre par ISBN pour Streamlit
    
    Args:
        isbn (str): ISBN du livre
        
    Returns:
        Dict: Informations détaillées du livre
    """
    api = get_api_client()
    return api.get_book_by_isbn(isbn)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test de l'API
    api = OpenLibraryAPI()
    
    # Test recherche
    print("=== Test de recherche ===")
    results = api.search_books("harry potter", limit=3)
    for book in results:
        print(f"- {book['title']} par {book['author_string']}")
    
    # Test récupération par ISBN
    print("\n=== Test récupération par ISBN ===")
    book = api.get_book_by_isbn("9780747532699")  # Harry Potter
    if book:
        print(f"Titre: {book['title']}")
        print(f"Auteurs: {book['author_string']}")
        print(f"Sujets: {book['subject_string']}")