# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false
# type: ignore

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st
from api_utils import get_api_client
from typing import List, Dict, Tuple

import mlflow
import mlflow.sklearn
import time


class DynamicKNNRecommender:
    """
    Système de recommandation KNN dynamique utilisant l'API Open Library
    """
    
    def __init__(self):
        self.api_client = get_api_client()
        
        # Vectorizers pour les features textuelles
        self.author_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=50,
            ngram_range=(1, 2)
        )
        
        self.subject_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=300,
            ngram_range=(1, 2)
        )

        self.description_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=200, 
            ngram_range=(1, 2),
            min_df=2 
        )
        
        self.publisher_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=30,
            ngram_range=(1, 1)
        )
        
        # Scaler pour les features numériques
        self.numeric_scaler = StandardScaler()
        
        # Dataset de livres en mémoire
        self.books_data = []
        self.feature_matrix = None
        self.knn_model = None
        
        # Indicateur si le modèle est entraîné
        self.is_fitted = False
    
    def search_and_add_books(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Recherche des livres via l'API et les ajoute au dataset avec déduplication
        """
        try:
            # Rechercher via l'API
            books = self.api_client.search_books(query, limit)
            
            added_books = []
            # Créer un set des livres existants pour déduplication
            existing_books = {
                f"{book['title'].lower().strip()}_{book['author_string'].lower().strip()}" 
                for book in self.books_data
            }
            
            for book in books:
                # Vérifier qu'on a les données minimales
                if (book.get('title') and 
                    book.get('author_string') and 
                    book.get('subject_string')):
                    
                    # Créer une clé unique pour déduplication
                    title = book.get('title', '').lower().strip()
                    author = book.get('author_string', '').lower().strip()
                    unique_key = f"{title}_{author}"
                    
                    # Vérifier si le livre n'existe pas déjà
                    if unique_key not in existing_books:
                        # Nettoyer et enrichir les données
                        cleaned_book = self._clean_book_data(book)
                        self.books_data.append(cleaned_book)
                        added_books.append(cleaned_book)
                        existing_books.add(unique_key)
            
            # Réentraîner le modèle avec les nouvelles données
            if len(self.books_data) >= 5:
                self._fit_model()
            
            return added_books
            
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {str(e)}")
            return []
    
    def _clean_book_data(self, book: Dict) -> Dict:
        """
        Nettoie et enrichit les données d'un livre
        """
        # Récupérer la description depuis l'API si disponible
        description = ""
        if 'description' in book and book['description']:
            description = str(book['description'])
        elif 'subject_string' in book:
            # Utiliser les sujets comme description de fallback
            description = book['subject_string']

        return {
            'title': book.get('title', '').strip(),
            'author_string': book.get('author_string', '').strip(),
            'subject_string': book.get('subject_string', '').strip(),
            'description': description.strip(),
            'publisher_string': ', '.join(book.get('publishers', [])),
            'first_publish_year': self._safe_int(book.get('first_publish_year')),
            'isbn': book.get('isbn', ''),
            'cover_url': book.get('cover_url', ''),
            'key': book.get('key', '')
        }
    
    def _safe_int(self, value) -> int:
        """Convertit une valeur en int de manière sécurisée"""
        try:
            if value is None:
                return 2000
            return int(value)
        except (ValueError, TypeError):
            return 2000
    
    def _fit_model(self):
        """
        Entraîne le modèle KNN avec les données actuelles + MLflow tracking
        """
        try:
            if len(self.books_data) < 5:
                return
        
        # ✨ NOUVEAU : Configuration MLflow pour KNN
            mlflow.set_experiment("Book_Recommender_KNN")
        
            with mlflow.start_run(run_name="KNN_Training"):
                print("📊 KNN MLflow tracking started...")
            
                # ✨ Mesurer le temps total d'entraînement
                start_time = time.time()
            
                # Préparer les données (TON CODE EXISTANT)
                df = pd.DataFrame(self.books_data)
            
                # ✨ Log des paramètres du dataset
                dataset_params = {
                    "total_books": len(self.books_data),
                    "unique_authors": df['author_string'].nunique(),
                    "avg_description_length": df['description'].str.len().mean(),
                    "books_with_subjects": (df['subject_string'] != '').sum()
                }
                mlflow.log_params(dataset_params)
            
                # ✨ Paramètres des vectorizers (TON CODE mais organisé)
                vectorizer_params = {
                    "author_max_features": 50,
                    "subject_max_features": 300,
                    "description_max_features": 200,
                    "publisher_max_features": 30,
                    "knn_neighbors": min(10, len(self.books_data) - 1),
                    "knn_metric": "cosine",
                    "knn_algorithm": "brute"
                }
                mlflow.log_params(vectorizer_params)
            
                # Features textuelles (TON CODE EXISTANT)
                vectorization_start = time.time()
            
                author_sparse = self.author_vectorizer.fit_transform(df['author_string'])
                subject_sparse = self.subject_vectorizer.fit_transform(df['subject_string'])
                publisher_sparse = self.publisher_vectorizer.fit_transform(df['publisher_string'])
                description_sparse = self.description_vectorizer.fit_transform(df['description'])
            
                vectorization_time = time.time() - vectorization_start
            
                # Convertir en arrays denses (TON CODE EXISTANT)
                author_features = author_sparse.toarray() 
                subject_features = subject_sparse.toarray()  
                publisher_features = publisher_sparse.toarray()
                description_features = description_sparse.toarray()
            
                # Features numériques (TON CODE EXISTANT)
                years = df['first_publish_year'].fillna(2000).astype(int)
                numeric_features = self.numeric_scaler.fit_transform(years.values.reshape(-1, 1))
            
                # Combiner toutes les features (TON CODE EXISTANT)
                feature_combination_start = time.time()
            
                self.feature_matrix = np.hstack([
                    author_features,
                    subject_features, 
                    publisher_features,
                    description_features,
                    numeric_features
                ])
            
                feature_combination_time = time.time() - feature_combination_start
            
                # Entraîner le modèle KNN (TON CODE EXISTANT)
                knn_training_start = time.time()
            
                self.knn_model = NearestNeighbors(
                    n_neighbors=vectorizer_params["knn_neighbors"],
                    metric=vectorizer_params["knn_metric"],
                    algorithm=vectorizer_params["knn_algorithm"]
                )   
            
                self.knn_model.fit(self.feature_matrix)
                self.is_fitted = True
            
                knn_training_time = time.time() - knn_training_start
                total_training_time = time.time() - start_time
            
                # ✨ NOUVEAU : Log des métriques de performance
                performance_metrics = {
                    "total_training_time": total_training_time,
                    "vectorization_time": vectorization_time,
                    "feature_combination_time": feature_combination_time,
                    "knn_fitting_time": knn_training_time,
                    "feature_matrix_shape_rows": self.feature_matrix.shape[0],
                    "feature_matrix_shape_cols": self.feature_matrix.shape[1],
                    "feature_density": np.count_nonzero(self.feature_matrix) / self.feature_matrix.size
                }
                mlflow.log_metrics(performance_metrics)
            
                # ✨ NOUVEAU : Log des métriques de qualité des features
                feature_quality = {
                    "author_vocab_size": len(self.author_vectorizer.vocabulary_),
                    "subject_vocab_size": len(self.subject_vectorizer.vocabulary_),
                    "description_vocab_size": len(self.description_vectorizer.vocabulary_),
                    "publisher_vocab_size": len(self.publisher_vectorizer.vocabulary_),
                    "avg_year": float(years.mean()),
                    "year_std": float(years.std())
                }
                mlflow.log_metrics(feature_quality)
            
                print(f"✅ KNN model trained in {total_training_time:.2f}s")
                print(f"📊 Feature matrix: {self.feature_matrix.shape}")
                print(f"📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            self.is_fitted = False
    
    def get_recommendations(self, book_title: str, n_recommendations: int = 5) -> List[Dict]:
        """
        Obtient des recommandations pour un livre donné
        """
        if not self.is_fitted:
            return []
        
        try:
            # Trouver le livre dans notre dataset
            book_idx = None
            for idx, book in enumerate(self.books_data):
                if book_title.lower() in book['title'].lower():
                    book_idx = idx
                    break
            
            if book_idx is None:
                return []
            
            # Obtenir les recommandations avec gestion des types
            distances, indices = self.knn_model.kneighbors(  # type: ignore
                [self.feature_matrix[book_idx]], 
                n_neighbors=min(n_recommendations + 1, len(self.books_data))
            )
            
            recommendations = []
            for i, idx in enumerate(indices[0]):
                if idx != book_idx:
                    book = self.books_data[idx].copy()
                    book['similarity_score'] = 1 - distances[0][i]
                    recommendations.append(book)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            st.error(f"Erreur lors des recommandations: {str(e)}")
            return []
    
    def get_dataset_stats(self) -> Dict:
        """Retourne des statistiques sur le dataset actuel"""
        if not self.books_data:
            return {"total_books": 0, "unique_authors": 0, "year_range": "N/A", "is_model_fitted": False}
        
        try:
            df = pd.DataFrame(self.books_data)
            return {
                "total_books": len(self.books_data),
                "unique_authors": df['author_string'].nunique(),
                "year_range": f"{df['first_publish_year'].min()}-{df['first_publish_year'].max()}",
                "is_model_fitted": self.is_fitted
            }
        except Exception as e:
            return {"total_books": len(self.books_data), "unique_authors": 0, "year_range": "N/A", "is_model_fitted": self.is_fitted}


# Fonctions utilitaires pour Streamlit
@st.cache_resource
def get_dynamic_knn():
    """Crée une instance cachée du recommandeur dynamique"""
    return DynamicKNNRecommender()

def search_and_recommend_streamlit(query: str, target_book: str = None, limit: int = 20) -> Tuple[List[Dict], List[Dict]]:  # type: ignore
    """
    Fonction helper pour Streamlit : recherche et recommande
    """
    try:
        knn = get_dynamic_knn()
        
        # Alimenter le dataset
        found_books = knn.search_and_add_books(query, limit)
        
        # Obtenir des recommandations si un livre cible est spécifié
        recommendations = []
        if target_book and knn.is_fitted:
            recommendations = knn.get_recommendations(target_book)
        
        return found_books, recommendations
        
    except Exception as e:
        st.error(f"Erreur dans search_and_recommend_streamlit: {str(e)}")
        return [], []