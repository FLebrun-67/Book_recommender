import pickle
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

@st.cache_resource
def load_model_and_data():
    """Load the pre-trained model and enriched data."""
    book_titles = pickle.load(open("artifacts/book_titles.pkl", "rb"))
    books_df = pickle.load(open("artifacts/book_df.pkl", "rb"))  # Dataset enrichi
    svd_model = pickle.load(open('artifacts/svd_model.pkl', 'rb'))

    return None, book_titles, books_df, svd_model, None, None

def fetch_poster(books_df, book_list):
    """Fetch poster URLs prioritizing API cover URLs."""
    book_names = []
    poster_urls = []
    book_descriptions = []
    default_image = "https://via.placeholder.com/150"

    for book in book_list:
        if book in books_df["Book-Title"].values:
            book_names.append(book)
            idx = books_df[books_df["Book-Title"] == book].index
            if len(idx) > 0:
                # PRIORITÉ: utiliser api_cover_url si disponible
                api_cover = books_df.loc[idx[0], "api_cover_url"] if "api_cover_url" in books_df.columns else None
                fallback_cover = books_df.loc[idx[0], "Image-URL-L"]
                
                if pd.notna(api_cover) and api_cover and api_cover.startswith("http"):
                    img_url = api_cover
                elif pd.notna(fallback_cover) and fallback_cover.startswith("http"):
                    img_url = fallback_cover
                else:
                    img_url = default_image
                
                poster_urls.append(img_url)
                
                # Description
                description = books_df.loc[idx[0], "Description"]
                book_descriptions.append(
                    description if pd.notna(description) else "Description non disponible"
                )
            else:
                poster_urls.append(default_image)
                book_descriptions.append("Description non disponible")
        else:
            book_names.append(book)
            poster_urls.append(default_image)
            book_descriptions.append("Description non disponible")

    return book_names, poster_urls, book_descriptions

def get_combined_genres(row) -> List[str]:
    """
    Extrait les genres combinés (API + fallback)
    """
    all_genres = []
    
    # Priorité 1: subject_string_final (API + fallback)
    if pd.notna(row.get('subject_string_final', '')) and row.get('subject_string_final', '') != '':
        tags = [tag.strip() for tag in str(row['subject_string_final']).split(',')]
        all_genres.extend(tags)
    
    # Priorité 2: Final_Tags originaux si rien trouvé
    elif pd.notna(row.get('Final_Tags', '')) and row.get('Final_Tags', '') != '':
        tags = [tag.strip() for tag in str(row['Final_Tags']).split(',')]
        all_genres.extend(tags)
    
    # Nettoyer et dédupliquer
    clean_genres = []
    for genre in all_genres:
        if genre and len(genre) > 2:  # Ignorer les tags trop courts
            clean_genres.append(genre.lower())
    
    return list(set(clean_genres))

def extract_user_preferences(user_id: str, books_df: pd.DataFrame) -> Dict:
    """
    Extrait les préférences utilisateur avec données enrichies
    """
    user_books = books_df[
        (books_df['User-ID'] == user_id) & 
        (books_df['Book-Rating'] >= 7)
    ]
    
    if user_books.empty:
        return {
            'preferred_genres': [],
            'preferred_year_range': (1990, 2024),
            'known_publishers': [],
            'avg_rating': 8.0
        }
    
    # Extraire tous les genres enrichis
    all_genres = []
    for _, book in user_books.iterrows():
        book_genres = get_combined_genres(book)
        all_genres.extend(book_genres)
    
    # Top genres
    genre_counts = Counter(all_genres)
    preferred_genres = [genre for genre, count in genre_counts.most_common(8)]
    
    # Années préférées (utiliser API si disponible)
    years = []
    if 'api_first_publish_year' in user_books.columns:
        api_years = user_books['api_first_publish_year'].dropna()
        years.extend(api_years.tolist())
    
    if years:
        min_year = max(1980, int(min(years)) - 5)
        max_year = min(2024, int(max(years)) + 5)
    else:
        min_year, max_year = 1990, 2024
    
    # Éditeurs connus
    known_publishers = []
    if 'api_publisher_string' in user_books.columns:
        api_pubs = user_books['api_publisher_string'].dropna().unique()
        known_publishers.extend(api_pubs.tolist())
    
    avg_rating = user_books['Book-Rating'].mean()
    
    return {
        'preferred_genres': preferred_genres,
        'preferred_year_range': (min_year, max_year),
        'known_publishers': known_publishers,
        'avg_rating': avg_rating
    }

def get_genre_similarity_score(user_genres: List[str], book_row) -> float:
    """
    Calcule le score de similarité de genres
    """
    if not user_genres:
        return 0.5  # Score neutre
    
    book_genres = get_combined_genres(book_row)
    
    if not book_genres:
        return 0.3  # Pénalité si pas de genres
    
    # Calcul de similarité
    matches = 0
    partial_matches = 0
    
    for user_genre in user_genres:
        for book_genre in book_genres:
            if user_genre == book_genre:
                matches += 2  # Match parfait
            elif user_genre in book_genre or book_genre in user_genre:
                partial_matches += 1  # Match partiel
    
    # Score final
    total_score = matches + (partial_matches * 0.5)
    max_possible = len(user_genres) * 2
    
    return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0

def get_year_relevance_score(target_year_range: Tuple[int, int], book_row) -> float:
    """
    Score basé sur l'année de publication
    """
    book_year = book_row.get('api_first_publish_year')
    
    if pd.isna(book_year):
        return 0.5  # Score neutre si pas d'info
    
    min_year, max_year = target_year_range
    book_year = int(book_year)
    
    if min_year <= book_year <= max_year:
        return 1.0  # Dans la plage préférée
    else:
        # Score décroissant selon la distance
        distance = min(abs(book_year - min_year), abs(book_year - max_year))
        return max(0.2, 1.0 - (distance / 30))

def get_publisher_diversity_score(known_publishers: List[str], book_row) -> float:
    """
    Score basé sur la diversité des éditeurs
    """
    book_publisher = book_row.get('api_publisher_string')
    
    if pd.isna(book_publisher) or not book_publisher:
        return 0.5  # Score neutre
    
    # Si l'utilisateur n'a pas encore lu cet éditeur = bonus diversité
    publisher_lower = str(book_publisher).lower()
    for known_pub in known_publishers:
        if str(known_pub).lower() in publisher_lower or publisher_lower in str(known_pub).lower():
            return 0.3  # Éditeur connu, score plus faible
    
    return 0.8  # Nouvel éditeur, bonus de diversité

def recommend_book_svd_hybrid(user_id, books_df, svd_model, n_recommendations=10):
    """
    Recommandations SVD hybrides avec métadonnées enrichies
    """
    print(f"🎯 Génération recommandations hybrides pour {user_id}")
    
    # 1. Extraire les préférences utilisateur
    user_prefs = extract_user_preferences(user_id, books_df)
    
    # 2. Obtenir les livres candidats (non lus)
    all_books = books_df["Book-Title"].unique()
    user_books = books_df[books_df["User-ID"] == user_id]["Book-Title"].unique()
    candidate_books = [book for book in all_books if book not in user_books]
    
    if not candidate_books:
        # Fallback sur les livres populaires
        popular_books = (
            books_df.groupby("Book-Title")["Book-Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n_recommendations)
        )
        book_names, poster_urls, book_descriptions = fetch_poster(books_df, popular_books.index.tolist())
        return list(zip(book_names, popular_books.values, poster_urls, book_descriptions))
    
    # 3. Calculer les scores hybrides
    recommendations = []
    
    # Limiter pour la performance (prendre les 1000 premiers candidats)
    for book in candidate_books[:1000]:
        # Score SVD
        svd_prediction = svd_model.predict(user_id, book)
        svd_score = svd_prediction.est / 10.0  # Normaliser sur [0,1]
        
        # Récupérer les métadonnées du livre
        book_info = books_df[books_df["Book-Title"] == book].iloc[0]
        
        # Scores métadonnées
        genre_score = get_genre_similarity_score(user_prefs['preferred_genres'], book_info)
        year_score = get_year_relevance_score(user_prefs['preferred_year_range'], book_info)
        publisher_score = get_publisher_diversity_score(user_prefs['known_publishers'], book_info)
        
        # Score final hybride (pondération optimisée)
        final_score = (
            0.60 * svd_score +          # 60% SVD (base)
            0.25 * genre_score +        # 25% Genres
            0.10 * year_score +         # 10% Année
            0.05 * publisher_score      # 5% Diversité éditeur
        )
        
        recommendations.append((book, final_score))
    
    # 4. Trier et prendre les meilleures recommandations
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_books = [book for book, score in recommendations[:n_recommendations]]
    
    # 5. Récupérer les métadonnées d'affichage
    book_names, poster_urls, book_descriptions = fetch_poster(books_df, top_books)
    scores = [score for book, score in recommendations[:n_recommendations]]
    
    return list(zip(book_names, scores, poster_urls, book_descriptions))

# Garder la fonction SVD classique comme fallback
def recommend_book_svd(user_id, books_df, svd_model, n_recommendations=10):
    """Version classique SVD (fallback)"""
    if user_id not in books_df["User-ID"].unique():
        popular_books = (
            books_df.groupby("Book-Title")["Book-Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n_recommendations)
        )
        book_name, poster_url, book_descriptions = fetch_poster(books_df, popular_books.index.tolist())
        return list(zip(book_name, popular_books.values, poster_url, book_descriptions))

    all_books = books_df["Book-Title"].unique()
    user_books = books_df[books_df["User-ID"] == user_id]["Book-Title"].unique()
    books_to_predict = [book for book in all_books if book not in user_books]
    
    if not books_to_predict:
        st.warning("No books to find for this user.")
        return []

    predictions = [
        (book, svd_model.predict(user_id, book).est) for book in books_to_predict
    ]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    book_name, poster_url, book_descriptions = fetch_poster(books_df, [book for book, _ in predictions])
    return list(zip(book_name, [rating for _, rating in predictions], poster_url, book_descriptions))

def render_aligned_image(image_url, title, height=500):
    """Render an image with the given title and height."""
    return f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="{title}" style="height: {height}px; object-fit: contain; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 5px; font-size: 14px;"><b>{title}</b></p>
    </div>
    """