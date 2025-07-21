import streamlit as st
import pandas as pd
from utils import render_aligned_image

def show_popular_books(books_df):
    """Affiche l'onglet des livres populaires avec améliorations."""
    st.subheader("📈 Livres populaires")
    
    # AMÉLIORATION 1: Validation des données
    if books_df.empty:
        st.warning("⚠️ Aucune donnée de livre disponible")
        return
    
    if 'Rating-Count' not in books_df.columns:
        st.error("❌ Colonne Rating-Count non trouvée dans le dataset")
        return
    
    # AMÉLIORATION 2: Options de personalisation pour l'utilisateur
    col1, col2 = st.columns(2)
    with col1:
        num_books = st.selectbox(
            "Nombre de livres à afficher :",
            options=[5, 10, 15, 20],
            key="tab5_num_books"  # 10 par défaut
        )
    
    with col2:
        st.info("📊 Classement basé sur le **nombre total d'évaluations** reçues")
    
        # Tri par nombre de ratings (code original)
    popular_books = (
        books_df.sort_values(by="Rating-Count", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(num_books)
    )
    
    #Vérification qu'on a des résultats
    if popular_books.empty:
        st.warning("⚠️ Aucun livre trouvé correspondant aux critères")
        return
    
    # Information pour l'utilisateur
    st.info(f"📊 Affichage du top {len(popular_books)} livres les plus populaires")
    
    #  Layout responsive
    # Calcul dynamique du nombre de colonnes selon le nombre de livres
    if len(popular_books) <= 3:
        num_cols = len(popular_books)
    elif len(popular_books) <= 5:
        num_cols = min(5, len(popular_books))
    else:
        # Pour plus de 5 livres, afficher sur plusieurs lignes
        num_cols = 5
    
    # Créer les colonnes
    cols = st.columns(num_cols)
    
    # Affichage avec gestion d'erreurs
    for idx in range(len(popular_books)):
        col_idx = idx % num_cols  # Pour gérer les lignes multiples
        
        # Extraire les données avec gestion d'erreurs
        try:
            book_title = popular_books.iloc[idx]["Book-Title"]
            rating_count = popular_books.iloc[idx]["Rating-Count"]
            
            # AMÉLIORATION 8: Priorité aux images API puis fallback
            api_image = popular_books.iloc[idx].get("api_cover_url")
            original_image = popular_books.iloc[idx].get("Image-URL-L")
            
            if pd.notna(api_image) and api_image and api_image.startswith('http'):
                book_image = api_image
            elif pd.notna(original_image) and original_image and original_image.startswith('http'):
                book_image = original_image
            else:
                book_image = "https://via.placeholder.com/300x400/cccccc/666666?text=Pas+de+Couverture"
            
            #Titre tronqué pour l'affichage
            display_title = book_title if len(book_title) <= 40 else book_title[:37] + "..."
    
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du livre {idx + 1}: {str(e)}")
            continue
        
        # Affichage dans la colonne appropriée
        with cols[col_idx]:
            # Image avec titre
            cols[col_idx].markdown(
                render_aligned_image(book_image, display_title),
                unsafe_allow_html=True
            )
            
            # Métrique sous l'image
            cols[col_idx].metric(
                label="Popularité",
                value=f"📊 {rating_count} évaluations",
                help=f"Titre complet: {book_title}"
            )
            
# AMÉLIORATION 11: Badge de popularité
            if rating_count >= 200:
                cols[col_idx].markdown("🔥 **Très populaire**", help="Plus de 200 évaluations")
            elif rating_count >= 100:
                cols[col_idx].markdown("📈 **Populaire**", help="Plus de 100 évaluations")
            elif rating_count >= 50:
                cols[col_idx].markdown("👍 **Apprécié**", help="Plus de 50 évaluations")
            else:
                cols[col_idx].markdown("📚 **Découverte**", help="Livre en cours de découverte")
            
            # Bouton d'action
            if st.button(f"📖 Détails", key=f"details_popular_{idx}"):
                # Calculer la note moyenne si disponible
                if 'Book-Rating' in books_df.columns:
                    book_ratings = books_df[books_df['Book-Title'] == book_title]['Book-Rating']
                    avg_rating = book_ratings.mean()
                    st.info(f"📚 **{book_title}**\n\n📊 **{rating_count} évaluations**\n⭐ **Note moyenne: {avg_rating:.1f}/10**")
                else:
                    st.info(f"📚 **{book_title}**\n\n📊 **{rating_count} évaluations**")
    
    # AMÉLIORATION 12: Statistiques spécialisées pour la popularité
    with st.expander("📊 Statistiques de popularité"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_evaluations = popular_books['Rating-Count'].sum()
            st.metric(
                "Total évaluations",
                f"{total_evaluations:,}",
                help="Somme des évaluations de tous les livres affichés"
            )
        
        with col2:
            avg_popularity = popular_books['Rating-Count'].mean()
            st.metric(
                "Popularité moyenne",
                f"{avg_popularity:.0f}",
                help="Nombre moyen d'évaluations par livre affiché"
            )
        
        with col3:
            max_popularity = popular_books['Rating-Count'].max()
            st.metric(
                "Plus populaire",
                f"{max_popularity} évaluations",
                help="Livre avec le plus d'évaluations"
            )
        
        # Graphique de distribution de popularité
        if len(popular_books) > 3:
            st.subheader("📈 Distribution de popularité")
            popularity_data = popular_books.set_index('Book-Title')['Rating-Count'].head(10)
            st.bar_chart(popularity_data)
