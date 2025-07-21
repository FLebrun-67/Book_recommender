import streamlit as st
import pandas as pd
from utils import render_aligned_image

def show_top_rated_books(books_df):
    """Affiche l'onglet des livres les mieux notés avec améliorations."""
    st.subheader("⭐ Livres les mieux notés")
    
    # Validation des données
    if books_df.empty:
        st.warning("⚠️ Aucune donnée de livre disponible")
        return
    
    if 'Book-Rating' not in books_df.columns:
        st.error("❌ Colonne Book-Rating non trouvée dans le dataset")
        return
    
    # Options de personnalisation pour l'utilisateur
    col1, col2 = st.columns(2)
    with col1:
        num_books = st.selectbox(
            "Nombre de livres à afficher :",
            options=[5, 10, 15, 20],
            index=1,  # 10 par défaut
            key="tab6_num_books"  
        )
    
    with col2:
        rating_method = st.radio(
            "Méthode de sélection :",
            options=["Note la plus élevée", "Note pondérée (min 5 évaluations)"],
            index=1,  # Note pondérée par défaut (plus fiable)
            help="La note pondérée évite les biais des livres avec peu d'évaluations",
            key="tab6_rating_method"
        )
    
    # Logique de tri améliorée
    if rating_method == "Note la plus élevée":
        # Méthode simple : tri direct par Book-Rating
        top_rated_books = (
            books_df.sort_values(by="Book-Rating", ascending=False)
            .drop_duplicates(subset="Book-Title")
            .head(num_books)
        )
        
    else:
        # Note pondérée pour éviter les biais
        if 'Rating-Count' not in books_df.columns:
            st.error("❌ Colonne Rating-Count nécessaire pour la note pondérée")
            return
        
        # Calculer les statistiques par livre
        book_stats = (
            books_df.groupby('Book-Title')
            .agg({
                'Book-Rating': 'mean',           # Note moyenne
                'Rating-Count': 'first',         # Nombre d'évaluations
                'Image-URL-L': 'first',          # Image originale
                'api_cover_url': 'first'         # Image API
            })
            .reset_index()
        )
        
        #Minimum 5 évaluations pour être fiable
        min_ratings = st.sidebar.slider(
            "Minimum d'évaluations requises :",
            min_value=3,
            max_value=50,
            value=30,
            help="Les livres avec moins d'évaluations sont exclus",
            key="tab6_min_ratings"  # ← CORRECTION: Clé unique
        )
        
        reliable_books = book_stats[book_stats['Rating-Count'] >= min_ratings]
        
        if reliable_books.empty:
            st.warning(f"⚠️ Aucun livre avec au moins {min_ratings} évaluations")
            st.info("💡 Réduisez le nombre minimum d'évaluations dans la barre latérale")
            return
        
        # Tri par note moyenne des livres fiables
        top_rated_books = (
            reliable_books.sort_values(by="Book-Rating", ascending=False)
            .head(num_books)
        )
    
    # Vérification qu'on a des résultats
    if top_rated_books.empty:
        st.warning("⚠️ Aucun livre trouvé correspondant aux critères")
        return
    
    #Information pour l'utilisateur
    if rating_method == "Note pondérée (min 5 évaluations)":
        avg_rating_count = top_rated_books['Rating-Count'].mean()
        st.info(f"📊 Top {len(top_rated_books)} livres les mieux notés (moyenne {avg_rating_count:.1f} évaluations par livre)")
    else:
        st.info(f"📊 Top {len(top_rated_books)} livres par note la plus élevée")
    
    # Alert si beaucoup de notes parfaites
    perfect_ratings = (top_rated_books['Book-Rating'] == 10).sum()
    if perfect_ratings > 7:
        st.warning(f"⚠️ {perfect_ratings} livres ont une note parfaite de 10/10. Considérez la 'Note pondérée' pour plus de variété.")
    
    #Layout responsive
    if len(top_rated_books) <= 3:
        num_cols = len(top_rated_books)
    elif len(top_rated_books) <= 5:
        num_cols = min(5, len(top_rated_books))
    else:
        num_cols = 5
    
    # Créer les colonnes
    cols = st.columns(num_cols)
    
    #Affichage avec gestion d'erreurs et métadonnées enrichies
    for idx in range(len(top_rated_books)):
        col_idx = idx % num_cols  # Pour gérer les lignes multiples
        
        # Extraire les données avec gestion d'erreurs
        try:
            book_title = top_rated_books.iloc[idx]["Book-Title"]
            
            #Priorité aux images API puis fallback
            api_image = top_rated_books.iloc[idx].get("api_cover_url")
            original_image = top_rated_books.iloc[idx].get("Image-URL-L")
            
            if pd.notna(api_image) and api_image and api_image.startswith('http'):
                book_image = api_image
            elif pd.notna(original_image) and original_image and original_image.startswith('http'):
                book_image = original_image
            else:
                book_image = "https://via.placeholder.com/300x400/FFD700/333333?text=Livre+Étoile"
            
            #Titre tronqué pour l'affichage
            display_title = book_title if len(book_title) <= 40 else book_title[:37] + "..."
            
            # Métrique à afficher avec étoiles
            rating_value = top_rated_books.iloc[idx]["Book-Rating"]
            rating_display = f"⭐ {rating_value:.1f}/10"
            
            # Ajouter le nombre d'évaluations si disponible
            if 'Rating-Count' in top_rated_books.columns:
                rating_count = top_rated_books.iloc[idx]["Rating-Count"]
                rating_help = f"Note: {rating_value:.1f}/10 sur {rating_count} évaluations"
            else:
                rating_count = None  # ← CORRECTION: Définir rating_count même dans le cas 'else'
                rating_help = f"Note: {rating_value:.1f}/10"
            
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
            
            # AMÉLIORATION 11: Métrique avec étoiles
            cols[col_idx].metric(
                label="Note",  # ← CORRECTION: Label non vide pour l'accessibilité
                value=rating_display,
                help=rating_help
            )
            
            #Badge de qualité selon la note
            if rating_value >= 9.5:
                cols[col_idx].markdown("🏆 **Chef-d'œuvre**", help="Note exceptionnelle ≥ 9.5/10")
            elif rating_value >= 9.0:
                cols[col_idx].markdown("🥇 **Excellence**", help="Très haute note ≥ 9.0/10")
            elif rating_value >= 8.5:
                cols[col_idx].markdown("🥈 **Très bon**", help="Haute note ≥ 8.5/10")
            else:
                cols[col_idx].markdown("📚 **Recommandé**", help="Bonne note")
            
            # Bouton d'action
            if st.button("📖 Détails", key=f"details_rated_{idx}"):
                # CORRECTION: Gestion sécurisée de rating_count
                if rating_count is not None:
                    rating_count_text = f" ({rating_count} évaluations)"
                else:
                    rating_count_text = ""
                st.info(f"📚 **{book_title}**\n\n⭐ Note: {rating_value:.1f}/10{rating_count_text}")
    
    # Statistiques spécialisées pour les notes
    with st.expander("📊 Statistiques des notes"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_displayed_rating = top_rated_books['Book-Rating'].mean()
            st.metric(
                "Note moyenne affichée", 
                f"{avg_displayed_rating:.2f}/10",
                help="Moyenne des livres actuellement affichés"
            )
        
        with col2:
            if 'Rating-Count' in top_rated_books.columns:
                total_evaluations = top_rated_books['Rating-Count'].sum()
                st.metric(
                    "Total évaluations", 
                    f"{total_evaluations:,}",
                    help="Somme des évaluations de tous les livres affichés"
                )
            else:
                overall_avg = books_df['Book-Rating'].mean()
                st.metric("Note moyenne générale", f"{overall_avg:.2f}/10")
        
        with col3:
            perfect_count = (top_rated_books['Book-Rating'] == 10).sum()
            st.metric(
                "Notes parfaites", 
                f"{perfect_count}/{len(top_rated_books)}",
                help="Nombre de livres avec 10/10"
            )
        
        # Graphique de distribution des notes (si plus de 5 livres)
        if len(top_rated_books) > 5:
            st.subheader("📈 Distribution des notes")
            rating_counts = top_rated_books['Book-Rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)


# Version simplifiée améliorée
def show_top_rated_books_simple(books_df):
    """Version simplifiée avec améliorations minimales."""
    st.subheader("⭐ Livres les mieux notés")
    
    # Validation de base
    if books_df.empty or 'Book-Rating' not in books_df.columns:
        st.error("❌ Dataset invalide ou colonne Book-Rating manquante")
        return
    
    # Filtrer les livres avec au moins 3 évaluations pour éviter les biais
    if 'Rating-Count' in books_df.columns:
        reliable_books = books_df[books_df['Rating-Count'] >= 3]
        if not reliable_books.empty:
            books_to_use = reliable_books
            st.info("📊 Affichage des livres avec au moins 3 évaluations")
        else:
            books_to_use = books_df
            st.warning("⚠️ Aucun livre avec 3+ évaluations, affichage de tous les livres")
    else:
        books_to_use = books_df
    
    # Code amélioré
    top_rated_books = (
        books_to_use.sort_values(by="Book-Rating", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )
    
    # Gestion du layout
    num_cols = min(5, len(top_rated_books))
    cols = st.columns(num_cols)
    
    for idx, col in enumerate(cols):
        if idx < len(top_rated_books):
            book_title = top_rated_books.iloc[idx]["Book-Title"]
            book_rating = top_rated_books.iloc[idx]["Book-Rating"]
            
            # Priorité aux images API
            api_image = top_rated_books.iloc[idx].get("api_cover_url")
            original_image = top_rated_books.iloc[idx].get("Image-URL-L")
            
            book_image = api_image if pd.notna(api_image) and api_image.startswith('http') else original_image
            
            with col:
                col.markdown(
                    render_aligned_image(book_image, book_title),
                    unsafe_allow_html=True
                )
                # Afficher la note sous l'image
                col.metric(
                    label="Note",  # ← CORRECTION: Label non vide
                    value=f"⭐ {book_rating:.1f}/10"
                )