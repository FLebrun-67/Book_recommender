# tabs/tab_bookstore_demo.py
import streamlit as st
import pandas as pd
from knn_dynamic import get_dynamic_knn
#from utils import render_aligned_image
import random

def load_books_from_csv(csv_path: str) -> list:
    """Charge les livres depuis le CSV avec toutes les corrections"""
    df = pd.read_csv(csv_path)
    catalogue_df = df.drop_duplicates(subset="ISBN")
    books = []

    for _, row in catalogue_df.iterrows():
        
        safe_year: int = 2000 
        year_value = row.get('api_first_publish_year')
        if not (pd.isna(year_value) or year_value == '' or str(year_value) == 'nan'):
            try:
                safe_year = int(float(year_value))
            except (ValueError, TypeError):
                safe_year = 2000

        # ✅ CORRECTION 2: Description propre
        description = row.get('Description', '')
        if pd.isna(description) or str(description) == 'nan':
            description = ''

        books.append({
            'title': row.get('Book-Title', ''),
            'author_string': row.get('Book-Author', ''),
            'subject_string': row.get('subject_string_final', ''), 
            'description': str(description).strip(), 
            'publisher_string': row.get('api_publisher_string', ''), 
            'first_publish_year': safe_year,  
            'isbn': row.get('ISBN', ''),
            'cover_url': row.get('api_cover_url', ''),
            'key': row.get('key', '')
        })
    return books


def show_bookstore_demo():
    """Onglet pour les recommandations dynamiques avec KNN"""
    
    st.header("Recommandations par livre")
    st.markdown("### Système de recommandation basé sur l'API Open Library")
    
    try:
        books_data = load_books_from_csv("data/catalog_clean.csv")

        knn = get_dynamic_knn()
        knn.books_data = books_data

        if len(books_data) >= 5:
            knn._fit_model()

        stats={
            'total_books': len(books_data),
            'unique_authors': len(set(book.get('author_string', '') for book in books_data)),
            'is_model_fitted': len(books_data) > 0
        }
        
        # Afficher les statistiques en colonnes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Livres", stats['total_books'])
        with col2:
            st.metric("👥 Auteurs", stats['unique_authors'])
        with col3:
            st.metric("🤖 Modèle", "✅" if stats['is_model_fitted'] else "❌")
        
        if stats['total_books'] > 0:
            st.write(f"**Période:** {stats.get('year_range', 'N/A')}")
        
        st.divider()
        
        # Interface simplifiée pour libraire
        st.subheader("🔍 Recherche de recommandations")

        
        # Interface de recommandation principale
        if stats['total_books'] > 0:
            book_titles = [book['title'] for book in knn.books_data]
            
            # Recherche par titre
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_book = st.selectbox(
                    "📖 Choisir un livre :",
                    options=book_titles,
                    help="Tapez pour rechercher ou sélectionnez dans la liste"
                )
            
            with col2:
                n_recommendations = st.selectbox(
                    "Nombre de suggestions :",
                    options=[3, 5, 8, 10],
                    index=1
                )
            
            if selected_book:
                st.subheader("📖 Livre de référence pour la recommendation")
                
                # Trouver le livre dans les données KNN
                reference_book = None
                for book in knn.books_data:
                    if book['title'] == selected_book:
                        reference_book = book
                        break
                
                if reference_book:
                    # Affichage optimisé pour le libraire
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        # Couverture
                        cover_url = reference_book.get("cover_url")
                        if cover_url and str(cover_url) != 'nan' and cover_url.startswith('http'):
                            st.image(cover_url, width=120, caption="Couverture")
                        else:
                            st.info("📷 Pas de couverture")
                    
                    with col2:
                        # Informations principales pour le libraire
                        st.markdown(f"**📖 Titre :** {reference_book.get('title', 'N/A')}")
                        st.markdown(f"**✍️ Auteur :** {reference_book.get('author_string', 'N/A')}")
                        st.markdown(f"**📅 Année :** {reference_book.get('first_publish_year', 'N/A')}")
                        
                        # Genres - important pour les recommandations
                        subjects = reference_book.get('subject_string', 'N/A')
                        if len(subjects) > 80:
                            subjects = subjects[:80] + "..."
                        st.markdown(f"**🏷️ Genres :** {subjects}")
                        
                        # Éditeur si disponible (utile pour le libraire)
                        publisher = reference_book.get('publisher_string')
                        if publisher and str(publisher) != 'nan' and publisher.strip():
                            if len(publisher) > 40:
                                publisher = publisher[:40] + "..."
                            st.markdown(f"**🏢 Éditeur :** {publisher}")
                    
                    with col3:
                        # Informations business pour le libraire
                        st.markdown("**💼 Info Libraire :**")
                        
                        # Simulation stock du livre de référence
                        ref_stock = random.choice([
                            ("✅ En stock", "success"),
                            ("⚠️ Stock faible", "warning"), 
                            ("❌ Épuisé", "error")
                        ])
                        st.markdown(f"**Stock :** {ref_stock[0]}")
                        
                        # Prix simulé
                        ref_price = random.randint(15, 28)
                        st.markdown(f"**Prix :** {ref_price}€")
                        
                        # Genre principal pour cibler les recommandations
                        if reference_book.get('subject_string'):
                            main_genre = reference_book['subject_string'].split(',')[0].strip()
                            st.markdown(f"**🎯 Genre principal :** {main_genre}")
                    
                    # Description courte si disponible (utile pour argumenter)
                    description = reference_book.get('description', '')
                    if description and str(description) != 'nan' and len(str(description).strip()) > 10:
                        st.markdown("**📝 Synopsis :**")
                        desc_text = str(description)
                        if len(desc_text) > 500:
                            desc_text = desc_text[:500] + "..."
                        st.markdown(f"*{desc_text}*")
                
                # Séparateur visuel
                st.markdown("---")
            
            if st.button("🎯 Obtenir les recommandations", key="get_bookstore_recs", type="primary"):
                if selected_book:
                    with st.spinner("🤔 Recherche des meilleures recommandations..."):
                        recommendations = knn.get_recommendations(selected_book, n_recommendations)
                        
                        if recommendations:
                            # Interface adaptée libraire
                            st.success(f"✨ Voici {len(recommendations)} excellentes suggestions pour votre client !")
                            
                            # Affichage des recommandations en format "vitrine"
                            for i, rec in enumerate(recommendations, 1):
                                with st.expander(f"📚 Suggestion #{i}: {rec['title']} ⭐ {rec['similarity_score']:.0%} de similarité"):
                                    
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    
                                    with col1:
                                        if rec.get('cover_url'):
                                            st.image(rec['cover_url'], width=120)
                                    
                                    with col2:
                                        st.markdown(f"**📖 Titre :** {rec['title']}")
                                        st.markdown(f"**✍️ Auteur :** {rec['author_string']}")
                                        st.markdown(f"**📅 Année :** {rec['first_publish_year']}")
                                        
                                        # Points de vente pour le libraire
                                        subjects = rec['subject_string'][:100]
                                        if subjects:
                                            st.markdown(f"**🏷️ Genres :** {subjects}...")
                                        
                                        # Score de similarité pour le libraire
                                        similarity_percent = rec['similarity_score'] * 100
                                        st.progress(rec['similarity_score'], 
                                                  text=f"Similarité: {similarity_percent:.0f}%")
                                    
                                    st.markdown("---")  # Séparateur
                                    
                                    # Résumé/Description complète
                                    description = rec.get('description', '')
                                    if description and str(description) != 'nan' and len(str(description).strip()) > 10:
                                        st.markdown("**📖 Résumé du livre :**")
                                        desc_text = str(description).strip()
                                        
                                        # Gérer la longueur : affichage progressif
                                        if len(desc_text) > 300:
                                            # Affichage avec option "Voir plus"
                                            short_desc = desc_text[:300] + "..."
                                            
                                            # Utiliser l'expander pour le texte long
                                            st.markdown(f"*{short_desc}*")
                                            
                                            if st.button("📚 Voir le résumé complet", key=f"full_desc_{i}"):
                                                st.markdown("**📖 Résumé complet :**")
                                                st.markdown(f"*{desc_text}*")
                                        else:
                                            # Description courte : affichage direct
                                            st.markdown(f"*{desc_text}*")
                                    
                                    elif rec.get('subject_string'):
                                        # Si pas de vraie description, utiliser les sujets comme description de fallback
                                        st.markdown("**📖 Résumé du livre :**")
                                        subjects_desc = f"Livre de genre {rec['subject_string']} par {rec['author_string']}"
                                        if rec.get('first_publish_year'):
                                            subjects_desc += f", publié en {rec['first_publish_year']}"
                                        subjects_desc += "."
                                        st.markdown(f"*{subjects_desc}*")
                                    
                                    else:
                                        # Aucune description disponible
                                        st.markdown("**📖 Résumé du livre :**")
                                        st.info("📝 Résumé non disponible dans la base de données")
                                        
                                        # Suggestion pour le libraire
                                        st.markdown("💡 **Conseil libraire :** Consultez la quatrième de couverture ou votre fournisseur pour plus d'informations")

                                    
                                    with col3:
                                        # Outils pour le libraire
                                        st.markdown("**💼 Infos pratiques :**")
                                        
                                        # Simulation de disponibilité
                                        stock_status = random.choice([
                                            ("✅ En stock", "success"),
                                            ("⚠️ Stock faible", "warning"), 
                                            ("❌ Épuisé", "error"),
                                            ("📦 Sur commande", "info")
                                        ])
                                        st.markdown(f"**Stock :** {stock_status[0]}")
                                        
                                        # Prix simulé
                                        price = random.randint(12, 25)
                                        st.markdown(f"**Prix :** {price}€")
                                        
                                        if st.button("🔍 Détails", key=f"details_{i}"):
                                            st.info("Fonctionnalité : Voir fiche produit complète")
                            
                            # Statistiques de session pour le libraire
                            st.divider()
                            with st.expander("📊 Statistiques de la session (pour le libraire)"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🎯 Recommandations générées", len(recommendations))
                                with col2:
                                    avg_similarity = sum(r['similarity_score'] for r in recommendations) / len(recommendations)
                                    st.metric("⭐ Pertinence moyenne", f"{avg_similarity:.0%}")
                                with col3:
                                    available_books = sum(1 for _ in recommendations if random.choice([True, True, False]))
                                    st.metric("📦 Livres disponibles", f"{available_books}/{len(recommendations)}")
                        
                        else:
                            st.warning("😔 Aucune recommandation trouvée pour ce livre")
                            st.info("💡 Conseil : Essayez d'alimenter le dataset avec plus de livres du même genre")
                
                else:
                    st.warning("⚠️ Veuillez sélectionner un livre")
        
        else:
            # Interface si pas de dataset
            st.info("📚 **Dataset vide** - Alimentez d'abord le système via l'onglet 'Dataset Builder'")
            
            st.markdown("""
            ### 🏗️ Configuration initiale pour librairie
            
            **Pour utiliser ce système dans votre librairie :**
            
            1. **Construisez votre catalogue** via l'onglet "Dataset Builder"
            2. **Choisissez vos genres principaux** (fantasy, policier, romance, etc.)
            3. **Lancez la construction** (une seule fois, 30-60 minutes)
            4. **Utilisez le système** quotidiennement pour conseiller vos clients
            
            """)
        
        # Section informative pour la présentation
        with st.expander("ℹ️ À propos de ce système (pour la présentation)"):
            st.markdown("""
            ### 🎯 Objectif du projet
            
            **Simuler un système de recommandation réaliste pour librairie** en utilisant :
            - L'API Open Library (gratuite) comme source de données
            - Un algorithme KNN pour calculer les similarités
            - Une interface adaptée aux besoins des libraires
            
            ### 🔧 Adaptations pour le contexte réel
            
            **Dans une vraie librairie, ce système pourrait être connecté à :**
            - 📦 Système de gestion des stocks
            - 💰 Base de données des prix
            - 📊 Historique des ventes
            - 👥 Profils clients et historique d'achats
            
            ### 💡 Valeur ajoutée
            
            - **Pour le client :** Découverte personnalisée, gain de temps
            - **Pour le libraire :** Outil d'aide à la vente, expertise renforcée
            - **Pour la librairie :** Augmentation des ventes, fidélisation client
            """)
    
    except Exception as e:
        st.error(f"❌ Erreur dans l'interface librairie: {str(e)}")
        st.info("Vérifiez que le dataset est construit et que tous les fichiers sont présents.")