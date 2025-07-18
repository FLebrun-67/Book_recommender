# tabs/tab3.py
import streamlit as st
import pandas as pd
import random
import sys
sys.path.append('..')
from book_recommender.api_utils import get_api_client

@st.fragment
def show_search_tab():
    """Display search functionality tab using API Open Library only."""
    
    st.subheader("🔍 Recherche de livres")
    st.markdown("### Recherche via API Open Library")
    
    try:
        # Obtenir l'instance API
        api_client = get_api_client()
        
        # Afficher les statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌐 Source", "API Open Library")
        with col2:
            st.metric("📚 Catalogue", "Mondial")
        with col3:
            st.metric("🔍 Type", "Recherche + Descriptions")
        
        st.divider()
        
        # Section 1: Recherche classique
        st.subheader("🔍 Recherche par mot-clé")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Rechercher un livre par titre, auteur ou genre:",
                placeholder="Ex: Harry Potter, Tolkien, fantasy..."
            )
        with col2:
            search_limit = st.selectbox(
                "Nombre de résultats:",
                [5, 10, 15, 20],
                index=1
            )
        
        if search_query:
            with st.spinner(f"🔍 Recherche de '{search_query}' via Open Library..."):
                try:
                    # Recherche via API avec récupération des descriptions
                    api_books = api_client.search_books(search_query, search_limit)
                    
                    if api_books:
                        st.success(f"✅ **{len(api_books)} livre(s) trouvé(s)**")
                        
                        # Enrichir avec les descriptions
                        with st.spinner("📖 Récupération des descriptions..."):
                            enriched_books = []
                            
                            for i, book in enumerate(api_books):
                                # Récupérer la description détaillée si la clé existe
                                description = ""
                                if book.get('key'):
                                    try:
                                        description = _get_book_description_from_api(api_client, book['key'])
                                    except:
                                        description = ""
                                
                                # Si pas de description, utiliser les sujets comme fallback
                                if not description.strip():
                                    description = book.get('subject_string', '')
                                
                                enriched_books.append({
                                    'title': book.get('title', 'N/A'),
                                    'author': book.get('author_string', 'N/A'),
                                    'year': book.get('first_publish_year', 'N/A'),
                                    'genres': book.get('subject_string', 'N/A')[:100],
                                    'cover_url': book.get('cover_url', ''),
                                    'description': description,
                                    'isbn': book.get('isbn', 'N/A'),
                                    'key': book.get('key', '')
                                })
                                
                                # Afficher le progrès
                                if (i + 1) % 3 == 0:
                                    st.write(f"    📖 Descriptions récupérées: {i + 1}/{len(api_books)}")
                        
                        # Tableau des résultats
                        display_data = []
                        for book in enriched_books:
                            display_data.append({
                                'Titre': book['title'][:50] + "..." if len(book['title']) > 50 else book['title'],
                                'Auteur': book['author'][:30] + "..." if len(book['author']) > 30 else book['author'],
                                'Année': book['year'],
                                'Genres': book['genres'][:40] + "..." if len(book['genres']) > 40 else book['genres'],
                                'Description': "✅" if book['description'] and len(book['description'].strip()) > 10 else "❌"
                            })
                        
                        results_df = pd.DataFrame(display_data)
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True,
                            height=300
                        )
                        
                        # Sélection pour détails
                        st.subheader("📖 Détails du livre")
                        
                        selected_book_index = st.selectbox(
                            "Sélectionner un livre pour voir les détails complets:",
                            range(len(enriched_books)),
                            format_func=lambda x: f"{enriched_books[x]['title']} - {enriched_books[x]['author']}"
                        )
                        
                        if selected_book_index is not None:
                            selected_book = enriched_books[selected_book_index]
                            
                            # Affichage détaillé
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if selected_book['cover_url'] and selected_book['cover_url'].startswith('http'):
                                    st.image(selected_book['cover_url'], width=150, caption="Couverture")
                                else:
                                    st.info("📷 Pas de couverture disponible")
                            
                            with col2:
                                st.markdown(f"**📖 Titre :** {selected_book['title']}")
                                st.markdown(f"**✍️ Auteur :** {selected_book['author']}")
                                st.markdown(f"**📅 Année :** {selected_book['year']}")
                                st.markdown(f"**🏷️ Genres :** {selected_book['genres']}")
                                st.markdown(f"**📚 ISBN :** {selected_book['isbn']}")
                                st.markdown("**🌐 Source :** Open Library API")
                            
                            # Description complète
                            st.markdown("---")
                            if selected_book['description'] and len(selected_book['description'].strip()) > 10:
                                st.markdown("**📝 Description complète :**")
                                desc_text = selected_book['description'].strip()
                                
                                # Affichage avec gestion de longueur
                                if len(desc_text) > 500:
                                    # Description longue - avec bouton pour voir plus
                                    short_desc = desc_text[:500] + "..."
                                    st.markdown(f"*{short_desc}*")
                                    
                                    if st.button("📚 Voir la description complète", key=f"full_desc_{selected_book_index}"):
                                        st.markdown("**📖 Description intégrale :**")
                                        st.markdown(f"*{desc_text}*")
                                else:
                                    # Description courte - affichage direct
                                    st.markdown(f"*{desc_text}*")
                            else:
                                st.info("📝 Aucune description détaillée disponible")
                                # Fallback avec les genres
                                if selected_book['genres'] != 'N/A':
                                    fallback_desc = f"Livre de genre {selected_book['genres']} par {selected_book['author']}"
                                    if selected_book['year'] != 'N/A':
                                        fallback_desc += f", publié en {selected_book['year']}"
                                    fallback_desc += "."
                                    st.markdown(f"**📋 Informations générales :** *{fallback_desc}*")
                            
                            # Simulation de disponibilité
                            st.markdown("---")
                            st.markdown("**📦 Informations de disponibilité :**")
                            
                            col_stock, col_price = st.columns(2)
                            with col_stock:
                                stock_status = random.choice([
                                    ("✅ En stock", "success"),
                                    ("⚠️ Stock faible", "warning"), 
                                    ("❌ Épuisé", "error"),
                                    ("📦 Sur commande", "info")
                                ])
                                st.markdown(f"**Stock :** {stock_status[0]}")
                            
                            with col_price:
                                price = random.randint(12, 28)
                                st.markdown(f"**Prix estimé :** {price}€")
                    
                    else:
                        st.warning("❌ Aucun livre trouvé pour cette recherche")
                        st.info("💡 Essayez avec des termes différents ou plus généraux")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {str(e)}")
        
        st.divider()
        
        # Section 2: Découverte aléatoire (API uniquement)
        st.subheader("🎲 Découverte aléatoire")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎲 Livre aléatoire via API", key="random_api"):
                try:
                    # Termes aléatoires pour diversifier
                    random_terms = [
                        "fiction", "fantasy", "mystery", "romance", "science", "history", 
                        "adventure", "classic", "novel", "literature", "drama", "poetry",
                        "biography", "philosophy", "psychology", "art", "music", "travel"
                    ]
                    random_term = random.choice(random_terms)
                    
                    with st.spinner(f"🌐 Recherche aléatoire pour '{random_term}'..."):
                        api_books = api_client.search_books(random_term, 30)
                        
                        if api_books:
                            # Choisir un livre aléatoire
                            random_api_book = random.choice(api_books)
                            
                            # Récupérer sa description
                            description = ""
                            if random_api_book.get('key'):
                                try:
                                    description = _get_book_description_from_api(api_client, random_api_book['key'])
                                except:
                                    description = ""
                            
                            if not description.strip():
                                description = random_api_book.get('subject_string', '')
                            
                            st.success("🌐 **Découverte aléatoire :**")
                            
                            # Affichage du livre aléatoire
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                if random_api_book.get('cover_url') and random_api_book['cover_url'].startswith('http'):
                                    st.image(random_api_book['cover_url'], width=120)
                                else:
                                    st.info("📷 Pas de couverture")
                            
                            with col_info:
                                st.markdown(f"**📖** {random_api_book.get('title', 'N/A')}")
                                st.markdown(f"**✍️** {random_api_book.get('author_string', 'N/A')}")
                                st.markdown(f"**📅** {random_api_book.get('first_publish_year', 'N/A')}")
                                genres = random_api_book.get('subject_string', 'N/A')
                                if len(genres) > 80:
                                    genres = genres[:80] + "..."
                                st.markdown(f"**🏷️** {genres}")
                            
                            # Description
                            if description and len(description.strip()) > 10:
                                st.markdown("**📝 Description :**")
                                desc_text = description.strip()
                                if len(desc_text) > 300:
                                    desc_text = desc_text[:300] + "..."
                                st.markdown(f"*{desc_text}*")
                            else:
                                st.info("📝 Description non disponible")
                            
                            # Disponibilité
                            st.markdown("**📦 Disponibilité :**")
                            stock = random.choice(["✅ En stock", "⚠️ Stock faible", "❌ Épuisé", "📦 Sur commande"])
                            price = random.randint(12, 25)
                            st.markdown(f"{stock} - Prix: {price}€")
                        
                        else:
                            st.warning("❌ Aucun livre trouvé via l'API")
                
                except Exception as e:
                    st.error(f"❌ Erreur API: {str(e)}")
        
        with col2:
            st.info("""
            **🎲 Comment ça marche ?**
            
            La découverte aléatoire :
            1. Choisit un terme aléatoire
            2. Recherche via l'API
            3. Sélectionne un livre au hasard
            4. Récupère sa description
            5. Affiche le tout avec disponibilité
            """)
        
        # Section informative
        with st.expander("ℹ️ À propos de cette recherche"):
            st.markdown("""
            ### 🌐 **Recherche API Open Library**
            
            **Avantages :**
            - 🌍 Accès au catalogue mondial Open Library
            - 📖 Descriptions détaillées récupérées automatiquement
            - 🔍 Recherche dans titres, auteurs et genres
            - 🎲 Découverte aléatoire diversifiée
            
            **Fonctionnalités :**
            - **Recherche intelligente** : Titres, auteurs, genres
            - **Descriptions enrichies** : Récupération automatique depuis l'API Works
            - **Métadonnées complètes** : Année, genres, ISBN, couverture
            - **Disponibilité simulée** : Stock et prix pour démonstration libraire
            
            **Performances :**
            - Recherche rapide dans la base Open Library
            - Enrichissement progressif avec descriptions
            - Gestion des erreurs et fallbacks automatiques
            """)
    
    except Exception as e:
        st.error(f"❌ Erreur dans l'onglet recherche: {str(e)}")
        st.info("Vérifiez que le module api_utils est disponible et fonctionnel")


def _get_book_description_from_api(api_client, book_key):
    """Récupère la description détaillée d'un livre depuis l'API Works"""
    try:
        # Si c'est une clé d'édition, récupérer d'abord le work
        if '/books/' in book_key:
            edition_url = f"https://openlibrary.org{book_key}.json"
            edition_response = api_client.session.get(edition_url, timeout=5)
            
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
        work_response = api_client.session.get(work_url, timeout=5)
        
        if work_response.status_code == 200:
            work_data = work_response.json()
            
            description = ""
            if 'description' in work_data:
                desc = work_data['description']
                if isinstance(desc, dict) and 'value' in desc:
                    description = desc['value']
                elif isinstance(desc, str):
                    description = desc
            
            return description.strip()
        
        return ""
        
    except Exception:
        return ""