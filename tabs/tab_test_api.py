# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false
# type: ignore


import streamlit as st
import pandas as pd
from api_utils import search_books_streamlit, get_book_details_streamlit

def show_test_api_tab():
    """Onglet de test pour l'API Open Library"""
    
    st.header("🧪 Test de l'API Open Library")
    st.write("Cet onglet permet de tester les fonctions de récupération de données")
    
    # Section 1: Test de recherche
    st.subheader("🔍 Test de recherche de livres")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Tapez un titre de livre ou nom d'auteur:", 
            value="harry potter",
            key="api_search"
        )
    with col2:
        limit = st.number_input("Nombre de résultats:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Rechercher", key="search_btn"):
        if search_query:
            with st.spinner("Recherche en cours..."):
                try:
                    # Utiliser notre fonction API
                    results_df = search_books_streamlit(search_query, limit)
                    
                    if not results_df.empty:
                        st.success(f"✅ {len(results_df)} livres trouvés !")
                        
                        # Afficher les résultats dans un format lisible
                        for idx, book in results_df.iterrows():
                            with st.expander(f"📖 {book.get('title', 'Titre non disponible')}"):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Afficher la couverture si disponible
                                    if 'cover_url' in book and pd.notna(book['cover_url']):
                                        st.image(book['cover_url'], width=150)
                                
                                with col2:
                                    st.write(f"**Auteur(s):** {book.get('author_string', 'Non disponible')}")
                                    st.write(f"**Première publication:** {book.get('first_publish_year', 'Non disponible')}")
                                    st.write(f"**ISBN:** {book.get('isbn', 'Non disponible')}")
                                    
                                    # Afficher les sujets (limité à 5 pour la lisibilité)
                                    subjects = book.get('subjects', [])
                                    if subjects:
                                        subjects_display = subjects[:5] if len(subjects) > 5 else subjects
                                        st.write(f"**Sujets:** {', '.join(subjects_display)}")
                                        if len(subjects) > 5:
                                            st.write(f"... et {len(subjects) - 5} autres")
                    else:
                        st.warning("❌ Aucun livre trouvé pour cette recherche")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {str(e)}")
        else:
            st.warning("⚠️ Veuillez entrer un terme de recherche")
    
    st.divider()


    # Section 2: Test de récupération par ISBN
    st.subheader("📚 Test de récupération par ISBN")
    
    isbn_input = st.text_input(
        "Entrez un ISBN (10 ou 13 chiffres):", 
        value="9780747532699",  # Harry Potter ISBN
        key="isbn_input"
    )
    
    if st.button("📖 Récupérer les détails", key="isbn_btn"):
        if isbn_input:
            with st.spinner("Récupération des détails..."):
                try:
                    book_details = get_book_details_streamlit(isbn_input)
                    
                    if book_details:
                        st.success("✅ Livre trouvé !")
                        
                        # Affichage détaillé
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Couverture
                            if 'cover_urls' in book_details and book_details['cover_urls']:
                                cover_url = book_details['cover_urls'].get('large') or book_details['cover_urls'].get('medium')
                                if cover_url:
                                    st.image(cover_url, width=200)
                        
                        with col2:
                            st.markdown(f"### 📖 {book_details.get('title', 'Titre non disponible')}")
                            st.write(f"**Auteur(s):** {book_details.get('author_string', 'Non disponible')}")
                            st.write(f"**Date de publication:** {book_details.get('publish_date', 'Non disponible')}")
                            st.write(f"**Nombre de pages:** {book_details.get('number_of_pages', 'Non disponible')}")
                            
                            # Description
                            if book_details.get('description'):
                                st.write("**Description:**")
                                st.write(book_details['description'][:300] + "..." if len(book_details['description']) > 300 else book_details['description'])
                        
                        # Sujets et classifications
                        if book_details.get('subject_string'):
                            st.write("**Sujets/Genres:**")
                            st.write(book_details['subject_string'])
                        
                        if book_details.get('dewey_string'):
                            st.write("**Classification Dewey:**")
                            st.write(book_details['dewey_string'])
                        
                        # Données brutes pour debug (dans un expander)
                        with st.expander("🔧 Données brutes (pour développement)"):
                            st.json(book_details)
                    
                    else:
                        st.warning("❌ Aucun livre trouvé pour cet ISBN")
                        st.info("💡 Essayez avec un autre ISBN ou utilisez la recherche par titre")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la récupération: {str(e)}")
        else:
            st.warning("⚠️ Veuillez entrer un ISBN")
    
    st.divider()
    
    # Section 4: Informations techniques
    st.subheader("ℹ️ Informations techniques")
    
    
    with st.expander("🔧 Statistiques API"):
        st.info("""
        **Open Library API :**
        - ✅ Gratuit et sans limite stricte
        - ✅ ~30 millions de livres
        - ✅ Métadonnées complètes
        - ⚠️ Délai de 0.1s entre requêtes (respect de l'API)
        - ⚠️ Beaucoup de doublons
        """)
        

    