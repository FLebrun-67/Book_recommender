# tabs/tab_svd.py
import streamlit as st
from utils import recommend_book_svd_hybrid, recommend_book_svd, render_aligned_image

@st.fragment
def show_user_recommendations(books_df, svd_model):
    """Affichage des recommandations utilisateur avec SVD hybride."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
        
    st.subheader("📚 Trouver et obtenir des recommandations")
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "Guest user"
    selected_user = st.session_state.user_id
    
    if selected_user not in st.session_state:
        st.session_state[selected_user] = []

    # Vérifier si on a des données enrichies
    has_enriched_data = any(col in books_df.columns for col in 
                          ['subject_string_final', 'api_cover_url', 'api_first_publish_year'])

    # Options d'affichage
    col1, col2 = st.columns(2)
    with col1:
        show_descriptions = st.checkbox("Résumé des livres", value=False)
    
    with col2:
        if has_enriched_data:
            use_hybrid = st.checkbox("Recommandations hybrides 🌟", value=True, 
                                    help="Utilise les métadonnées enrichies pour de meilleures recommandations")
        else:
            use_hybrid = False
            st.info("💡 Enrichissez vos données pour des recommandations hybrides")
    
    # Affichage du statut des données
    if has_enriched_data:
        enriched_cols = ['subject_string_final', 'api_cover_url', 'api_first_publish_year', 'api_publisher_string']
        available_cols = [col for col in enriched_cols if col in books_df.columns]
        st.success(f"🌟 Dataset enrichi détecté ! ({len(available_cols)}/4 features)")
    else:
        st.warning("⚠️ Dataset de base - Considérez l'enrichissement pour de meilleures recommandations")
    
    # Bouton de recommandation principal
    if st.button("Voici vos recommandations", key="recommendations_button"):
        with st.spinner("🔮 Generating personalized recommendations..."):
            
            # Choisir le type de recommandation
            if use_hybrid and has_enriched_data:
                st.info("🌟 Utilisez les recommandations hybrides avec les métadonnées enrichies")
                recommendations = recommend_book_svd_hybrid(selected_user, books_df, svd_model)
                method_used = "Hybrid SVD"
            else:
                if not has_enriched_data:
                    st.info("📊 Recommandations SVD classique")
                else:
                    st.info("📊 Recommandations SVD classique (hybrid disabled)")
                recommendations = recommend_book_svd(selected_user, books_df, svd_model)
                method_used = "Classic SVD"

        if recommendations:
            # Sauvegarder dans l'historique
            st.session_state[selected_user].append({
                "type": "user",
                "user": selected_user,
                "method": method_used,
                "recommendations": [
                    (book_name, rating, poster_url, description) 
                    for book_name, rating, poster_url, description in recommendations
                ],
            })
            
            # Affichage des recommandations
            st.subheader("🔍 Vos recommandations:")
            
            # Afficher les préférences détectées si mode hybride
            if use_hybrid and has_enriched_data:
                try:
                    from utils import extract_user_preferences
                    user_prefs = extract_user_preferences(selected_user, books_df)
                    if user_prefs['preferred_genres']:
                        with st.expander("🎯 Vos préférences détectées"):
                            st.write(f"**Genres préférés:** {', '.join(user_prefs['preferred_genres'][:5])}")
                            st.write(f"**Période préférée:** {user_prefs['preferred_year_range'][0]} - {user_prefs['preferred_year_range'][1]}")
                            if user_prefs['known_publishers']:
                                st.write(f"**Éditeurs connus:** {len(user_prefs['known_publishers'])} éditeurs")
                except Exception as e:
                    st.warning(f"Impossible d'extraire les préférences: {str(e)}")
            
            # Organiser en deux lignes de 5
            rows = [recommendations[:5], recommendations[5:10]]

            for row_idx, row in enumerate(rows):
                if row:  # Vérifier que la ligne n'est pas vide
                    cols = st.columns(len(row))
                    for idx, col in enumerate(cols):
                        book_name, rating, poster, book_description = row[idx]
                        
                        with col:
                            # Afficher l'image avec le titre
                            col.markdown(render_aligned_image(poster, book_name), unsafe_allow_html=True)
                            
                            # Afficher le score de recommandation
                            if use_hybrid and has_enriched_data:
                                col.caption(f"🎯 Score hybride: {rating:.2f}")
                            else:
                                col.caption(f"⭐ Notes prédites: {rating:.1f}/10")
                            
                            # Afficher la description si demandée
                            if show_descriptions and book_description != "Description non disponible":
                                with col.expander("📖 Description"):
                                    description_text = book_description[:200] + "..." if len(book_description) > 200 else book_description
                                    st.write(description_text)
        else:
            st.warning("Pas de recommandation trouvé pour cette utilisateur.")

    # Section historique
    if st.checkbox("Votre historique", key="history_user"):
        st.subheader("📜 Historique de vos recommandations")
        if selected_user in st.session_state and st.session_state[selected_user]:
            for i, entry in enumerate(reversed(st.session_state[selected_user]), 1):
                if entry.get("type") == "user":
                    method = entry.get("method", "Unknown")
                    book_list = [book_name for book_name, *_ in entry['recommendations']]
                    
                    with st.expander(f"📚 Recommendation {i} - {method} ({len(book_list)} books)"):
                        st.write(f"**Method:** {method}")
                        st.write(f"**Books:** {', '.join(book_list[:5])}")
                        if len(book_list) > 5:
                            st.write(f"... and {len(book_list) - 5} more books")
        else:
            st.write("Pas d'historique trouvé.")
    
    # Informations techniques sur le dataset (optionnel)
    if st.checkbox("📊 Information technique", key="tech_info"):
        st.subheader("📊 Information sur le Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Livres totals", books_df['Book-Title'].nunique())
            st.metric("Utilisateurs totals", books_df['User-ID'].nunique())
            st.metric("Notes totales", len(books_df))
        
        with col2:
            # Informations sur l'enrichissement
            if has_enriched_data:
                enriched_cols = ['subject_string_final', 'api_cover_url', 'api_first_publish_year', 'api_publisher_string']
                available_enriched = [col for col in enriched_cols if col in books_df.columns]
                st.metric("Enriched Features", f"{len(available_enriched)}/4")
                
                if 'api_enriched' in books_df.columns:
                    api_enriched_count = books_df['api_enriched'].sum()
                    st.metric("API Enriched Books", api_enriched_count)
                
                if 'used_fallback' in books_df.columns:
                    fallback_count = books_df['used_fallback'].sum()
                    st.metric("Fallback to Original", fallback_count)
            else:
                st.metric("Enriched Features", "0/4")
                st.info("💡 Run dataset enrichment to unlock hybrid recommendations")