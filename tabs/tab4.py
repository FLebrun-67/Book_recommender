# tabs/tab4.py
import streamlit as st

def show_about_tab():
    """Onglet About - Informations sur le projet"""
    
    st.header("📊 À propos du système")
    st.markdown("### Système de recommandation de livres pour librairies")
    
    # Section 1: Description du projet
    st.subheader("🎯 Objectif du projet")
    st.markdown("""
    Ce système a été conçu pour **aider les libraires** à conseiller leurs clients en temps réel. 
    Lorsqu'un client dit *"J'ai aimé ce livre"*, le libraire peut obtenir instantanément 
    des recommandations personnalisées et pertinentes.
    """)
    
    # Section 2: Technologies utilisées
    st.subheader("🛠️ Technologies utilisées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Backend & IA :**
        - 🐍 Python 3.11
        - 🤖 Scikit-learn (KNN)
        - 📊 Pandas pour les données
        - 🔢 NumPy pour les calculs
        - 📈 TF-IDF pour l'analyse textuelle
        """)
    
    with col2:
        st.markdown("""
        **Frontend & API :**
        - 🚀 Streamlit pour l'interface
        - 🌐 Open Library API
        - 📚 Système de cache intelligent
        - 🎨 Interface adaptée aux libraires
        """)
    
    # Section 3: Fonctionnalités
    st.subheader("⚡ Fonctionnalités principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Pour les libraires :**
        - 🎯 Recommandations instantanées
        - 📖 Interface simple et intuitive
        - 📊 Scores de similarité
        - 🏷️ Informations détaillées sur chaque livre
        """)
    
    with col2:
        st.markdown("""
        **Pour les clients :**
        - 🔍 Découverte personnalisée
        - 📚 Recommandations basées sur leurs goûts
        - ⭐ Livres similaires
        - 🎨 Suggestions diversifiées
        """)
    
    # Section 4: Comment ça marche
    st.subheader("🧠 Comment fonctionne le système")
    
    with st.expander("🔍 Algorithme de recommandation (détails techniques)"):
        st.markdown("""
        **1. Analyse des caractéristiques des livres :**
        - 📝 **Descriptions** : Analyse du contenu et des thèmes
        - 🏷️ **Genres/Tags** : Classification par catégories
        - ✍️ **Auteurs** : Style et univers littéraire
        - 📅 **Époque de publication** : Contexte temporel
        
        **2. Transformation en données numériques :**
        - 🔤 **TF-IDF** : Conversion du texte en vecteurs numériques
        - 📊 **Normalisation** : Mise à l'échelle des données
        - 🎯 **Pondération** : Importance relative de chaque critère
        
        **3. Calcul de similarité :**
        - 📏 **Distance cosinus** : Mesure de proximité entre livres
        - 🤖 **KNN (K-Nearest Neighbors)** : Recherche des plus proches voisins
        - ⭐ **Score de pertinence** : Pourcentage de similarité
        
        **4. Recommandations finales :**
        - 🎯 Sélection des livres les plus similaires
        - 🚫 Élimination des doublons
        - 📈 Classement par score de pertinence
        """)
    
    # Section 5: Statistiques du système
    st.subheader("📈 Statistiques du système")
    
    # Ici on peut afficher les stats en temps réel
    try:
        from knn_dynamic import get_dynamic_knn
        knn = get_dynamic_knn()
        stats = knn.get_dataset_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Livres", f"{stats['total_books']:,}")
        with col2:
            st.metric("👥 Auteurs", f"{stats['unique_authors']:,}")
        with col3:
            st.metric("🤖 Statut", "✅ Opérationnel" if stats['is_model_fitted'] else "❌ Non entraîné")
        with col4:
            if stats['total_books'] > 0:
                st.metric("📅 Période", stats.get('year_range', 'N/A'))
            else:
                st.metric("📅 Période", "N/A")
                
    except Exception:
        st.info("📊 Statistiques indisponibles - Construisez d'abord le dataset")
    
    # Section 6: Avantages pour la librairie
    st.subheader("💡 Avantages pour votre librairie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Business :**
        - Amélioration de l'expérience client
        - Différenciation concurrentielle
        - Fidélisation de la clientèle
        """)
    
    with col2:
        st.markdown("""
        **⚡ Opérationnel :**
        - Gain de temps pour les libraires
        - Expertise renforcée
        - Découverte de nouveaux titres
        - Service client premium
        """)
    
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 14px; margin-top: 20px;">
        📚 Système de recommandation de livres - Conçu pour les libraires modernes
    </div>
    """, unsafe_allow_html=True)