import streamlit as st
from utils import load_model_and_data
from tabs.tab_svd import show_user_recommendations
from tabs.tab3 import show_search_tab
from tabs.tab4 import show_about_tab
from tabs.tab5 import show_popular_books
from tabs.tab6 import show_top_rated_books
from tabs.tab_test_api import show_test_api_tab
from tabs.tab_bookstore_demo import show_bookstore_demo

# Fonction pour extraire les genres favoris
def get_favorite_genres(user_data):
    if 'subject_string_final' not in user_data.columns:
        return 'N/A'
    
    # 1. Filtrer les livres bien notés (≥ 7)
    liked_books = user_data[user_data['Book-Rating'] >= 7]
    
    if liked_books.empty:
        return 'Aucune préférence'
    
    # 2. Extraire tous les genres
    all_genres = []
    for genres_string in liked_books['subject_string_final'].dropna():
        if genres_string and str(genres_string) != 'nan':
            # Séparer les genres (assumons qu'ils sont séparés par des virgules)
            genres_list = [g.strip() for g in str(genres_string).split(',')]
            all_genres.extend(genres_list)
    
    # 3. Compter les fréquences
    if not all_genres:
        return 'Genres non disponibles'
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    # 4. Prendre les 2 genres les plus fréquents
    top_genres = [genre for genre, count in genre_counts.most_common(2)]
    
    return ', '.join(top_genres)

# Configure the Streamlit app
st.set_page_config(
    page_title="📚 Book Recommender System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data and models
try:
    knn_model, book_titles, books_df, svd_model, X_final, books_df_knn = load_model_and_data()
    
    # Simple vérification que les données sont chargées
    total_books = books_df['Book-Title'].nunique()
    total_users = books_df['User-ID'].nunique()

    st.sidebar.divider()
    st.sidebar.header("👤 Login")

    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "Invité"

    # User selection
    user_ids = ["Invité"] + list(books_df["User-ID"].unique())

    # Selectbox dans la sidebar
    selected_user = st.sidebar.selectbox(
        "Choisir un ID:",
        options=user_ids,
        index=user_ids.index(st.session_state.user_id),
        key="sidebar_user_selector"
    )

    # Mettre à jour l'état si changement
    if selected_user != st.session_state.user_id:
        st.session_state.user_id = selected_user
        # Initialiser l'historique pour le nouvel utilisateur
        if "history" not in st.session_state:
            st.session_state["history"] = []

    # Affichage de l'utilisateur actuel
    if st.session_state.user_id == "Invité":
        st.sidebar.info("👋 Bienvenue !")
        st.sidebar.write("Sélectionner un utilisateur pour avoir des recommandations personnalisées")
    else:
        st.sidebar.success(f"✅ Logged in as: **{st.session_state.user_id}**")
    
        # Afficher quelques stats utilisateur dans la sidebar
        user_data = books_df[books_df["User-ID"] == st.session_state.user_id]
        if not user_data.empty:
            user_stats = {
                "Livres évalués": len(user_data),
                "Note moyenne": f"{user_data['Book-Rating'].mean():.1f}",
                "Genre favoris": get_favorite_genres(user_data)
            }
        
            st.sidebar.write("**📊 Vos statistiques:**")
            for key, value in user_stats.items():
                st.sidebar.write(f"- {key}: {value}")

    # Bouton de déconnexion
    if st.session_state.user_id != "Guest user":
        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            st.session_state.user_id = "Guest user"
            st.rerun()

    st.sidebar.divider()

except Exception as e:
    st.error(f"❌ Erreur chargement des données: {e}")
    st.info("💡 Soyez sur que vos artefacts soient proprement chargés")
    st.stop()

# Title and introduction
st.title("📚 Book Recommender System")
st.markdown("### Trouvez votre prochain livre favori avec notre système de recommandation !")

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = "Guest user"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Par défaut, Tab 0 est actif

# Tabs for different sections
login_tab = f"Logout / {st.session_state.user_id}" if st.session_state.user_id != "Guest user" else "Login / Guest"
tabsvd, tab_bookstore, tab3, tab5, tab6, tab_api, tab4 = st.tabs([
    "🧑‍💻 Mes recommandations",
    "📚 Recommandations par livre",
    "🔍 Recherche",
    "📈 Livres populaires",
    "⭐ Les livres les mieux notés",
    "🧪 Test API",
    "📊 A propos"
])

# Tab contents
with tabsvd:
    show_user_recommendations(books_df, svd_model)

with tab_bookstore:
    show_bookstore_demo()

with tab3:
    show_search_tab()

with tab5:
    show_popular_books(books_df)

with tab6:
    show_top_rated_books(books_df)

with tab_api:
    show_test_api_tab()

with tab4:
    show_about_tab()


# Dataset Builder déplacé SOUS les onglets
st.divider()
st.markdown("---")


# Footer
FOOTER_HTML = """
    <hr style="margin-top: 50px; border: none; border-top: 1px solid #ccc;" />
    <div style="text-align: center; color: gray; font-size: 14px; margin-top: 10px;">
        Made with ❤️ using 
        <a href="https://streamlit.io" target="_blank" style="text-decoration: none; color: #4CAF50;">
            Streamlit
        </a>
    </div>
"""

