import streamlit as st
from utils import load_model_and_data
from tabs.tab_svd import show_user_recommendations
from tabs.tab3 import show_search_tab
from tabs.tab4 import show_about_tab
from tabs.tab5 import show_popular_books
from tabs.tab6 import show_top_rated_books
from tabs.tab_test_api import show_test_api_tab
from tabs.tab_bookstore_demo import show_bookstore_demo
from tabs.tab_dataset_builder import show_dataset_builder_tab

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
    st.sidebar.header("👤 User Login")

    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "Guest user"

    # User selection
    user_ids = ["Guest user"] + list(books_df["User-ID"].unique())

    # Selectbox dans la sidebar
    selected_user = st.sidebar.selectbox(
        "Choose a user:",
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
    if st.session_state.user_id == "Guest user":
        st.sidebar.info("👋 Welcome, Guest!")
        st.sidebar.write("Select a user to get personalized recommendations")
    else:
        st.sidebar.success(f"✅ Logged in as: **{st.session_state.user_id}**")
    
        # Afficher quelques stats utilisateur dans la sidebar
        user_data = books_df[books_df["User-ID"] == st.session_state.user_id]
        if not user_data.empty:
            user_stats = {
                "Books rated": len(user_data),
                "Average rating": f"{user_data['Book-Rating'].mean():.1f}",
                "Favorite genre": user_data.groupby('Book-Title').first().get('Final_Tags', 'N/A') if 'Final_Tags' in books_df.columns else 'N/A'
            }
        
            st.sidebar.write("**📊 Your stats:**")
            for key, value in user_stats.items():
                if key != "Favorite genre":  # Éviter l'erreur si Final_Tags n'existe pas
                    st.sidebar.write(f"- {key}: {value}")

    # Bouton de déconnexion
    if st.session_state.user_id != "Guest user":
        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            st.session_state.user_id = "Guest user"
            st.rerun()

    st.sidebar.divider()

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure your artifacts are properly generated")
    st.stop()

# Title and introduction
st.title("📚 Book Recommender System")
st.markdown("### Find your next favorite book with our recommendation system!")

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = "Guest user"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Par défaut, Tab 0 est actif

# Tabs for different sections
login_tab = f"Logout / {st.session_state.user_id}" if st.session_state.user_id != "Guest user" else "Login / Guest"
tabsvd, tab_bookstore, tab3, tab5, tab6, tab_api, tab4 = st.tabs([
    "🧑‍💻 My Recommendations",
    "📚 Recommendations by books",
    "🔍 Search",
    "📈 Popular books",
    "⭐ Top-Rated books",
    "🧪 Test API",
    "📊 About"
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

# Section Dataset Builder (en dessous des onglets)
with st.expander("🏗️ **Dataset Builder** - Configuration du système", expanded=False):
    st.markdown("### Gestion du dataset pour les recommandations")
    show_dataset_builder_tab()

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