import streamlit as st
from utils import load_model_and_data
from tabs.tab0 import show_login
from tabs.tab1 import show_user_recommendations
from tabs.tab2 import show_book_recommendations
from tabs.tab3 import show_search_tab
from tabs.tab5 import show_popular_books
from tabs.tab6 import show_top_rated_books

# Configure the Streamlit app
st.set_page_config(
    page_title="📚 Book Recommender System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data and models
knn_model, book_titles, books_df, svd_model, X_final, books_df_knn = load_model_and_data()

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
tab0, tab1, tab2, tab3, tab5, tab6, tab4 = st.tabs([
    "🦄" + login_tab,
    "🧑‍💻 My Recommendations",
    "📚 Recommendations by books",
    "🔍 Search",
    "📈 Popular books",
    "⭐ Top-Rated books",
    "📊 About",
])

# Tab contents
with tab0:  # Onglet de connexion / login
    show_login(books_df)

with tab1:
    show_user_recommendations(books_df, svd_model)

with tab2:
    show_book_recommendations(books_df, book_titles, books_df_knn, knn_model, X_final)

with tab3:
    show_search_tab(books_df)

with tab5:
    show_popular_books(books_df)

with tab6:
    show_top_rated_books(books_df)


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
st.markdown(FOOTER_HTML, unsafe_allow_html=True)