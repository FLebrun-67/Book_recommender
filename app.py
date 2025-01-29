""" Streamlit app for the Book Recommender System. """

import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

# Configure the Streamlit app
st.set_page_config(
    page_title="📚 Book Recommender System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load data and model
@st.cache_resource
def load_model_and_data():
    """Loads the KNN model and data artifacts."""
    knn_model = pickle.load(open("artifacts/knn_model.pkl", "rb"))
    book_titles = pickle.load(open("artifacts/book_titles.pkl", "rb"))
    books_df = pickle.load(open("artifacts/book_df.pkl", "rb"))
    book_sparse = pickle.load(
        open("artifacts/sparse_user_item_matrix_full_csr.pkl", "rb")
    )
    SVD_model = pickle.load(open('artifacts/svd_model.pkl', 'rb'))
    return knn_model, book_titles, books_df, book_sparse, SVD_model

knn_model, book_titles, books_df, book_sparse, SVD_model = load_model_and_data()

#modif gitflo de la fonction fetch_poster 
# Helper functions
def fetch_poster(book_list):
    """
    Récupère les affiches (couvertures) des livres.
    Si une couverture n'est pas trouvée, une image par défaut est utilisée.
    """
    book_names = []
    poster_urls = []
    default_image = "https://via.placeholder.com/150"  # URL par défaut pour une image manquante

    for book in book_list:
        if book in books_df["Book-Title"].values:
            book_names.append(book)
            # Récupérer l'index correspondant
            idx = books_df[books_df["Book-Title"] == book].index
            if len(idx) > 0:
                img_url = books_df.loc[idx[0], "Image-URL-L"]
                # Ajouter l'URL si valide, sinon utiliser l'image par défaut
                poster_urls.append(img_url if pd.notna(img_url) and img_url.startswith("http") else default_image)
            else:
                poster_urls.append(default_image)
        else:
            book_names.append(book)
            poster_urls.append(default_image)

    return book_names, poster_urls

#modif gitflo recommend_book_knn 
# Fonction pour recommander des livres (KNN)
def recommend_book_knn(book_name):
    try:
        if book_name not in book_titles:
            st.error("Le livre sélectionné n'existe pas dans nos données.")
            return [], []

        # Obtenir l'indice du livre sélectionné
        book_id = book_titles.get_loc(book_name)

        # Trouver les voisins avec KNN
        distances, suggestions = knn_model.kneighbors(
            book_sparse.T[book_id], n_neighbors=11  # Ajouter un voisin supplémentaire pour inclure le livre sélectionné
        )

        # Récupérer les informations des livres recommandés
        books_list = [book_titles[suggestion_id] for suggestion_id in suggestions[0]]

        # Exclure le livre sélectionné des résultats
        books_list = [book for book in books_list if book != book_name]

        # Récupérer les informations des posters pour les livres recommandés
        book_names, poster_urls = fetch_poster(books_list[:10])  # Limiter à 10 livres après filtrage
        return book_names, poster_urls
    except Exception as e:
        st.error("Erreur lors de la génération des recommandations (KNN) : " + str(e))
        return [], []

# Fonction pour recommander des livres (SVD)
def recommend_book_svd(user_id, n_recommendations=10):
    if user_id not in books_df['User-ID'].unique():
        # Recommandations générales pour les utilisateurs non enregistrés
        popular_books = books_df.groupby('Book-Title')['New-Rating'].mean().sort_values(ascending=False).head(n_recommendations)
        book_name, poster_url = fetch_poster(popular_books.index.tolist())
        return list(zip(book_name, popular_books.values, poster_url))

    all_books = books_df['Book-Title'].unique()
    user_books = books_df[books_df['User-ID'] == user_id]['Book-Title'].unique()
    books_to_predict = [book for book in all_books if book not in user_books]
    if not books_to_predict:
        st.warning("Aucun livre à prédire pour cet utilisateur.")

    predictions = [(book, SVD_model.predict(user_id, book).est) for book in books_to_predict]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    book_name, poster_url = fetch_poster([book for book, _ in predictions])
    return list(zip(book_name, [rating for _, rating in predictions], poster_url))

def render_aligned_image(image_url, title, height=500):
    """
    Génère un conteneur HTML aligné pour afficher une image et son titre.
    """
    return f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="{title}" style="height: {height}px; object-fit: contain; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 5px; font-size: 14px;"><b>{title}</b></p>
    </div>
    """


# Title and introduction (above tabs)
st.title("📚 Book Recommender System")
st.markdown("### Find your next favorite book with our recommendation system!")

# General Statistics (above tabs)
col1, col2 = st.columns(2)
col1.metric("Total Books", len(book_titles))
col2.metric("Total Users", book_sparse.shape[0])

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🧑‍💻 Recommendations for Users", "📚 Recommendations by Books", "🔍 Search", "📊 Visualizations", "📈 Popular Books", "⭐ Top-Rated Books"]
)

# Tab 1: Recommendations for users
with tab1:
    # Initialiser l'historique s'il n'existe pas
    if "history" not in st.session_state:
        st.session_state["history"] = []
# Recommendation system interface
    st.subheader("📚 Find and Get Recommendations")
    user_ids = list(books_df['User-ID'].unique()) + ["Utilisateur invité"]
    selected_user = st.selectbox("Choisissez un utilisateur :", user_ids)

    if st.button("Show Recommendations (SVD)", key="recommendations_svd"):
        with st.spinner("Loading recommendations..."):
            recommendations = recommend_book_svd(selected_user)

        if recommendations:
            # Ajouter les recommandations à l'historique avec une clé cohérente
            st.session_state["history"].append({
                "type": "user",  # Type de recommandation
                "user": selected_user,
                "recommendations": [book_name for book_name, _, _ in recommendations]
            })
            st.subheader("🔍 Recommendations for You:")
            # Divisez les recommandations en deux groupes de 5 pour afficher en deux lignes
            rows = [recommendations[:5], recommendations[5:10]]

            for i, row in enumerate(rows):
                st.markdown(f"**Ligne {i + 1}**")
                cols = st.columns(len(row))  # Créez des colonnes dynamiques
                for idx, col in enumerate(cols):
                    book_name, rating, poster = row[idx]
                    col.markdown(render_aligned_image(poster, book_name), unsafe_allow_html=True)
        else:
            st.warning("Aucune recommandation trouvée pour cet utilisateur.")

    # Afficher l'historique des recommandations
    if st.checkbox("Afficher l'historique des recommandations", key="history_user"):
        st.subheader("📜 Historique des recommandations")
        for entry in st.session_state["history"]:
            if entry.get("type") == "user":  # Filtrer les recommandations par utilisateur
                st.write(f"**Utilisateur : {entry['user']}** → {', '.join(entry['recommendations'])}")

# Tab 2: Recommendations by books
with tab2:
    # Initialiser l'historique s'il n'existe pas
    if "history" not in st.session_state:
        st.session_state["history"] = []
    # Afficher les recommandations basées sur le livre (KNN)
    st.subheader("Recommandations similaires basées sur un livre sélectionné")
    selected_book = st.selectbox("Tapez ou sélectionnez un livre dans le menu déroulant :", book_titles)

    if st.button("Afficher les recommandations (KNN)", key="recommendations_knn"):
        with st.spinner("Loading recommendations..."):
            book_list, poster_list = recommend_book_knn(selected_book)

        if book_list:
            st.session_state["history"].append({"type":"book", "book": selected_book, "recommendations": book_list})
            st.subheader("🔍 Recommendations for You:")
            st.markdown(f" ### Livres similaires à : {selected_book}")
            # Diviser les recommandations en deux lignes
            rows = [book_list[:5], book_list[5:10]]
            poster_rows = [poster_list[:5], poster_list[5:10]]

            for i, (row_books, row_posters) in enumerate(zip(rows, poster_rows)):
                st.markdown(f"**Ligne {i + 1}**")
                cols = st.columns(len(row_books))  # Créez des colonnes dynamiques
                for idx, col in enumerate(cols):
                    col.markdown(render_aligned_image(row_posters[idx], row_books[idx]), unsafe_allow_html=True)            
        else:
            st.warning("Aucune recommandation trouvée pour ce livre.")

    # Afficher l'historique des recommandations
    if st.checkbox("Afficher l'historique des recommandations", key="history_book"):
        st.subheader("📜 Historique des recommandations")
        for entry in st.session_state["history"]:
            if entry.get("type") == "book":  # Filtrer les recommandations par livre
                st.write(f"**{entry['book']}** → {', '.join(entry['recommendations'])}")

with tab3:
    st.subheader("🔍 Search for a Book")
    
    search_query = st.text_input("Search for a book by keyword")
    if search_query:
        unique_books = books_df.drop_duplicates(subset=["Book-Title"])
        filtered_books = unique_books[
            unique_books["Book-Title"].str.contains(search_query, case=False, na=False)
        ][["Book-Title", "Book-Author", "Book-Rating", "Rating-Count"]]

        if not filtered_books.empty:
            st.write(f"**{len(filtered_books)} books found:**")

            filtered_books = filtered_books.rename(
                columns={
                    "Book-Title": "Title",
                    "Book-Author": "Author",
                    "Book-Rating": "Average Rating",
                    "Rating-Count": "Number of Ratings",
                }
            )

            st.dataframe(
                filtered_books.sort_values(by="Average Rating", ascending=False),
                use_container_width=True,
                hide_index=True,
                height=500,
            )
        else:
            st.warning("No books found matching your search.")

    # Recherche aléatoire
    st.subheader("🎲 Discover a Random Book")
    if st.button("Discover a random book"):
        random_book = books_df.sample(1)
        st.write(f"📖 Discover: **{random_book.iloc[0]['Book-Title']}**")
        st.image(random_book.iloc[0]["Image-URL-L"], use_container_width=True)

#Tab 4 viz
with tab4:
# Statistics visualizations
    st.subheader("📊 Visualizations")
    if st.checkbox("Show rating distribution"):
        fig = px.histogram(
            books_df,
            x="Book-Rating",
            nbins=20,
            title="Book ratings distribution",
            labels={"Book-Rating": "Rating"},
            template="plotly_dark",
        )
        st.plotly_chart(fig)

    if st.checkbox("Show most popular books"):
        popular_books = books_df.sort_values(by="Rating-Count", ascending=False).drop_duplicates(subset="Book-Title").head(10)
        fig = px.bar(
            popular_books,
            x="Book-Title",
            y="Rating-Count",
            title="Most Popular Books",
            labels={"Book-Title": "Title", "Rating-Count": "Number of Ratings"},
            template="plotly_dark",
        )
        st.plotly_chart(fig)

    if st.checkbox("Show user rating distribution"):
        user_ratings = book_sparse.getnnz(axis=1)
        fig = px.histogram(
            x=user_ratings,
            nbins=20,
            title="User rating distribution",
            labels={"x": "Number of ratings", "y": "Number of users"},
        )
        st.plotly_chart(fig)

with tab5:
    # Display popular books
    st.subheader("📈 Popular Books")
    # Supprimer les doublons basés sur 'Book-Title'
    popular_books = books_df.sort_values(by="Rating-Count", ascending=False).drop_duplicates(subset="Book-Title").head(10)

    # Afficher les livres
    cols = st.columns(min(5, len(popular_books)))
    for idx, col in enumerate(cols):
        book_title = popular_books.iloc[idx]["Book-Title"]
        book_image = popular_books.iloc[idx]["Image-URL-L"]
        with col:
            col.markdown(render_aligned_image(book_image, book_title), unsafe_allow_html=True)

    st.divider()

with tab6:
    # Display top-rated books
    st.subheader("⭐ Top-Rated Books")
    top_rated_books = (
        books_df.sort_values(by="Book-Rating", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )
    cols = st.columns(min(5, len(top_rated_books)))  # Maximum 5 columns
    for idx, col in enumerate(cols):
        book_title = top_rated_books.iloc[idx]["Book-Title"]
        book_image = top_rated_books.iloc[idx]["Image-URL-L"]
        with col:
            col.markdown(render_aligned_image(book_image, book_title), unsafe_allow_html=True)

    st.divider()

# Footer
st.markdown(
    """
    <hr style="margin-top: 50px; border: none; border-top: 1px solid #ccc;" />
    <div style="text-align: center; color: gray; font-size: 14px; margin-top: 10px;">
        Made with ❤️ using <a href="https://streamlit.io" target="_blank" style="text-decoration: none; color: #4CAF50;">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True,
)
