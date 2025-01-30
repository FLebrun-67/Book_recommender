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


@st.cache_resource
def load_model_and_data():
    """Load the pre-trained model and data."""
    knn_model = pickle.load(open("artifacts/knn_model.pkl", "rb"))
    book_titles = pickle.load(open("artifacts/book_titles.pkl", "rb"))
    books_df = pickle.load(open("artifacts/book_df.pkl", "rb"))
    svd_model = pickle.load(open('artifacts/svd_model.pkl', 'rb'))
    X_final = pickle.load(open('artifacts/X_final.pkl', 'rb'))
    books_df_knn = pickle.load(open('artifacts/book_df_knn.pkl', 'rb'))

    return knn_model, book_titles, books_df, svd_model, X_final, books_df_knn

knn_model, book_titles, books_df, svd_model, X_final, books_df_knn = load_model_and_data()

def fetch_poster(book_list):
    """Fetch the poster URLs for the given list of books."""
    book_names = []
    poster_urls = []
    default_image = "https://via.placeholder.com/150"

    for book in book_list:
        if book in books_df["Book-Title"].values:
            book_names.append(book)
            idx = books_df[books_df["Book-Title"] == book].index
            if len(idx) > 0:
                img_url = books_df.loc[idx[0], "Image-URL-L"]
                poster_urls.append(
                    img_url
                    if pd.notna(img_url) and img_url.startswith("http")
                    else default_image
                )
            else:
                poster_urls.append(default_image)
        else:
            book_names.append(book)
            poster_urls.append(default_image)

    return book_names, poster_urls

def recommend_book_knn(book_name, books_df_knn, knn_model, X_final):
    """
    Recommends books based on the enriched KNN model.
    
    Args:
        book_titles (str): Name of the selected book.
        book_df_knn (pd.DataFrame): Grouped dataset with metadata.
        knn_model (NearestNeighbors): Trained KNN model.
        X_final (np.array): Feature matrix used for training the KNN model.

    Returns:
        book_names (list): List of recommended book titles.
        poster_urls (list): List of poster URLs for the recommended books.
    """
    try:
        # Vérifier si le livre existe
        if book_name not in books_df_knn["Book-Title"].values:
            st.error("Selected book doesn't exist.")
            return [], []

        # Obtenir l'indice du livre sélectionné
        book_idx = books_df_knn[books_df_knn["Book-Title"] == book_name].index[0]

        # Trouver les voisins avec le modèle KNN
        distances, suggestions = knn_model.kneighbors([X_final[book_idx]], n_neighbors=10)  # Inclure un voisin supplémentaire pour filtrer le livre lui-même

        # Récupérer les informations des livres recommandés
        books_list = [books_df_knn.iloc[suggestion_id]["Book-Title"] for suggestion_id in suggestions[0]]

        books_list = [book for book in books_list if book != book_name]

        book_names, poster_urls = fetch_poster(
            books_list[:10]
        )  # Limiter à 10 livres après filtrage
        return book_names, poster_urls
    except Exception as e:
        st.error("Recommendation display error (KNN): " + str(e))
        return [], []


def recommend_book_svd(user_id, n_recommendations=10):
    """Recommend books for a given user using the SVD model."""
    if user_id not in books_df["User-ID"].unique():
        popular_books = (
            books_df.groupby("Book-Title")["New-Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n_recommendations)
        )
        book_name, poster_url = fetch_poster(popular_books.index.tolist())
        return list(zip(book_name, popular_books.values, poster_url))

    all_books = books_df["Book-Title"].unique()
    user_books = books_df[books_df["User-ID"] == user_id]["Book-Title"].unique()
    books_to_predict = [book for book in all_books if book not in user_books]
    if not books_to_predict:
        st.warning("No books to find for this user.")

    predictions = [
        (book, svd_model.predict(user_id, book).est) for book in books_to_predict
    ]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[
        :n_recommendations
    ]

    book_name, poster_url = fetch_poster([book for book, _ in predictions])
    return list(zip(book_name, [rating for _, rating in predictions], poster_url))


def render_aligned_image(image_url, title, height=500):
    """Render an image with the given title and height."""
    return f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="{title}" style="height: {height}px; object-fit: contain; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 5px; font-size: 14px;"><b>{title}</b></p>
    </div>
    """


# Title and introduction
st.title("📚 Book Recommender System")
st.markdown("### Find your next favorite book with our recommendation system!")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🧑‍💻 Recommendations for users",
        "📚 Recommendations by books",
        "🔍 Search",
        "📊 Visualizations",
        "📈 Popular books",
        "⭐ Top-Rated books",
    ]
)

# Tab 1: Recommendations for users
with tab1:
    if "history" not in st.session_state:
        st.session_state["history"] = []
# Recommendation system interface
    st.subheader("📚 Find You and Get Recommendations")
    user_ids = list(books_df['User-ID'].unique()) + ["Invited Users"]
    selected_user = st.selectbox("Find your user ID:", user_ids)

    if st.button("Show Recommendations", key="recommendations_svd"):
        with st.spinner("Loading recommendations..."):
            recommendations = recommend_book_svd(selected_user)

        if recommendations:
            st.session_state["history"].append(
                {
                    "type": "user",
                    "user": selected_user,
                    "recommendations": [
                        book_name for book_name, _, _ in recommendations
                    ],
                }
            )
            st.subheader("🔍 Recommendations for you:")
            rows = [recommendations[:5], recommendations[5:10]]

            for i, row in enumerate(rows):
                cols = st.columns(len(row))
                for idx, col in enumerate(cols):
                    book_name, rating, poster = row[idx]
                    col.markdown(
                        render_aligned_image(poster, book_name), unsafe_allow_html=True
                    )
        else:
            st.warning("No recommendations found for this user.")

    if st.checkbox("View recommendation history", key="history_user"):
        st.subheader("📜 History of recommendations")
        for entry in st.session_state["history"]:
            if (
                entry.get("type") == "user"
            ):
                st.write(
                    f"**User : {entry['user']}** → {', '.join(entry['recommendations'])}"
                )

# Tab 2: Recommendations by books
with tab2:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    # Afficher les recommandations basées sur le livre (KNN)
    st.subheader("📚 Find a Book and Get Recommendations!")
    selected_book = st.selectbox("Write or find a Book :", book_titles)

    if st.button("Display recommendations (KNN)", key="recommendations_knn"):
        with st.spinner("Loading recommendations..."):
            book_list, poster_list = recommend_book_knn(selected_book, books_df_knn, knn_model, X_final)

        if book_list:
            st.session_state["history"].append(
                {"type": "book", "book": selected_book, "recommendations": book_list}
            )
            st.subheader("🔍 Recommendations for you:")
            st.markdown(f" ### Books similar to : {selected_book}")
            rows = [book_list[:5], book_list[5:10]]
            poster_rows = [poster_list[:5], poster_list[5:10]]

            for i, (row_books, row_posters) in enumerate(zip(rows, poster_rows)):
                cols = st.columns(len(row_books))
                for idx, col in enumerate(cols):
                    col.markdown(
                        render_aligned_image(row_posters[idx], row_books[idx]),
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No recommendations found for this book.")

    if st.checkbox("View recommendation history", key="history_book"):
        st.subheader("📜 History of recommendations")
        for entry in st.session_state["history"]:
            if entry.get("type") == "book":
                st.write(f"**{entry['book']}** → {', '.join(entry['recommendations'])}")

with tab3:
    st.subheader("🔍 Search for a book")

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

    st.subheader("🎲 Discover a random book")
    if st.button("Discover a random book"):
        random_book = books_df.sample(1)
        st.write(f"📖 Discover: **{random_book.iloc[0]['Book-Title']}**")
        st.image(random_book.iloc[0]["Image-URL-L"], use_container_width=True)

# Tab 4: Visualizations
with tab4:
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
        popular_books = (
            books_df.sort_values(by="Rating-Count", ascending=False)
            .drop_duplicates(subset="Book-Title")
            .head(10)
        )
        fig = px.bar(
            popular_books,
            x="Book-Title",
            y="Rating-Count",
            title="Most Popular Books",
            labels={"Book-Title": "Title", "Rating-Count": "Number of Ratings"},
            template="plotly_dark",
        )
        st.plotly_chart(fig)


# Tab 5: Popular books
with tab5:
    st.subheader("📈 Popular books")
    popular_books = (
        books_df.sort_values(by="Rating-Count", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )

    cols = st.columns(min(5, len(popular_books)))
    for idx, col in enumerate(cols):
        book_title = popular_books.iloc[idx]["Book-Title"]
        book_image = popular_books.iloc[idx]["Image-URL-L"]
        with col:
            col.markdown(
                render_aligned_image(book_image, book_title), unsafe_allow_html=True
            )

# Tab 6: Top-rated books
with tab6:
    st.subheader("⭐ Top-rated books")
    top_rated_books = (
        books_df.sort_values(by="Weighted-Rating", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )
    cols = st.columns(min(5, len(top_rated_books)))
    for idx, col in enumerate(cols):
        book_title = top_rated_books.iloc[idx]["Book-Title"]
        book_image = top_rated_books.iloc[idx]["Image-URL-L"]
        with col:
            col.markdown(
                render_aligned_image(book_image, book_title), unsafe_allow_html=True
            )

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
