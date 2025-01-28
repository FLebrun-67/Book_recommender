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

# Custom styles
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color:white;
            border: None;
            border-radius: 5px;
            height: 40px;
            width: 200px;
        }
        div.stButton > button:first-child:hover {
            background-color: #45a049;
        }
        .stMetric {
            font-size: 16px;
            font-weight: bold;
        }
        img {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
    return knn_model, book_titles, books_df, book_sparse


knn_model, book_titles, books_df, book_sparse = load_model_and_data()


# Helper functions
def fetch_poster(suggestion):
    """Fetches book posters based on the suggestions."""
    book_names = []
    poster_urls = []

    for book_id in suggestion:
        if 0 <= book_id < len(book_titles):
            book_names.append(book_titles[book_id])
            idx = books_df[books_df["Book-Title"] == book_titles[book_id]].index
            if len(idx) > 0:
                poster_urls.append(books_df.loc[idx[0], "Image-URL-L"])
            else:
                poster_urls.append("https://via.placeholder.com/150")  # Default image
        else:
            book_names.append(None)
            poster_urls.append("https://via.placeholder.com/150")

    return book_names, poster_urls


def recommend_book(book_name):
    """Recommends books based on the selected book."""
    try:
        if book_name not in book_titles:
            st.error("The selected book does not exist in our data.")
            return [], []

        book_id = book_titles.get_loc(book_name)  # Get book index
        distances, suggestions = knn_model.kneighbors(
            book_sparse.T[book_id], n_neighbors=10
        )

        books_list, poster_url = fetch_poster(suggestions[0])
        return books_list, poster_url
    except Exception as e:
        st.error("Error generating recommendations: " + str(e))
        return [], []


# Title and introduction (above tabs)
st.title("📚 Book Recommender System")
st.markdown("### Find your next favorite book with our recommendation system!")

# General Statistics (above tabs)
st.subheader("📈 General Statistics")
col1, col2 = st.columns(2)
col1.metric("Total Books", len(book_titles))
col2.metric("Total Users", book_sparse.shape[0])

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["📚 Recommendations", "🔍 Search", "📊 Visualizations", "📈 Popular Books"]
)

# Tab 1: Recommendations
with tab1:
    st.subheader("📚 Find and Get Recommendations")
    selected_book = st.selectbox(
        "Type or select a book from the dropdown menu", book_titles, key="selectbox_knn"
    )
    if st.button("Show Recommendations"):
        with st.spinner("Loading recommendations..."):
            recommended_books, poster_url = recommend_book(selected_book)
        if recommended_books:
            st.subheader("🔍 Recommendations for You:")
            cols = st.columns(len(recommended_books))
            for idx, col in enumerate(cols):
                with col:
                    st.image(poster_url[idx], use_container_width=True)
                    st.caption(recommended_books[idx])
        else:
            st.warning("No recommendations found. Try selecting another book.")

# Tab 2: Search
with tab2:
    st.subheader("🔍 Search for a Book")
    search_query = st.text_input("Search for a book by keyword")
    if search_query:
        filtered_books = [
            book for book in book_titles if search_query.lower() in book.lower()
        ]
        if filtered_books:
            st.write("Books Found:")
            st.write(filtered_books)
        else:
            st.warning("No books found matching your search.")

# Tab 3: Visualizations
with tab3:
    st.subheader("📊 Visualizations")
    if st.checkbox("Show Rating Distribution"):
        fig = px.histogram(
            books_df,
            x="Book-Rating",
            nbins=20,
            title="Book Ratings Distribution",
            labels={"Book-Rating": "Rating"},
            template="plotly_dark",
        )
        st.plotly_chart(fig)

    if st.checkbox("Show User Rating Distribution"):
        user_ratings = book_sparse.getnnz(axis=1)
        fig = px.histogram(
            x=user_ratings,
            nbins=20,
            title="User Rating Distribution",
            labels={"x": "Number of Ratings", "y": "Number of Users"},
        )
        st.plotly_chart(fig)

# Tab 4: Popular Books
with tab4:
    st.subheader("📈 Popular Books")
    popular_books = books_df.sort_values(by="Rating-Count", ascending=False).head(10)
    cols = st.columns(min(5, len(popular_books)))  # Maximum 5 columns
    for idx, col in enumerate(cols):
        book_title = popular_books.iloc[idx]["Book-Title"]
        book_image = popular_books.iloc[idx]["Image-URL-L"]
        with col:
            st.image(book_image, use_container_width=True)
            st.caption(f"**{book_title}**")

    st.divider()

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
            st.image(book_image, use_container_width=True)
            st.caption(f"**{book_title}**")

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
