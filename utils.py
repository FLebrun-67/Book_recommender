import pickle
import pandas as pd
import streamlit as st

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

def fetch_poster(books_df, book_list):
    """Fetch the poster URLs for the given list of books."""
    book_names = []
    poster_urls = []
    book_descriptions = []
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
                # Récupération de la description du livre
                description = books_df.loc[idx[0], "Description"]
                book_descriptions.append(
                    description if pd.notna(description) else "Description non disponible"
                    )
            else:
                poster_urls.append(default_image)
                book_descriptions.append("Description non disponible")
        else:
            book_names.append(book)
            poster_urls.append(default_image)
            book_descriptions.append("Description non disponible")

    return book_names, poster_urls, book_descriptions

def recommend_book_knn(books_df, book_name, books_df_knn, knn_model, X_final):
    """Recommends books based on the enriched KNN model."""
    try:
        if book_name not in books_df_knn["Book-Title"].values:
            st.error("Selected book doesn't exist.")
            return [], []

        book_idx = books_df_knn[books_df_knn["Book-Title"] == book_name].index[0]
        distances, suggestions = knn_model.kneighbors([X_final[book_idx]], n_neighbors=10)
        books_list = [books_df_knn.iloc[suggestion_id]["Book-Title"] for suggestion_id in suggestions[0]]
        books_list = [book for book in books_list if book != book_name]
        
        book_names, poster_urls, book_descriptions = fetch_poster(books_df, books_list[:10])
        return book_names, poster_urls, book_descriptions
    except Exception as e:
        st.error("Recommendation display error (KNN): " + str(e))
        return [], []

def recommend_book_svd(user_id, books_df, svd_model, n_recommendations=10):
    """Recommend books for a given user using the SVD model."""
    if user_id not in books_df["User-ID"].unique():
        popular_books = (
            books_df.groupby("Book-Title")["Book-Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n_recommendations)
        )
        book_name, poster_url, book_descriptions = fetch_poster(books_df, popular_books.index.tolist())
        return list(zip(book_name, popular_books.values, poster_url, book_descriptions))

    all_books = books_df["Book-Title"].unique()
    user_books = books_df[books_df["User-ID"] == user_id]["Book-Title"].unique()
    books_to_predict = [book for book in all_books if book not in user_books]
    
    if not books_to_predict:
        st.warning("No books to find for this user.")
        return []

    predictions = [
        (book, svd_model.predict(user_id, book).est) for book in books_to_predict
    ]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    book_name, poster_url, book_descriptions = fetch_poster(books_df, [book for book, _ in predictions])
    return list(zip(book_name, [rating for _, rating in predictions], poster_url, book_descriptions))

def render_aligned_image(image_url, title, height=500):
    """Render an image with the given title and height."""
    return f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="{title}" style="height: {height}px; object-fit: contain; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 5px; font-size: 14px;"><b>{title}</b></p>
    </div>
    """