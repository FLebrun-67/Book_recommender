# tabs/tab5.py
import streamlit as st
from utils import render_aligned_image

def show_popular_books(books_df):
    """Display popular books tab."""
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
                render_aligned_image(book_image, book_title),
                unsafe_allow_html=True
            )