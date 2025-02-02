# tabs/tab6.py
import streamlit as st
from utils import render_aligned_image

def show_top_rated_books(books_df):
    """Display top-rated books tab."""
    st.subheader("⭐ Top-rated books")
    
    top_rated_books = (
        books_df.sort_values(by="Book-Rating", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )
    
    cols = st.columns(min(5, len(top_rated_books)))
    for idx, col in enumerate(cols):
        book_title = top_rated_books.iloc[idx]["Book-Title"]
        book_image = top_rated_books.iloc[idx]["Image-URL-L"]
        with col:
            col.markdown(
                render_aligned_image(book_image, book_title),
                unsafe_allow_html=True
            )