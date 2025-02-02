# tabs/tab3.py
import streamlit as st
from utils import render_aligned_image

@st.fragment
def show_search_tab(books_df):
    """Display search functionality tab."""
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
                    "Book-Rating": "Rating",
                    "Rating-Count": "Number of Ratings",
                }
            )

            st.dataframe(
                filtered_books.sort_values(by="Rating", ascending=False),
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