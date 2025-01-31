# tabs/tab2.py
import streamlit as st
from utils import recommend_book_knn, render_aligned_image

@st.fragment
def show_book_recommendations(books_df, book_titles, books_df_knn, knn_model, X_final):
    """Display recommendations by books tab."""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.subheader("📚 Find a Book and Get Recommendations!")
    selected_book = st.selectbox("Write or find a Book:", book_titles)

    if st.button("Display recommendations (KNN)", key="recommendations_knn"):
        with st.spinner("Loading recommendations..."):
            book_list, poster_list = recommend_book_knn(books_df, selected_book, books_df_knn, knn_model, X_final)

        if book_list:
            st.session_state["history"].append({
                "type": "book",
                "book": selected_book,
                "recommendations": book_list
            })
            
            st.subheader("🔍 Recommendations for you:")
            st.markdown(f"### Books similar to: {selected_book}")
            rows = [book_list[:5], book_list[5:10]]
            poster_rows = [poster_list[:5], poster_list[5:10]]

            for row_books, row_posters in zip(rows, poster_rows):
                cols = st.columns(len(row_books))
                for idx, col in enumerate(cols):
                    col.markdown(
                        render_aligned_image(row_posters[idx], row_books[idx]),
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No recommendations found for this book.")

    if st.checkbox("View recommendation history", key="history_book"):
        st.subheader("📜 History of recommendations")
        for entry in st.session_state["history"]:
            if entry.get("type") == "book":
                st.write(f"**{entry['book']}** → {', '.join(entry['recommendations'])}")