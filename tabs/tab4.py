# tabs/tab4.py
import streamlit as st
import plotly.express as px

def show_visualizations(books_df):
    """Display visualizations tab."""
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