import pickle
import streamlit as st
import plotly.express as px

# Configure the Streamlit app
st.set_page_config(page_title="Book Recommender System", layout="wide")

# Load data and model
@st.cache_resource
def load_model_and_data():
    knn_model = pickle.load(open("artifacts/knn_model.pkl", "rb"))
    book_titles = pickle.load(open("artifacts/book_titles.pkl", "rb"))
    books_df = pickle.load(open("artifacts/book_df.pkl", "rb"))
    book_sparse = pickle.load(
        open("artifacts/sparse_user_item_matrix_full_csr.pkl", "rb")
    )
    return knn_model, book_titles, books_df, book_sparse

knn_model, book_titles, books_df, book_sparse = load_model_and_data()

# Initialize session states for history and ratings
if "history" not in st.session_state:
    st.session_state["history"] = []

if "ratings" not in st.session_state:
    st.session_state["ratings"] = []

# Fetch posters for books based on suggestions
def fetch_poster(suggestion):
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

# Generate book recommendations
def recommend_book(book_name):
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

# Title and introduction
st.title("📚 Book Recommender System")
st.markdown(
    "### Find your next favorite book with our recommendation system!"
)

# General statistics
st.markdown("### General Statistics")
col1, col2 = st.columns(2)
col1.metric("Total Books", len(book_titles))
col2.metric("Total Users", book_sparse.shape[0])

# Display popular books
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
        st.image(book_image, use_container_width=True)
        st.caption(f"**{book_title}**")

st.divider()

# Recommendation system interface
st.subheader("📚 Find and Get Recommendations")
selected_book = st.selectbox(
    "Type or select a book from the dropdown menu", book_titles, key="selectbox_knn"
)

if st.button("Show Recommendations"):
    with st.spinner("Loading recommendations..."):
        recommended_books, poster_url = recommend_book(selected_book)
    if recommended_books:
        st.subheader("🔍 Recommendations for You:")
        st.session_state["history"].append(
            {"book": selected_book, "recommendations": recommended_books}
        )
        cols = st.columns(len(recommended_books))
        for idx, col in enumerate(cols):
            if idx > 0:  # Skip the selected book itself
                with col:
                    st.image(poster_url[idx], use_container_width=True)
                    st.caption(recommended_books[idx])

        # Rating the recommendations
        st.markdown("### Rate the Recommendations")
        rating = st.slider("How would you rate these recommendations?", 1, 5, 3)
        if st.button("Submit Your Feedback"):
            st.session_state["ratings"].append(
                {"book": selected_book, "rating": rating}
            )
            st.success(f"Thank you for your feedback: {rating} stars!")
    else:
        st.warning("No recommendations found. Try selecting another book.")

# Recommendation history
if st.checkbox("Show recommendation history"):
    st.subheader("📜 Recommendation History")
    for entry in st.session_state["history"]:
        st.write(f"**{entry['book']}** → {', '.join(entry['recommendations'])}")

# Advanced search
st.sidebar.header("🔎 Search for a book")
search_query = st.sidebar.text_input("Search for a book by keyword")
if search_query:
    filtered_books = [
        book for book in book_titles if search_query.lower() in book.lower()
    ]
    if filtered_books:
        st.sidebar.write("Books found:")
        st.sidebar.write(filtered_books)
    else:
        st.sidebar.warning("No books found matching your search.")

# Random book discovery
if st.sidebar.button("Discover a random book"):
    random_book = books_df.sample(1)
    st.sidebar.write(f"📖 Discover: **{random_book.iloc[0]['Book-Title']}**")
    st.sidebar.image(random_book.iloc[0]["Image-URL-L"], use_container_width=True)

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
    top_books = books_df.nlargest(10, "Rating-Count")
    fig = px.bar(
        top_books,
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