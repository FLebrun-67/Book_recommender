import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Step 1: Load the data
print("Step 1: Loading data...")
book_df = pd.read_csv("./data/cleaned_data.csv")
print(f"Data loaded with {book_df.shape[0]} rows and {book_df.shape[1]} columns.")

# Step 2: Overview of the data
print("\nStep 2: Data overview...")
print(book_df.head())
print("\nData information:")
print(book_df.info())

# Step 3: Create user-item matrix
print("\nStep 3: Creating user-item matrix...")
users = book_df["User-ID"].astype("category").cat.codes
books = book_df["Book-Title"].astype("category").cat.codes
ratings = book_df["Book-Rating"]

sparse_user_item_matrix_full = coo_matrix((ratings, (users, books)))
print(
    f"User-item matrix created with dimensions: {sparse_user_item_matrix_full.shape}."
)

# Step 4: Initialize and train the KNN model
print("\nStep 4: Initializing and training the KNN model...")
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=8)
knn_model.fit(sparse_user_item_matrix_full.T)
print("KNN model trained successfully.")

# Step 5: Find similar books
print("\nStep 5: Finding similar books...")
example_book_index = 0  # By default, the first book in the matrix
sparse_user_item_matrix_full_csr = sparse_user_item_matrix_full.tocsr()

print(f"Finding similar books for: {book_df['Book-Title'].iloc[example_book_index]}...")
distances, indices = knn_model.kneighbors(
    sparse_user_item_matrix_full_csr.T[example_book_index], n_neighbors=5
)
book_titles = book_df["Book-Title"].astype("category").cat.categories
similar_books = book_titles[indices.flatten()]

print("Similar books found:")
for i, book in enumerate(similar_books, 1):
    print(f"{i}. {book}")

# Step 6: Save artifacts
print("\nStep 6: Saving artifacts...")
artifacts_path = "artifacts/"
with open(f"{artifacts_path}knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)
    print("KNN model saved to 'artifacts/knn_model.pkl'.")
with open(f"{artifacts_path}book_titles.pkl", "wb") as f:
    pickle.dump(book_titles, f)
    print("Book titles saved to 'artifacts/book_titles.pkl'.")
with open(f"{artifacts_path}book_df.pkl", "wb") as f:
    pickle.dump(book_df, f)
    print("Book DataFrame saved to 'artifacts/book_df.pkl'.")
with open(f"{artifacts_path}sparse_user_item_matrix_full_csr.pkl", "wb") as f:
    pickle.dump(sparse_user_item_matrix_full_csr, f)
    print(
        "Sparse user-item matrix saved to 'artifacts/sparse_user_item_matrix_full_csr.pkl'."
    )

print("\nAll artifacts have been saved successfully.")

print("\nScript completed successfully.")
