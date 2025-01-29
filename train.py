""" Trains a book recommender system using a KNN model. """

import os
import pickle
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
#modif for git: rajout des librairies surprise for SVD model
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV


def load_data(file_path):
    """Loads the dataset from a CSV file."""
    print("Step 1: Loading data...")
    data = pd.read_csv(file_path)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def data_overview(data):
    """Prints an overview of the data."""
    print("\nStep 2: Data overview...")
    print(data.head())
    print("\nData information:")
    print(data.info())


def create_user_item_matrix(data):
    """Creates a sparse user-item matrix."""
    print("\nStep 3: Creating user-item matrix...")
    users = data["User-ID"].astype("category").cat.codes
    books = data["Book-Title"].astype("category").cat.codes
    ratings = data["New-Rating"]   #New-Rating
    sparse_matrix = coo_matrix((ratings, (users, books)))
    print(f"User-item matrix created with dimensions: {sparse_matrix.shape}.")
    return sparse_matrix, data["Book-Title"].astype("category").cat.categories


def train_knn_model(sparse_matrix):
    """Initializes and trains the KNN model."""
    print("\nStep 4: Initializing and training the KNN model...")
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=8)
    knn_model.fit(sparse_matrix.T)
    print("KNN model trained successfully.")
    return knn_model


def find_similar_books(knn_model, book_titles, sparse_matrix, example_book_index=0):
    """Finds similar books using the trained KNN model."""
    print("\nStep 5: Finding similar books...")
    sparse_matrix_csr = sparse_matrix.tocsr()
    print(f"Finding similar books for: {book_titles[example_book_index]}...")
    distances, indices = knn_model.kneighbors(
        sparse_matrix_csr.T[example_book_index], n_neighbors=5
    )
    similar_books = book_titles[indices.flatten()]
    print("Similar books found:")
    for i, book in enumerate(similar_books, 1):
        print(f"{i}. {book}")

def train_svd_model(data):
    """Trains on SVD model using surprise library"""
    print("\nStep 6: Training the SVD model ... ")
    reader = Reader(rating_scale=(1, 7))
    dataset = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'New-Rating']], reader)

    param_grid = {
        'n_factors': [20, 50, 100],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.05, 0.1]
    }

    grid_search_SVD = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
    grid_search_SVD.fit(dataset)

    best_params = grid_search_SVD.best_params['rmse']
    optimized_SVDmodel = SVD(
        n_factors=best_params['n_factors'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all']
    )

    trainset = dataset.build_full_trainset()
    optimized_SVDmodel.fit(trainset)
    
    print("Best SVD parameters:", best_params)
    print("SVD model trained successfully.")
    return optimized_SVDmodel

def save_artifacts(artifacts_path, **artifacts):
    """Saves artifacts to the specified directory."""
    print("\nStep 7: Saving artifacts...")
    os.makedirs(artifacts_path, exist_ok=True)
    for name, artifact in artifacts.items():
        file_path = os.path.join(artifacts_path, f"{name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
            print(f"{name} saved to '{file_path}'.")


def main():
    """Main function to train the book recommender system."""
    # File paths
    data_file_path = "./data/dataset.csv"   # cleaned_data.csv
    artifacts_path = "artifacts/"

    # Load and inspect data
    book_df = load_data(data_file_path)
    data_overview(book_df)

    # Create user-item matrix
    sparse_matrix, book_titles = create_user_item_matrix(book_df)

    # Train KNN model & SVD model
    knn_model = train_knn_model(sparse_matrix)
    svd_model = train_svd_model(book_df)

    # Find similar books
    find_similar_books(knn_model, book_titles, sparse_matrix)

    # Save artifacts
    save_artifacts(
        artifacts_path,
        knn_model=knn_model,
        svd_model=svd_model,
        book_titles=book_titles,
        book_df=book_df,
        sparse_user_item_matrix_full_csr=sparse_matrix.tocsr(),
    )

    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()
