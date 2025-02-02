""" Trains a book recommender system using a KNN model. """

import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from surprise import Dataset
from surprise import Reader
from surprise import SVD
#from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
#from surprise.accuracy import rmse, mae
from surprise import accuracy


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
    ratings = data["Book-Rating"]
    sparse_matrix = coo_matrix((ratings, (users, books)))
    print(f"User-item matrix created with dimensions: {sparse_matrix.shape}.")
    return sparse_matrix, data["Book-Title"].astype("category").cat.categories

def preprocess_data_for_knn(data):
    """Preprocesses the dataset by grouping books by ISBN for KNN."""
    print("\nStep 1.1: Preprocessing data for KNN...")
    data_grouped = data.groupby("ISBN").agg({
    "Book-Title": "first",
    "Book-Author": "first",
    "Final_Tags": lambda x: ", ".join(set(x.dropna())),  # Concaténer les tags uniques
    "Book-Rating": "mean",  # Moyenne des notes
    "Cluster_hdbscan": "first"
}).reset_index()
    return data_grouped

def train_knn_model_with_metadata(data):
    """Trains a KNN model using metadata (tags + authors)"""
    print("\nStep 4: Initializing and training the enriched KNN model (with tags and authors)...")
    
    # Vectorization of tags and authors
    data["Final_Tags"] = data["Final_Tags"].fillna("")
    data["Book-Author"] = data["Book-Author"].fillna("")

    tfidf_authors = TfidfVectorizer(stop_words="english", max_features=100)
    X_authors = tfidf_authors.fit_transform(data["Book-Author"]).toarray()

    X_numeric = data[["Book-Rating"]].fillna(data["Book-Rating"]).to_numpy()

    X_clusters = data[["Cluster_hdbscan"]].to_numpy()

    if issparse(X_authors):  # Vérifie si c'est une matrice sparse
        X_authors = X_authors.toarray()  # Convertit en array dense

    # Combine all features
    X_final = np.hstack((X_authors, X_numeric, X_clusters))
    print(f"Feature matrix created with shape: {X_final.shape}")

    # Normalisation L2 de la matrice X_final
    X_final_normalized = normalize(X_final, norm='l2')

    # Train the KNN model
    knn_model = NearestNeighbors(metric="manhattan", algorithm="brute", n_neighbors=10)
    knn_model.fit(X_final_normalized)
    print("Enriched KNN model trained successfully.")
    return knn_model, X_final_normalized


def train_svd_model(data):
    """Trains on SVD model using surprise library"""
    print("\nStep 6: Training the SVD model ... ")
    reader = Reader(rating_scale=(data['Book-Rating'].min(), data['Book-Rating'].max()))
    data = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02)  # Paramètres optimisés
    svd.fit(trainset)

    # Évaluer sur l'ensemble d'entraînement
    train_predictions = svd.test(trainset.build_testset())
    train_rmse = accuracy.rmse(train_predictions, verbose=False)
    train_mae = accuracy.mae(train_predictions, verbose=False)

    # Évaluer sur l'ensemble de test
    test_predictions = svd.test(testset)
    test_rmse = accuracy.rmse(test_predictions, verbose=False)
    test_mae = accuracy.mae(test_predictions, verbose=False)

    # Résultats des performances
    comparison_results = pd.DataFrame({
        "Ensemble": ["Entraînement", "Test"],
        "RMSE": [train_rmse, test_rmse],
        "MAE": [train_mae, test_mae]
    })

    comparison_results

    return svd

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
    data_file_path = "./data/dataset_final3.csv"   # cleaned_data.csv
    artifacts_path = "artifacts/"

    # Load and inspect data
    book_df = load_data(data_file_path)
    data_overview(book_df)
    book_df_knn = preprocess_data_for_knn(book_df)


    _ , book_titles = create_user_item_matrix(book_df)

    # Train KNN model & SVD model
    # Train KNN model with metadata
    knn_model, X_final_normalized = train_knn_model_with_metadata(book_df_knn)
    svd_model = train_svd_model(book_df)

    # Save artifacts
    save_artifacts(
        artifacts_path,
        knn_model=knn_model,
        svd_model=svd_model,
        book_titles=book_titles,
        X_final=X_final_normalized,
        book_df=book_df,
        book_df_knn=book_df_knn
    )

    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()
