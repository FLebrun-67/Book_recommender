""" Trains a book recommender system using a KNN model. """

import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
#modif for git: rajout des librairies surprise for SVD model
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise.model_selection import KFold


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

def preprocess_data_for_knn(data):
    """Preprocesses the dataset by grouping books by ISBN for KNN."""
    print("\nStep 1.1: Preprocessing data for KNN...")
    data_grouped = data.groupby("ISBN").agg({
    "Book-Title": "first",
    "Book-Author": "first",
    "Final_Tags": lambda x: ", ".join(set(x.dropna())),  # Concaténer les tags uniques
    "Book-Rating": "mean",  # Moyenne des notes
    "Cluster": lambda x: ", ".join(map(str, set(x.dropna())))
}).reset_index()
    return data_grouped

def train_knn_model_with_metadata(data, author_weight=0.5):
    """Trains a KNN model using metadata (tags + authors)"""
    print("\nStep 4: Initializing and training the enriched KNN model (with tags and authors)...")
    
    # Vectorization of tags and authors
    data["Final_Tags"] = data["Final_Tags"].fillna("")
    data["Book-Author"] = data["Book-Author"].fillna("")

    tfidf_authors = TfidfVectorizer(stop_words="english", max_features=100)
    
    X_authors = tfidf_authors.fit_transform(data["Book-Author"]).toarray()

    # Ajouter d'autres features numériques (ex : note moyenne)
    X_numeric = data[["Book-Rating"]].values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Convertir les clusters en one-hot encoding
    encoder = OneHotEncoder()
    X_clusters = encoder.fit_transform(data[["Cluster"]]).toarray()

    # Pondérer les auteurs par un facteur inférieur
    X_authors_weighted = X_authors * author_weight

    # Combine all features
    X_final = np.hstack((X_authors_weighted, X_numeric_scaled, X_clusters))
    print(f"Feature matrix created with shape: {X_final.shape}")

    # Train the KNN model
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
    knn_model.fit(X_final)
    print("Enriched KNN model trained successfully.")
    return knn_model, X_final
    


def find_similar_books_with_metadata(knn_model, book_title, df_grouped, X_final, n_neighbors=5):
    """Finds similar books using the enriched KNN model."""
    print("\nStep 5: Finding similar books using enriched metadata...")

    # Trouver l'index du livre dans df_grouped
    try:
        idx = df_grouped[df_grouped["Book-Title"] == book_title].index[0]
    except IndexError:
        print(f"Error: Book '{book_title}' not found in the dataset.")
        return

    # Trouver les voisins avec KNN
    distances, indices = knn_model.kneighbors([X_final[idx]], n_neighbors=n_neighbors)

    # Informations sur le livre de référence
    book_tags = df_grouped.iloc[idx]["Final_Tags"]
    book_author = df_grouped.iloc[idx]["Book-Author"]

    print(f"\n📖 **Livre de référence** : {book_title}")
    print(f"🏷️ **Tags** : {book_tags}")
    print(f"✍️ **Auteur** : {book_author}")

    # Afficher les recommandations
    print("\n🔎 **Livres recommandés** :")
    for i in indices[0][1:]:  # Exclure le livre lui-même
        rec_title = df_grouped.iloc[i]["Book-Title"]
        rec_tags = df_grouped.iloc[i]["Final_Tags"]
        rec_author = df_grouped.iloc[i]["Book-Author"]
        print(f"📘 {rec_title} | 🏷️ Tags: {rec_tags} | ✍️ Auteur: {rec_author}")


def train_svd_model(data):
    """Trains on SVD model using surprise library"""
    print("\nStep 6: Training the SVD model ... ")
    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    kf = KFold(n_splits=5)

    param_grid = {
        'n_factors': [50],
        'lr_all': [0.01],
        'reg_all': [0.2]
    }

    grid_search_SVD = GridSearchCV(SVD, param_grid, measures=['rmse'])
    grid_search_SVD.fit(dataset)

    best_params = grid_search_SVD.best_params['rmse']
    optimized_SVDmodel = SVD(
        n_factors=best_params['n_factors'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all']
    )
    for trainset, _ in kf.split(dataset):

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
    book_df_knn = preprocess_data_for_knn(book_df)

    # Create user-item matrix
    sparse_matrix, book_titles = create_user_item_matrix(book_df)

    # Train KNN model & SVD model
    # Train KNN model with metadata
    knn_model, X_final = train_knn_model_with_metadata(book_df_knn, author_weight=0.5)
    svd_model = train_svd_model(book_df)

    # Test recommendations
    find_similar_books_with_metadata(knn_model, book_titles, book_df_knn, X_final, n_neighbors=10)

    # Save artifacts
    save_artifacts(
        artifacts_path,
        knn_model=knn_model,
        svd_model=svd_model,
        book_titles=book_titles,
        X_final=X_final,
        book_df=book_df,
        book_df_knn=book_df_knn
    )

    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()
