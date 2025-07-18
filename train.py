""" 
Trains a book recommender system using enriched dataset.
Focuses on SVD training with enriched metadata support.
"""

import os
import pickle
import pandas as pd
from scipy.sparse import coo_matrix
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

import mlflow
import mlflow.sklearn
import time


def load_data(file_path):
    """Loads the enriched dataset from a CSV file."""
    print("Step 1: Loading enriched data...")
    
    # Essayer d'abord le dataset enrichi, sinon fallback sur l'original
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"✅ Enriched data loaded: {data.shape[0]} rows, {data.shape[1]} columns.")
        
        # Vérifier les colonnes d'enrichissement
        enriched_columns = ['subject_string_final', 'api_cover_url', 'api_first_publish_year', 'api_publisher_string']
        available_enriched = [col for col in enriched_columns if col in data.columns]
        
        if available_enriched:
            print(f"🌟 Enriched columns found: {available_enriched}")
            
            # Statistiques d'enrichissement
            if 'api_enriched' in data.columns:
                api_count = data['api_enriched'].sum()
                print(f"📊 API enriched records: {api_count}")
            
            if 'used_fallback' in data.columns:
                fallback_count = data['used_fallback'].sum()
                print(f"🔄 Fallback records: {fallback_count}")
            
            if 'subject_string_final' in data.columns:
                with_subjects = (data['subject_string_final'].notna() & (data['subject_string_final'] != '')).sum()
                print(f"🏷️ Records with subjects: {with_subjects} ({(with_subjects/len(data)*100):.1f}%)")
        else:
            print("⚠️ No enriched columns found, using original dataset")
    else:
        print(f"❌ File not found: {file_path}")
        # Fallback vers le dataset original
        fallback_path = "./data/dataset_final3.csv"
        if os.path.exists(fallback_path):
            print(f"🔄 Falling back to: {fallback_path}")
            data = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(f"Neither {file_path} nor {fallback_path} found")
    
    return data


def data_overview(data):
    """Prints an overview of the enriched data."""
    print("\nStep 2: Data overview...")
    print(f"📊 Dataset shape: {data.shape}")
    print(f"📚 Unique books: {data['Book-Title'].nunique()}")
    print(f"👥 Unique users: {data['User-ID'].nunique()}")
    print(f"⭐ Rating range: {data['Book-Rating'].min():.1f} - {data['Book-Rating'].max():.1f}")
    print(f"📈 Average rating: {data['Book-Rating'].mean():.2f}")
    
    # Informations sur les colonnes enrichies
    enriched_info = {}
    if 'subject_string_final' in data.columns:
        subjects_count = (data['subject_string_final'].notna() & (data['subject_string_final'] != '')).sum()
        enriched_info['Subjects'] = f"{subjects_count} ({subjects_count/len(data)*100:.1f}%)"
    
    if 'api_cover_url' in data.columns:
        covers_count = (data['api_cover_url'].notna() & (data['api_cover_url'] != '')).sum()
        enriched_info['Cover URLs'] = f"{covers_count} ({covers_count/len(data)*100:.1f}%)"
    
    if 'api_first_publish_year' in data.columns:
        years_count = data['api_first_publish_year'].notna().sum()
        enriched_info['Publish Years'] = f"{years_count} ({years_count/len(data)*100:.1f}%)"
    
    if 'api_publisher_string' in data.columns:
        publishers_count = (data['api_publisher_string'].notna() & (data['api_publisher_string'] != '')).sum()
        enriched_info['Publishers'] = f"{publishers_count} ({publishers_count/len(data)*100:.1f}%)"
    
    if enriched_info:
        print("\n🌟 Enriched metadata coverage:")
        for key, value in enriched_info.items():
            print(f"   {key}: {value}")


def create_user_item_matrix(data):
    """Creates a sparse user-item matrix."""
    print("\nStep 3: Creating user-item matrix...")
    users = data["User-ID"].astype("category").cat.codes
    books = data["Book-Title"].astype("category").cat.codes
    ratings = data["Book-Rating"]
    sparse_matrix = coo_matrix((ratings, (users, books)))
    print(f"User-item matrix created with dimensions: {sparse_matrix.shape}.")
    return sparse_matrix, data["Book-Title"].astype("category").cat.categories


def train_svd_model(data):
    """Trains SVD model using surprise library with MLflow tracking."""
    print("\nStep 4: Training the SVD model with MLflow tracking...")
    
    # Préparer les données pour Surprise (TON CODE EXISTANT)
    reader = Reader(rating_scale=(data['Book-Rating'].min(), data['Book-Rating'].max()))
    dataset = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    
    # Split train/test (TON CODE EXISTANT)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # ✨ NOUVEAU : Configuration MLflow
    mlflow.set_experiment("Book_Recommender_SVD")
    
    with mlflow.start_run(run_name="SVD_Training"):
        print("📊 MLflow tracking started...")
        
        # Paramètres du modèle
        model_params = {
            "n_factors": 35,
            "lr_all": 0.005,
            "reg_all": 0.1,
            "n_epochs": 15,
            "random_state": 42
        }
        
        mlflow.log_params(model_params)
        
        dataset_info = {
            "total_ratings": len(data),
            "unique_users": data['User-ID'].nunique(),
            "unique_books": data['Book-Title'].nunique(),
            "rating_min": float(data['Book-Rating'].min()),
            "rating_max": float(data['Book-Rating'].max()),
            "test_size": 0.2
        }
        mlflow.log_params(dataset_info)
        
        svd = SVD(
            n_factors=model_params["n_factors"],
            lr_all=model_params["lr_all"],
            reg_all=model_params["reg_all"],
            n_epochs=model_params["n_epochs"],
            random_state=model_params["random_state"]
        )
        
        print("🏃‍♂️ Training SVD model...")
        
        start_time = time.time()
        svd.fit(trainset)
        training_time = time.time() - start_time
        
        print("📊 Evaluating model performance...")
        
        # Évaluer sur l'ensemble d'entraînement
        train_predictions = svd.test(trainset.build_testset())
        train_rmse = accuracy.rmse(train_predictions, verbose=False)
        train_mae = accuracy.mae(train_predictions, verbose=False)
        
        # Évaluer sur l'ensemble de test
        test_predictions = svd.test(testset)
        test_rmse = accuracy.rmse(test_predictions, verbose=False)
        test_mae = accuracy.mae(test_predictions, verbose=False)
        
        metrics = {
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "training_time_seconds": float(training_time),
            "overfitting_diff": float(test_rmse - train_rmse)
        }
        mlflow.log_metrics(metrics)
        
        print("\n📈 Model Performance:")
        print(f"   Training   - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"   Test       - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")
        
        rmse_diff = test_rmse - train_rmse
        if rmse_diff > 0.5:
            print(f"⚠️  Possible overfitting detected (RMSE diff: {rmse_diff:.4f})")
        else:
            print(f"✅ Good generalization (RMSE diff: {rmse_diff:.4f})")
        
        #Afficher l'URL MLflow
        print(f"\n📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print("💡 Run 'mlflow ui' in terminal to view results")
        
        return svd


def save_artifacts(artifacts_path, **artifacts):
    """Saves artifacts to the specified directory with enriched naming."""
    print(f"\nStep 5: Saving artifacts to {artifacts_path}...")
    os.makedirs(artifacts_path, exist_ok=True)
    
    saved_files = []
    for name, artifact in artifacts.items():
        file_path = os.path.join(artifacts_path, f"{name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
            saved_files.append(f"{name}.pkl")
            print(f"✅ {name} saved to '{file_path}'")
    
    print(f"\n📁 Total files saved: {len(saved_files)}")
    return saved_files


def validate_dataset_for_recommendations(data):
    """Validates that the dataset is suitable for recommendations."""
    print("\nStep 6: Validating dataset for recommendations...")
    
    # Vérifications de base
    min_users = 10
    min_books = 10
    min_ratings_per_user = 2
    min_ratings_per_book = 2
    
    unique_users = data['User-ID'].nunique()
    unique_books = data['Book-Title'].nunique()
    
    print("📊 Basic validation:")
    print(f"   Users: {unique_users} (min required: {min_users})")
    print(f"   Books: {unique_books} (min required: {min_books})")
    
    # Vérifier la distribution des ratings
    user_rating_counts = data.groupby('User-ID').size()
    book_rating_counts = data.groupby('Book-Title').size()
    
    users_with_enough_ratings = (user_rating_counts >= min_ratings_per_user).sum()
    books_with_enough_ratings = (book_rating_counts >= min_ratings_per_book).sum()
    
    print(f"   Users with ≥{min_ratings_per_user} ratings: {users_with_enough_ratings}")
    print(f"   Books with ≥{min_ratings_per_book} ratings: {books_with_enough_ratings}")
    
    # Vérifier la sparsité
    total_possible_ratings = unique_users * unique_books
    actual_ratings = len(data)
    sparsity = (1 - actual_ratings / total_possible_ratings) * 100
    
    print(f"   Matrix sparsity: {sparsity:.2f}%")
    
    # Validation des colonnes enrichies
    if 'subject_string_final' in data.columns:
        books_with_subjects = (data['subject_string_final'].notna() & 
                              (data['subject_string_final'] != '')).sum()
        print(f"   Books with subjects: {books_with_subjects} ({books_with_subjects/len(data)*100:.1f}%)")
    
    warnings = []
    if unique_users < min_users:
        warnings.append(f"Too few users ({unique_users} < {min_users})")
    if unique_books < min_books:
        warnings.append(f"Too few books ({unique_books} < {min_books})")
    if sparsity > 99.5:
        warnings.append(f"Very sparse matrix ({sparsity:.2f}%)")
    
    if warnings:
        print(f"⚠️  Warnings: {', '.join(warnings)}")
    else:
        print("✅ Dataset validation passed")
    
    return len(warnings) == 0


def main():
    """Main function to train the book recommender system with enriched data."""
    print("🚀 Starting enhanced book recommender training...")
    
    # File paths - priorité au dataset enrichi
    enriched_data_path = "./data/Ratings_enriched.csv"
    fallback_data_path = "./data/Ratings_enriched.csv"
    artifacts_path = "artifacts/"
    
    # Déterminer quel dataset utiliser
    if os.path.exists(enriched_data_path):
        data_file_path = enriched_data_path
        is_enriched = True
        print(f"📁 Using enriched dataset: {data_file_path}")
    else:
        data_file_path = fallback_data_path
        is_enriched = False
        print(f"📁 Using original dataset: {data_file_path}")
        print("💡 Tip: Run dataset_enricher.py to create enriched dataset")
    
    # Load and inspect data
    book_df = load_data(data_file_path)
    data_overview(book_df)
    
    # Validate dataset
    is_valid = validate_dataset_for_recommendations(book_df)
    if not is_valid:
        print("⚠️  Dataset validation warnings detected, but continuing...")
    
    # Create user-item matrix
    _, book_titles = create_user_item_matrix(book_df)
    
    # Train SVD model
    svd_model = train_svd_model(book_df)
    
    # Prepare artifacts to save
    artifacts_to_save = {
        'svd_model': svd_model,
        'book_titles': book_titles,
        'book_df': book_df,  # Le dataset utilisé (enrichi ou non)
    }
    
    # Ajouter des métadonnées sur l'enrichissement
    training_metadata = {
        'is_enriched': is_enriched,
        'dataset_path': data_file_path,
        'training_date': pd.Timestamp.now().isoformat(),
        'enriched_columns': [col for col in ['subject_string_final', 'api_cover_url', 
                                           'api_first_publish_year', 'api_publisher_string'] 
                           if col in book_df.columns]
    }
    artifacts_to_save['training_metadata'] = training_metadata
    
    # Save artifacts
    saved_files = save_artifacts(artifacts_path, **artifacts_to_save)
    
    # Final summary
    print("\n🎉 Training completed successfully!")
    print(f"📊 Model trained on {len(book_df):,} ratings")
    print(f"📚 Covering {book_df['Book-Title'].nunique():,} unique books")
    print(f"👥 From {book_df['User-ID'].nunique():,} unique users")
    
    if is_enriched:
        print(f"🌟 Enhanced with {len(training_metadata['enriched_columns'])} enriched features")
    
    print(f"💾 Artifacts saved: {', '.join(saved_files)}")
    print("\n🚀 Ready to run: streamlit run app.py")


if __name__ == "__main__":
    main()