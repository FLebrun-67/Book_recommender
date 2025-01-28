# 📚 Book Recommender System

Welcome to the **Book Recommender System**! This project is a Streamlit-based application that provides personalized book recommendations using a k-Nearest Neighbors (kNN) model.

---

## 🔍 Features

### 🔢 General Statistics
- Displays the total number of books and users in the dataset.

### 🎮 Popular Books
- Shows the most popular books based on the number of ratings.

### 🌟 Top-Rated Books
- Highlights the top-rated books in the dataset.

### 🔍 Recommendations
- Provides book recommendations based on user selection.
- Allows users to rate recommendations and save their feedback.

### 🕵️‍♂️ Advanced Search
- Search for books by keywords.

### 🌐 Random Book Discovery
- Discover a random book from the dataset.

### 🎨 Visualizations
- View distributions of ratings and user interactions with books.

---

## 💪 Technologies Used

- **Streamlit**: For building the web-based user interface.
- **scikit-learn**: For implementing the kNN recommendation algorithm.
- **pandas**: For data manipulation.
- **plotly**: For creating interactive visualizations.
- **pickle**: For saving and loading the pre-trained model and datasets.

---

## 🚀 How to Run the Application

### Prerequisites

- Python 3.9 or later
- Required Python libraries (listed in `requirements.txt`):
  ```
  streamlit
  scikit-learn
  pandas
  plotly
  numpy
  ```

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/book-recommender-system.git
   cd book-recommender-system
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the required data and model files in the `artifacts/` directory:
   - `knn_model.pkl`
   - `book_titles.pkl`
   - `book_df.pkl`
   - `sparse_user_item_matrix_full_csr.pkl`

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`.

---

## 🌄 File Structure

```
book-recommender-system/
├── app.py                 # Main application script
├── artifacts/             # Contains model and dataset files
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── Dockerfile             # Docker configuration file
├── runtime.txt            # Runtime environment specification
├── train.py               # Python script for training
├── notebooks/             # Contains Jupyter notebooks
│   ├── train.ipynb        # Notebook for training the model
├── data/                  # Raw and cleaned datasets
│   ├── cleaned_data.csv   # Preprocessed dataset
```

### Description of Key Files

- **artifacts/**: Contains all the generated artifacts from the training process, including the trained model and processed data.
- **data/**: Directory for raw and cleaned datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and training.
- **Dockerfile**: Used for containerizing the application.
- **app.py**: The main application file for deploying the recommendation engine.
- **requirements.txt**: Lists all the Python dependencies required for the project.
- **runtime.txt**: Specifies the runtime environment for deployment (e.g., Python version).
- **train.py**: A standalone Python script for training the recommendation engine model.

---

## 🎨 Screenshots

### 🌐 Home Page
![Home Page](https://via.placeholder.com/600x300)

### 🔍 Recommendations
![Recommendations](https://via.placeholder.com/600x300)

### 🎨 Visualizations
![Visualizations](https://via.placeholder.com/600x300)

---

## 🔧 Future Improvements

- Add user authentication to personalize recommendations.
- Include more filters (e.g., genre, author) to refine recommendations.
- Use a larger dataset for improved accuracy.
- Implement collaborative filtering for better personalization.

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thank you for exploring the Book Recommender System! 🚀