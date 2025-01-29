""" Scrape Open Library for book descriptions and themes using concurrent requests. """

import pickle
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from bs4 import BeautifulSoup


# Load cached ISBN descriptions
def load_cache():
    """Load the cache from disk."""
    try:
        with open("isbn_cache2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


# Save cache after each request
def save_cache(cache):
    """Save the cache to disk after each request."""
    with open("isbn_cache2.pkl", "wb") as f:
        pickle.dump(cache, f)


# Initialize cache
isbn_cache = load_cache()

# Create a persistent session
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


# Retry function with exponential backoff
def retry_with_backoff(func, max_retries=5):
    """Retry function with exponential backoff in case of network failures."""
    for attempt in range(max_retries):
        try:
            return func()  # Call the target function
        except requests.exceptions.RequestException as e:
            wait_time = 2**attempt + random.uniform(0, 2)  # Exponential backoff
            print(
                f"⚠️ Retry {attempt+1}/{max_retries} after {wait_time:.2f}s due to error: {e}"
            )
            time.sleep(wait_time)
    return None  # Failure after all retries


# Scraping function
def scrape_openlibrary_details(isbn):
    """Scrape Open Library for book descriptions and themes with retries."""
    if isbn in isbn_cache:
        return isbn, isbn_cache[isbn]

    url = f"https://openlibrary.org/isbn/{isbn}"

    def request_func():
        return session.get(url, timeout=20)

    response = retry_with_backoff(request_func)

    if response is None:
        return isbn, "Error: Failed after multiple retries", "No themes available"

    if response.status_code == 404:  # Handle 404 errors properly
        print(f"⚠️ ISBN {isbn} not found on Open Library (404). Skipping.")
        return isbn, "No description available", "No themes available"

    if response.status_code == 429:
        wait_time = random.uniform(5, 15)
        print(
            f"🚨 429 Too Many Requests for ISBN {isbn}. Retrying in {wait_time:.2f} seconds..."
        )
        time.sleep(wait_time)
        return scrape_openlibrary_details(isbn)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract book description
    description_div = soup.find("div", class_="read-more__content")
    description = (
        description_div.get_text(strip=True, separator=" ")
        if description_div
        else "No description found"
    )

    # Extract book themes
    themes_div = soup.find("div", class_="section link-box")
    themes = (
        ", ".join([a.get_text(strip=True) for a in themes_div.find_all("a")])
        if themes_div
        else "No themes available"
    )

    time.sleep(random.uniform(1, 3))  # Anti-bot delay

    isbn_cache[isbn] = (description, themes)
    save_cache(isbn_cache)
    return isbn, description, themes


# Load dataset
try:
    df_books = pd.read_csv("../data/dataset.csv")
except FileNotFoundError:
    print("❌ Error: 'dataset.csv' not found.")
    exit()

if "ISBN" not in df_books.columns:
    print("❌ Error: 'ISBN' column missing in CSV file.")
    exit()

df_books.dropna(subset=["ISBN"], inplace=True)
df_unique_isbn = df_books.drop_duplicates(subset="ISBN").copy()

# Scraping with multithreading
num_threads = min(4, len(df_unique_isbn))  # Reduce number of threads to avoid bans
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    future_to_isbn = {
        executor.submit(scrape_openlibrary_details, str(isbn).strip()): isbn
        for isbn in df_unique_isbn["ISBN"]
    }

    for future in as_completed(future_to_isbn):
        try:
            isbn, description, tags = future.result()
            df_unique_isbn.loc[df_unique_isbn["ISBN"] == isbn, "Description"] = (
                description
            )
            df_unique_isbn.loc[df_unique_isbn["ISBN"] == isbn, "Tags"] = tags
            print(f"✅ {isbn} → {description} | Tags: {tags}")
        except Exception as e:
            print(f"❌ Error processing ISBN: {future_to_isbn[future]} - {e}")

# Merge results and save

df_books = df_books.merge(
    df_unique_isbn[["ISBN", "Description", "Tags"]], on="ISBN", how="left"
)
df_books.to_csv("dataset_with_details.csv", index=False)
print("✅ Scraping completed! Results saved in 'dataset_with_details.csv'")
