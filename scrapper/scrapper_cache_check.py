""" Test script to check if the cache file exists and display a preview of its content. """

import pickle

try:
    with open("isbn_cache2.pkl", "rb") as f:
        cache = pickle.load(f)
        print(f"{len(cache)} ISBNs already scrapped.")

        PREVIEW_COUNT = 5
        print("\n📚 Preview of cached data:")
        for i, (isbn, data) in enumerate(cache.items()):
            description, themes = (
                data if isinstance(data, tuple) else (data, "No themes available")
            )
            print(
                f"{i+1}. ISBN: {isbn}\n   Description: {description[:100]}...\n   Themes: {themes}"
            )
            if i + 1 >= PREVIEW_COUNT:
                break

except FileNotFoundError:
    print("No cache file found.")
