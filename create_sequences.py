import pandas as pd
pd.set_option('display.max_colwidth', None)

import re
import string

import numpy as np

MOVIES_DATA_PATH = './rotten_tomatoes_movies.csv'
REVIEWS_DATA_PATH = './rotten_tomatoes_movie_reviews.csv'

def create_sequences():
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    reviews_df = pd.read_csv(REVIEWS_DATA_PATH)

    # Merge dataframes
    df = pd.merge(movies_df, reviews_df, on='id')

    # Remove unnecessary columns
    df = df[["title", "reviewText"]]

    # Drop null values
    df.dropna(inplace=True)

    # Select all rows except the last 394,027.
    df = df.iloc[:-394027]

    # Reformat data
    json_data = df.to_json(orient='records')
    parsed_data = pd.read_json(json_data)
    data = [
        "movie review:" + row['title'] + " | " + row['reviewText']
        for index, row in parsed_data.iterrows()
    ]

    def format_string(s):
        s = s.replace('\n', ' ')
        # Handle punctuation spacing
        s = re.sub(f"([{re.escape(string.punctuation)}])", r' \1 ', s)
        # Collapse multiple spaces
        s = re.sub(' +', ' ', s)
        return s.strip()

    data = [format_string(d) for d in data]

    np.save('./data/movie_reviews.npy', data)


if __name__ == '__main__':
    create_sequences()

