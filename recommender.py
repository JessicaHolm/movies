#! /bin/usr/python3

import sys
import warnings

import pandas as pd 
import numpy as np

from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

warnings.filterwarnings('ignore')

class Recommender(object):

    # Initalize the pandas DataFrames and create the training set.
    def __init__(self, ratings_file, movies_file, movie1, rating1, movie2, rating2):
        ratings = pd.read_csv(ratings_file)
        self.titles = pd.read_csv(movies_file)
        ratings = pd.merge(ratings, self.titles, on='movieId')
        df = pd.DataFrame([[611, self.parse_title(movie1), rating1, 1584386412], [611, self.parse_title(movie2), rating2, 1584386465]], columns=['userId','movieId','rating','timestamp'])
        ratings = ratings.append(df, ignore_index=True)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        self.train = data.build_full_trainset()

    # Parse a movie title given at the command line.
    def parse_title(self, title):
        for m in self.titles.itertuples():
            if title.lower() in m.title.lower():
                return m.movieId

    # Fit the model and then predict what the user would rate every movie.
    def build_model(self):
        model = SVD()
        model.fit(self.train)

        predicted_ratings = np.zeros((9742, 2))
        for i in self.titles.itertuples():
            prediction = model.predict(611, i.movieId)
            predicted_ratings[i.Index][0] = i.movieId
            predicted_ratings[i.Index][1] = prediction.est

        my_ratings = pd.DataFrame(predicted_ratings, columns=['movieId', 'rating'])
        my_ratings = pd.merge(my_ratings, self.titles, on='movieId')
        print(my_ratings.sort_values(by='rating', ascending=False).head(5))

def usage():
    print("usage: python3 recommender.py ratings_file movies_file movie_1 rating_1 movie_2 rating_2" + "\n"
            + "example: python3 recommender.py ratings.csv movies.csv \"ghostbusters\" 5.0 \"toy story\" 3.0")
    exit(0)

if len(sys.argv) != 7: usage()
ratings_file = sys.argv[1]
movies_file = sys.argv[2]
movie1 = sys.argv[3]
rating1 = float(sys.argv[4])
movie2 = sys.argv[5]
rating2 = float(sys.argv[6])

r = Recommender(ratings_file, movies_file, movie1, rating1, movie2, rating2)
r.build_model()
