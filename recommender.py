#! /bin/usr/python3

import sys

import pandas as pd 
import numpy as np

from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader

class Recommender(object):

    # Initalize the pandas DataFrames and create the training set.
    def __init__(self, ratings_file, movies_file, movie1, rating1, movie2, rating2):
        self.ratings = pd.read_csv(ratings_file)
        self.titles = pd.read_csv(movies_file)
        self.ratings = pd.merge(self.ratings, self.titles, on="movieId")

        # Add a new user based on the ratings passed in.
        df = pd.DataFrame([[611, self.parse_title(movie1), rating1, 1584386412], [611, self.parse_title(movie2), rating2, 1584386465]],
                columns=["userId","movieId","rating","timestamp"])
        self.ratings = self.ratings.append(df, ignore_index=True, sort=False)

        # Create a Dataset and using it to build the training set.
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
        self.train = data.build_full_trainset()

        # Options for KNNWithMeans.
        self.sim_options = {
                "name": "msd",
                "user_based": False,
        }

    # Parse a movie title given at the command line.
    def parse_title(self, title):
        for m in self.titles.itertuples():
            if title.lower() in m.title.lower():
                return m.movieId

    # Fit the model and then predict what the user would rate every movie.
    def build_model(self, model_name):
        if model_name == "svd":
            model = SVD()
        elif model_name == "knn":
            model = KNNWithMeans(sim_options=self.sim_options)
        else:
            model = KNNBasic()
        model.fit(self.train)

        # Loop through all movies and determining what the user would rate each one.
        predicted_ratings = np.zeros((9742, 2))
        for i in self.titles.itertuples():
            prediction = model.predict(611, i.movieId)
            predicted_ratings[i.Index][0] = i.movieId
            predicted_ratings[i.Index][1] = prediction.est

        # Create a DataFrame for the user's ratings and displaying the top 5 movies.
        my_ratings = pd.DataFrame(predicted_ratings, columns=["movieId", "rating"])
        my_ratings = pd.merge(my_ratings, self.titles, on="movieId")
        my_ratings_display = my_ratings[["rating", "title", "genres"]].copy()
        print(my_ratings_display.sort_values(by="rating", ascending=False).head(5).to_string(index=False))

def usage():
    print("usage: python3 recommender.py ratings_file movies_file movie_1 rating_1 movie_2 rating_2 model\n\nmodels: {knn, svd}\n\n"
            + "example: python3 recommender.py ratings.csv movies.csv \"ghostbusters\" 5.0 \"toy story\" 3.0 svd")
    exit(0)

if len(sys.argv) != 8: usage()
ratings_file = sys.argv[1]
movies_file = sys.argv[2]
movie1 = sys.argv[3]
rating1 = float(sys.argv[4])
movie2 = sys.argv[5]
rating2 = float(sys.argv[6])
model = sys.argv[7]

r = Recommender(ratings_file, movies_file, movie1, rating1, movie2, rating2)
r.build_model(model)
