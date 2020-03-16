#! /bin/usr/python3

import sys

import pandas as pd 
import numpy as np

from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

class Recommender(object):

    def __init__(self, ratings_file, movies_file):
        df = pd.read_csv(ratings_file)
        titles = pd.read_csv(movies_file)
        df = pd.merge(df, titles, on='movieId')
        reader = Reader(rating_scale=(1, 5))

        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        train, test = train_test_split(data, test_size=0.25)

        model = KNNBasic()

        model.fit(train)
        predictions = model.test(test)
        print(accuracy.rmse(predictions))

        '''
        ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
        ratings['number_of_ratings'] = data.groupby('title')['rating'].count()
        movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
        self.data = data
        self.ratings = ratings
        self.movie_matrix = movie_matrix
        '''
    
def usage():
    print("usage: python3 recommender.py ratings_file movies_file movie_1 movie_2")
    exit(0)

# if len(sys.argv) != 4: usage()
ratings_file = sys.argv[1]
movies_file = sys.argv[2]
# movie_1 = sys.argv[3]

r = Recommender(ratings_file, movies_file)
# r.build_model(movie_1)
# r.recommend()
