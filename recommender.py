#! /bin/usr/python3

import warnings
import sys

import pandas as pd 
import numpy as np

warnings.filterwarnings('ignore')

class Recommender(object):

    def __init__(self, ratings_file, movies_file):
        data = pd.read_csv(ratings_file)
        titles = pd.read_csv(movies_file)
        data = pd.merge(data, titles, on='movieId')
        ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
        ratings['number_of_ratings'] = data.groupby('title')['rating'].count()
        movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
        self.data = data
        self.ratings = ratings
        self.movie_matrix = movie_matrix

    def build_model(self, movie_1):
        movie_1_user_rating = self.movie_matrix[movie_1]
        # movie_2_user_rating = self.movie_matrix[movie_2]
        similar_to_movie_1 = self.movie_matrix.corrwith(movie_1_user_rating)
        # similar_to_movie_2 = self.movie_matrix.corrwith(movie_2_user_rating)
        corr_movie_1 = pd.DataFrame(similar_to_movie_1, columns=['correlation'])
        corr_movie_1.dropna(inplace=True)
        # corr_movie_2 = pd.DataFrame(similar_to_movie_2, columns=['correlation'])
        # corr_movie_2.dropna(inplace=True)
        self.corr_movie_1 = corr_movie_1.join(self.ratings['number_of_ratings'])
        # self.corr_movie_2 = corr_movie_2.join(ratings['number_of_ratings'])

    def recommend(self):
        print(self.corr_movie_1[self.corr_movie_1['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10))


def usage():
    print("usage: python3 recommender.py ratings_file movies_file movie_1 movie_2")
    exit(0)

if len(sys.argv) != 4: usage()
ratings_file = sys.argv[1]
movies_file = sys.argv[2]
movie_1 = sys.argv[3]

r = Recommender(ratings_file, movies_file)
r.build_model(movie_1)
r.recommend()
