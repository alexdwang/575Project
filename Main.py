#encoding:utf-8

import os
import multiprocessing
import heapq
import math
import numpy as np
import time
import pickle
from operator import itemgetter
from datetime import datetime
from collections import defaultdict, OrderedDict

import pearson
import euclidean
from dbhelper import DatabaseHelper


class MovieLens(object):
    def __init__(self):
        self.movies = OrderedDict()
        self.reviews = defaultdict(dict)

        db = DatabaseHelper(password='asdfghjkl')
        self.movies = db.get_all_movies()
        self.reviews = db.get_all_reviews()

    def reviews_for_movie(self, movieid):
        for review in self.reviews.values():
            if movieid in review:
                yield review[movieid]


    def average_reviews(self):
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = sum(reviews) / float(len(reviews))
            yield (movieid, average, len(reviews))


    def top_rated(self, n=10):
        # find top n recommendations and return
        return heapq.nlargest(n, self.bayesian_average(), key=itemgetter(1))


    def bayesian_average(self, c=59, m=3):
        # calculate Bayesian average and return
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = ((c * m) + sum(reviews)) / float(c + len(reviews))
            yield (movieid, average, len(reviews))

    def share_preferences(self, criticA, criticB):
        # find shared preferences between userA and userB
        if criticA not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data " % criticA)
        if criticB not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data " % criticB)
        moviesA = set(self.reviews[criticA].keys())
        moviesB = set(self.reviews[criticB].keys())
        shared = moviesA & moviesB

        reviews = {}
        for movieid in shared:
            reviews[movieid] = (
                self.reviews[criticA][movieid]['rating'],
                self.reviews[criticB][movieid]['rating'],
            )
        return reviews


    def euclidean_distance(self, criticA, criticB, prefs='users'):
        # use shared preferences to calculate euclidean distance between userA and userB (or movieA and movieB)
        if prefs == 'users':
            preferences = self.share_preferences(criticA, criticB)
        elif prefs == 'movies':
            preferences = self.shared_critics(criticA, criticB)
        else:
            raise Exception("No preferences of type '%s'." % prefs)

        # return 0 for no-preference pair
        if len(preferences) == 0: return 0

        # calculate sum of squares value
        sum_of_squares = sum([pow(a - b, 2) for a, b in preferences.values()])

        # unitilization
        return 1 / (1 + math.sqrt(sum_of_squares))


    def pearson_correlation(self, criticA, criticB, prefs='users'):
        # use shared preferences to calculate pearson correlation between userA and userB (or movieA and movieB)
        if prefs == 'users':
            preferences = self.share_preferences(criticA, criticB)
        elif prefs == 'movies':
            preferences = self.shared_critics(criticA, criticB)
        else:
            raise Exception("No preferences of type '%s'." % prefs)

        length = len(preferences)
        if length == 0: return 0

        sumA = sumB = sumSquareA = sumSquareB = sumProducts = 0
        for a, b in preferences.values():
            sumA += a
            sumB += b
            sumSquareA += pow(a, 2)
            sumSquareB += pow(b, 2)
            sumProducts += a * b

        numerator = (sumProducts * length) - (sumA * sumB)
        denominator = math.sqrt(((sumSquareA * length) - pow(sumA, 2)) * ((sumSquareB * length) - pow(sumB, 2)))
        if denominator == 0: return 0
        return abs(numerator / denominator)

    def similar_critics(self, user, metric='euclidean', n=None):
        # find similar critics for a specific user

        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation
        }

        distance = metrics.get(metric, None)

        if user not in self.reviews:
            raise KeyError("Unknown user, '%s'." % user)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unprogrammed distance metric '%s'." % metric)

        critics = {}
        for critic in self.reviews:
            if critic == user:
                continue
            critics[critic] = distance(user, critic)

        if n:
            return heapq.nlargest(n, critics.items(), key=itemgetter(1))
        return critics

    def predict_ranking(self, user, movie, metric='euclidean', critics=None):
        # predict ratings that a specific user will give to a movie using similarity as weight and perform weighted KNN algorithm
        critics = critics or self.similar_critics(user, metric=metric)
        total = 0.0
        simsum = 0.0

        for critic, similarity in critics.items():
            if movie in self.reviews[critic]:
                total += similarity * self.reviews[critic][movie]['rating']
                simsum += similarity

        if simsum == 0.0: return 0.0
        return total / simsum

    def predict_all_rankings(self, user, metric='euclidean', n=None):
        # predict ratings for all movies and return top n

        critics = self.similar_critics(user, metric=metric)
        movies = {
            movie: self.predict_ranking(user, movie, metric, critics)
            for movie in self.movies
        }

        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies


    def shared_critics(self, movieA, movieB):
        # find shared critics who watched both movies

        if movieA not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movieA)
        if movieB not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movieB)

        criticsA = set(critic for critic in self.reviews if movieA in self.reviews[critic])
        criticsB = set(critic for critic in self.reviews if movieB in self.reviews[critic])

        shared = criticsA & criticsB

        reviews = {}
        for critic in shared:
            reviews[critic] = (
                self.reviews[critic][movieA]['rating'],
                self.reviews[critic][movieB]['rating']
            )

        return reviews


    def similar_items(self, movie, metric='euclidean', n=None):
        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation,
        }

        distance = metrics.get(metric, None)
        if movie not in self.reviews:
            raise KeyError("Unknown movie, '%s'." % movie)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unprogrammed distance metric '%s'." % metric)

        items = {}
        for item in self.movies:
            if item == movie:
                continue

            items[item] = distance(item, movie, prefs='movies')

        if n:
            return heapq.nlargest(n, items.items(), key=itemgetter(1))
        return items

    def predict_items_recommendation(self, user, movie, metric='euclidean'):
        movie = self.similar_items(movie, metric=metric)
        total = 0.0
        simsum = 0.0

        for relmovie, similarity in movie.items():
            if relmovie in self.reviews[user]:
                total += similarity * self.reviews[user][relmovie]['rating']
                simsum += similarity

        if simsum == 0.0: return 0.0
        return total / simsum

def update_u2m_euclidean(model):
    result = euclidean.getRecomDict_User(model)
    db = DatabaseHelper(password='asdfghjkl')
    db.save_knn_euclidean_recommend_result(result, type='user')
    print('End training U2M euclidean')


def update_m2m_euclidean(model):
    result = euclidean.getRecomDict_Movie(model)
    db = DatabaseHelper(password='asdfghjkl')
    db.save_knn_euclidean_recommend_result(result, type='movie')
    print('End training M2M euclidean')


def update_u2m_pearson(model):
    result = pearson.getRecomDict_User(model)
    db = DatabaseHelper(password='asdfghjkl')
    db.save_knn_pearson_recommend_result(result, type='user')
    print('End training U2M pearson')


def update_m2m_pearson(model):
    result = pearson.getRecomDict_Movie(model)
    db = DatabaseHelper(password='asdfghjkl')
    db.save_knn_pearson_recommend_result(result, type='movie')
    print('End training M2M pearson')


if __name__ == '__main__':
    model = MovieLens()

    process_u2m_eu = multiprocessing.Process(target=update_u2m_euclidean, args=(model,))
    process_u2m_eu.start()
    process_m2m_eu = multiprocessing.Process(target=update_m2m_euclidean, args=(model,))
    process_m2m_eu.start()
    process_u2m_ps = multiprocessing.Process(target=update_u2m_pearson, args=(model,))
    process_u2m_ps.start()
    process_m2m_ps = multiprocessing.Process(target=update_m2m_pearson, args=(model,))
    process_m2m_ps.start()
