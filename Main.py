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
		self.num_of_movie = 3952

	# use shared preferences to calculate euclidean distance between userA and userB (or movieA and movieB)
	def euclidean_distance(self, criticA, criticB, prefs='users'):
		if prefs == 'users':
			preferences = self.shared_movies(criticA, criticB)
		elif prefs == 'movies':
			preferences = self.shared_users(criticA, criticB)
		else:
			raise Exception("No preferences of type '%s'." % prefs)

		# return 0 for no-preference pair
		if len(preferences) == 0: return 0

		# calculate sum of squares value
		sum_of_squares = sum([pow(a - b, 2) for a, b in preferences.values()])

		# unitilization
		return 1 / (1 + math.sqrt(sum_of_squares))


	# use shared preferences to calculate pearson correlation between userA and userB (or movieA and movieB)
	def pearson_correlation(self, criticA, criticB, prefs='users'):
		if prefs == 'users':
			preferences = self.shared_movies(criticA, criticB)
		elif prefs == 'movies':
			preferences = self.shared_users(criticA, criticB)
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

	# find shared preferences between userA and userB
	def shared_movies(self, criticA, criticB):
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


	# find shared critics who watched both movieA and movieB
	def shared_users(self, movieA, movieB):

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

	# find similar critics for a specific user
	def similar_users(self, user, metric='euclidean', n=None):
		metrics = {
			'euclidean': self.euclidean_distance,
			'pearson': self.pearson_correlation
		}

		distance = metrics.get(metric, None)

		if user not in self.reviews:
			raise KeyError("Unknown user, '%s'." % user)
		if not distance or not callable(distance):
			raise KeyError("Unknown metric '%s'." % metric)

		critics = {}
		for critic in self.reviews:
			if critic == user:
				continue
			critics[critic] = distance(user, critic)

		if n:
			return heapq.nlargest(n, critics.items(), key=itemgetter(1))
		return critics

	# predict ratings that a specific user will give to a movie using similarity as weight and perform weighted KNN algorithm
	def predict_ranking(self, user, movie, metric='euclidean', critics=None):
		critics = critics or self.similar_users(user, metric=metric)
		total = 0.0
		simsum = 0.0

		for critic, similarity in critics.items():
			if movie in self.reviews[critic]:
				total += similarity * self.reviews[critic][movie]['rating']
				simsum += similarity

		if simsum == 0.0: return 0.0
		return total / simsum

	# predict ratings for all movies and return top n
	def predict_all_rankings(self, user, metric='euclidean', n=None):
		critics = self.similar_users(user, metric=metric)
		movies = {
			movie: self.predict_ranking(user, movie, metric, critics)
			for movie in self.movies
		}

		if n:
			return heapq.nlargest(n, movies.items(), key=itemgetter(1))
		return movies

def load_test_matrix(path):
	testMatrix = np.zeros((7000, model.num_of_movie))
	lines = open(path, 'r', encoding='ISO-8859-1').readlines()
	for line in lines:
		mydata = line.split(sep="::")
		testMatrix[int(mydata[0]) - 1][int(mydata[1]) - 1] = int(mydata[2])
	return testMatrix


def getEMatrix(model, metric='euclidean'):
	EMatrix = np.zeros((7000, model.num_of_movie))
	count = 0
	for userid in model.reviews.keys():
		movies = model.predict_all_rankings(userid, metric=metric)
		for mid in movies.keys():
			EMatrix[int(userid) - 1][mid - 1] = movies[mid]
		count += 1
		if count%100 == 1:
			print(count)
		# print(userid)
	return EMatrix

def saveMatrix(path, matrix):
	file = open(path,'a')
	writer = csv.writer(file)
	writer.writerows(matrix)
	file.close()

def rmse(prediction, ground_truth):
	prediction = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))

def rootMeanSquareError(originalMatrix, predictMatrix):
	meanError = mean_squared_error(originalMatrix, predictMatrix)
	# print("meanError: {}".format(meanError))
	squaredError = sqrt(meanError)
	return squaredError

# calculate rmse for euclidean and pearson
def calculate_rmse_euclidean(model):
	testMatrix = load_test_matrix("data/test.dat")
	userPrediction_e = getEMatrix(model, 'euclidean')

	print('euclidean User-based CF RMSE: ' + str(rmse(userPrediction_e, testMatrix)))

def calculate_rmse_pearson(model):
	testMatrix = load_test_matrix("data/test.dat")
	userPrediction_p = getEMatrix(model, 'pearson')
	
	print('pearson User-based CF RMSE: ' + str(rmse(userPrediction_p, testMatrix)))

def calculate_root_square_error(model):
	testMatrix = load_test_matrix("data/ratings.dat")
	userPrediction_p = getEMatrix(model, 'pearson')
	#saveMatrix(pearson_path, userPrediction_p)
	
	userPrediction_e = getEMatrix(model, 'euclidean')
	#saveMatrix(euclidean_path, userPrediction_e)
	
	print('pearson User-based CF RMSE: ' + str(rootMeanSquareError(userPrediction_p, testMatrix)))
	print('euclidean User-based CF RMSE: ' + str(rootMeanSquareError(userPrediction_e, testMatrix)))
	
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

	calc_rmse_eu = multiprocessing.Process(target=calculate_rmse_euclidean, args=(model,))
	calc_rmse_eu.start()

	calc_rmse_pr = multiprocessing.Process(target=calculate_pearson, args=(model,))
	calc_rmse_pr.start()

	calc_rse = multiprocessing.Process(target=calculate_root_square_error, args=(model,))
	calc_rse.start()

