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
            yield (movieid, average, len(reviews))  # 返回了（movieid，评分平均分，长度(即评价人数)）


    def top_rated(self, n=10):
        # 返回一个前n的top排行
        return heapq.nlargest(n, self.bayesian_average(), key=itemgetter(1))


    def bayesian_average(self, c=59, m=3):
        # 返回一个修正后的贝叶斯平均值
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = ((c * m) + sum(reviews)) / float(c + len(reviews))
            yield (movieid, average, len(reviews))

    def share_preferences(self, criticA, criticB):
        '''''
        找出两个评论者之间的交集
        '''
        if criticA not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data " % criticA)
        if criticB not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data " % criticB)
        moviesA = set(self.reviews[criticA].keys())
        moviesB = set(self.reviews[criticB].keys())
        shared = moviesA & moviesB

        # 创建一个评论过的的字典返回
        reviews = {}
        for movieid in shared:
            reviews[movieid] = (
                self.reviews[criticA][movieid]['rating'],
                self.reviews[criticB][movieid]['rating'],
            )
        return reviews


    def euclidean_distance(self, criticA, criticB, prefs='users'):
        '''''
        通过两个人的共同偏好作为向量来计算两个用户之间的欧式距离
        '''
        # 创建两个用户的交集
        if prefs == 'users':
            preferences = self.share_preferences(criticA, criticB)
        elif prefs == 'movies':
            preferences = self.shared_critics(criticA, criticB)
        else:
            raise Exception("No preferences of type '%s'." % prefs)

        # 没有则返回0
        if len(preferences) == 0: return 0

        # 求偏差的平方的和
        sum_of_squares = sum([pow(a - b, 2) for a, b in preferences.values()])

        # 修正的欧式距离，返回值的范围为[0,1]
        return 1 / (1 + math.sqrt(sum_of_squares))


    def pearson_correlation(self, criticA, criticB, prefs='users'):
        '''''
        返回两个评论者之间的皮尔逊相关系数
        '''
        if prefs == 'users':
            preferences = self.share_preferences(criticA, criticB)
        elif prefs == 'movies':
            preferences = self.shared_critics(criticA, criticB)
        else:
            raise Exception("No preferences of type '%s'." % prefs)

        length = len(preferences)
        if length == 0: return 0

        # 循环处理每一个评论者之间的皮尔逊相关系数
        sumA = sumB = sumSquareA = sumSquareB = sumProducts = 0
        for a, b in preferences.values():
            sumA += a
            sumB += b
            sumSquareA += pow(a, 2)
            sumSquareB += pow(b, 2)
            sumProducts += a * b

            # 计算皮尔逊系数
        numerator = (sumProducts * length) - (sumA * sumB)
        denominator = math.sqrt(((sumSquareA * length) - pow(sumA, 2)) * ((sumSquareB * length) - pow(sumB, 2)))
        if denominator == 0: return 0
        return abs(numerator / denominator)

    def similar_critics(self, user, metric='euclidean', n=None):
        '''''
        为特定用户寻找一个合适的影评人
        '''

        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation
        }

        distance = metrics.get(metric, None)

        # 解决可能出现的状况
        if user not in self.reviews:
            raise KeyError("Unknown user, '%s'." % user)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unprogrammed distance metric '%s'." % metric)

            # 计算对用户最合适的影评人
        critics = {}
        for critic in self.reviews:
            # 不能与自己进行比较
            if critic == user:
                continue
            critics[critic] = distance(user, critic)

        if n:
            return heapq.nlargest(n, critics.items(), key=itemgetter(1))
        return critics

    def predict_ranking(self, user, movie, metric='euclidean', critics=None):
        '''''
        预测一个用户对一部电影的评分，相当于评论过这部电影的用户对当前用户的加权均值
        并且权重取决与其他用户和该用户的相似程度
        '''
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
        '''''
        为所有的电影预测评分，返回前n个评分的电影和它们的评分
        '''

        critics = self.similar_critics(user, metric=metric)
        movies = {
            movie: self.predict_ranking(user, movie, metric, critics)
            for movie in self.movies
        }

        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies


    def shared_critics(self, movieA, movieB):
        '''''
        返回两部电影的交集,即两部电影在同一个人观看过的情况
        '''

        if movieA not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movieA)
        if movieB not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movieB)

        criticsA = set(critic for critic in self.reviews if movieA in self.reviews[critic])
        criticsB = set(critic for critic in self.reviews if movieB in self.reviews[critic])

        shared = criticsA & criticsB  # 和操作

        # 创建一个评论过的字典以返回
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
        # 解决可能出现的状况
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

def initialize(R, K):
    N, M = R.shape
    P = np.mat([ [ 0 for i in range(K) ] for j in range(N) ])
    Q = np.mat([ [ 0 for i in range(K) ] for j in range(M) ])
    return P, Q

def factor2(R, P=None, Q=None, K=2, steps=5000, alpha=0.0002, beta=0.02):
    """
           依靠给定的参数训练矩阵R.

        :param R:  N x M的矩阵，即将要被训练的
        :param P: 一个初始的N x K矩阵
        :param Q: 一个初始的M x K矩阵
        :param K: 潜在的特征
        :param steps: 最大迭代次数
        :param alpha: 梯度下降法的下降率
        :param beta:  惩罚参数

        :returns:  P 和 Q
           """
    if not P or not Q:
        P, Q = initialize(R, K)
    Q = Q.T

    rows, cols = R.shape
    for step in range(steps):

        eR = np.dot(P, Q)  # 一次性内积即可

        for i in range(rows):
            for j in range(cols):
                if R[i, j] > 0:
                    eij = R[i, j] - eR[i, j]
                    for k in range(K):
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])
                        Q[k, j] = Q[k, j] + alpha * (2 * eij * P[i, k] - beta * Q[k, j])

        eR = np.dot(P, Q)  # Compute dot product only once
        e = 0

        for i in range(rows):
            for j in range(cols):
                if R[i, j] > 0:
                    e = e + pow((R[i, j] - eR[i, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i, k], 2) + pow(Q[k, j], 2))
        if e < 0.001:
            break
    return P, Q.T


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
