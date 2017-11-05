#encoding:utf-8

import os
import csv
import heapq
import math
import numpy as np
from numpy import linalg as la
import time
import pickle
from operator import itemgetter
from datetime import datetime
from collections import defaultdict


def load_reviews(path, **kwargs):
    '''''
    加载电影数据文件
    '''

    options = {
        'fieldnames': ('userid', 'movieid', 'rating', 'timestamp'),
        'delimiter': '\t'
    }

    options.update(kwargs)
    parse_date = lambda r, k: datetime.fromtimestamp(float(r[k]))
    parse_int = lambda r, k: int(r[k])

    with open(path, 'r',encoding='ISO-8859-1') as reviews:
        reader = csv.DictReader(reviews, **options)
        for row in reader:
            row['movieid'] = parse_int(row, 'movieid')
            row['userid'] = parse_int(row, 'userid')
            row['rating'] = parse_int(row, 'rating')
            row['timestamp'] = parse_date(row, 'timestamp')
            yield row

def relative_path(path):
    '''''
    辅助数据导入
    '''
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return  os.path.normpath(path)


def load_movies(path, **kwargs):
    '''''
    读取电影信息
    '''
    options = {
        'fieldnames': ('movieid', 'title', 'release', 'video', 'url'),
        'delimiter': '|',
        'restkey': 'genre'
    }
    options.update(**kwargs)

    parse_int = lambda r, k: int(r[k])
    parse_date = lambda r, k: datetime.strptime(r[k], '%d-%b-%Y') if r[k] else None

    with open(path, 'r',encoding='ISO-8859-1') as movies:
        reader = csv.DictReader(movies, **options)
        for row in reader:
            row['movieid'] = parse_int(row, 'movieid')
            # print row['movieid']
            row['release'] = parse_date(row, 'release')
            # print row['release']
            # print row['video']
            yield row


class MovieLens(object):
    def __init__(self, udata, uitem):
        self.udata = udata
        self.uitem = uitem
        self.movies = {}
        self.reviews = defaultdict(dict)
        self.load_dataset()

    def load_dataset(self):
        # 加载数据到内存中，按ID为索引
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']] = review
            # print self.reviews[review['userid']][review['movieid']]

    def reviews_for_movie(self, movieid):
        for review in self.reviews.values():
            if movieid in review:   #存在则返回
                yield review[movieid]


    def average_reviews(self):
        # 对所有的电影求平均水平

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
        preferences = self.share_preferences(criticA, criticB)

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


    def similar_items(self, movie, metric='eculidean', n=None):
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

class Recommender(object):
    @classmethod
    def load(klass, pickle_path):
        '''''
        接受磁盘上包含pickle序列化后的文件路径为参数，并用pickle模块载入文件。
        由于pickle模块在序列化是会保存导出时对象的所有属性和方法，因此反序列
        化出来的对象有可能已经和当前最新代码中的类不同。
        '''
        with open(pickle_path, 'rb') as pkl:
            return pickle.load(pkl)

    def __init__(self, udata):
        self.udata = udata
        self.users = None
        self.movies = None
        self.reviews = None

        # 描述性工程
        self.build_start = None
        self.build_finish = None
        self.description = None

        self.model = None
        self.features = 2
        self.steps = 5000
        self.alpha = 0.0002
        self.beta = 0.02

        self.load_dataset()

    def dump(self, pickle_path):
        '''''
        序列化方法、属性和数据到硬盘，以便在未来导入
        '''
        with open(pickle_path, 'wb') as pkl:
            pickle.dump(self, pkl)

    def load_dataset(self):
        '''''
        加载用户和电影的索引作为一个NxM的数组，N是用户的数量，M是电影的数量；标记这个顺序寻找矩阵的价值
        '''

        self.users = set([])
        self.movies = set([])
        for review in load_reviews(self.udata):
            self.users.add(review['userid'])
            self.movies.add(review['movieid'])

        self.users = sorted(self.users)
        self.movies = sorted(self.movies)

        self.reviews = np.zeros(shape=(len(self.users), len(self.movies)))
        for review in load_reviews(self.udata):
            uid = self.users.index(review['userid'])
            mid = self.movies.index(review['movieid'])
            self.reviews[uid, mid] = review['rating']

    def build(self, output=None):
        '''''
        训练模型
        '''
        options = {
            'K': self.features,
            'steps': self.steps,
            'alpha': self.alpha,
            'beta': self.beta
        }

        self.build_start = time.time()
        nnmf = factor2
        self.P, self.Q = nnmf(self.reviews, **options)
        self.model = np.dot(self.P, self.Q.T)
        self.build_finish = time.time()

        if output:
            self.dump(output)

    # 利用模型来访问预测的评分
    def predict_ranking(self, user, movie):
        uidx = self.users.index(user)
        midx = self.movies.index(movie)
        if self.reviews[uidx, midx] > 0:
            return None
        return self.model[uidx, midx]

    # 预测电影的排名
    def top_rated(self, user, n=12):
        movies = [(mid, self.predict_ranking(user, mid)) for mid in self.movies]
        return heapq.nlargest(n, movies, key=itemgetter(1))

if __name__ == '__main__':
    data = relative_path('u.data')
    item = relative_path('u.item')
    model = MovieLens(data, item)

    for mid, avg, num in model.top_rated(10):
        title = model.movies[mid]['title']
        print ("[%0.3f average rating (%i reviews)] %s" % (avg, num, title))

    # print(model.euclidean_distance(631, 532))  # A,B
    # print(model.pearson_correlation(232, 532))
    # for item in model.similar_critics(232, 'pearson', n=10):
    #     print("%4i: %0.3f" % item)
    # print(model.predict_ranking(422, 50, 'euclidean'))
    # print(model.predict_ranking(422, 50, 'pearson') )
    #
    # for mid, rating in model.predict_all_rankings(578, 'pearson', 10):
    #     print('%0.3f: %s' % (rating, model.movies[mid]['title']))

    # for movie, similarity in model.similar_items(631, 'pearson').items():
    #     print('%0.3f : %s' % (similarity, model.movies[movie]['title']))

    # print(model.predict_items_recommendation(232, 52, 'pearson'))

    model = Recommender(data)
    model.load_dataset()
    print ("building")
    model.build('svd')
    print(model.top_rated(1,12))