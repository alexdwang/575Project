import time
import csv

import numpy
import pandas
from scipy.sparse.linalg import svds

from dbhelper import DatabaseHelper


def createDataFrame(ratings, movies):
    # use pandas frame to create ratingDataFrame
    ratingsDataFrame = pandas.DataFrame(ratings, columns=["userID", "movieID", "rating"], dtype=int)
    ratingsDataFrame = ratingsDataFrame.pivot(index="userID", columns="movieID", values="rating").fillna(0)
    moviesDataFrame = pandas.DataFrame(movies, columns=["movieID", "name", "genres"])
    moviesDataFrame['movieID'] = moviesDataFrame['movieID'].apply(pandas.to_numeric)
    return ratingsDataFrame, moviesDataFrame


def svdPrediction(ratingsDataFrame, factor=50):
    ratingsDataMatrix = ratingsDataFrame.as_matrix()
    # normalize users rating, not really understand why to do this?
    meanRating = numpy.mean(ratingsDataMatrix, axis=1)
    demeanedRatingsMatrix = ratingsDataMatrix - meanRating.reshape(meanRating.shape[0], 1)
    userFeature, sigma, itemFeature = svds(demeanedRatingsMatrix, factor)
    # do prediction and add meanRating back
    predictDataMatrix = numpy.dot(numpy.dot(userFeature, numpy.diag(sigma)), itemFeature) + meanRating.reshape(meanRating.shape[0], 1)
    predictDataFrame = pandas.DataFrame(predictDataMatrix, columns=ratingsDataFrame.columns)
    return predictDataFrame


def getUserRatedFullTable(moviesDataFrame, ratings, userID):
    ratingsDataFrame = pandas.DataFrame(ratings, columns=["userID", "movieID", "rating"], dtype=int)
    userRated = ratingsDataFrame[ratingsDataFrame.userID == userID]
    userRatedFullTable = (userRated.merge(moviesDataFrame, how='left', left_on='movieID', right_on='movieID').
                 sort_values(['rating'], ascending=False))
    return userRatedFullTable


def svdRecommender(svdPredictDataFrame, moviesDataFrame, ratings, userID, nums):
    sortedPrediction = svdPredictDataFrame.ix[userID - 1].sort_values(ascending=False)
    userRatedFullTable = getUserRatedFullTable(moviesDataFrame, ratings, userID)
    recommendations = (moviesDataFrame[~moviesDataFrame['movieID'].isin(userRatedFullTable['movieID'])].
                       merge(pandas.DataFrame(sortedPrediction).reset_index(), how='left',
                             left_on='movieID',
                             right_on='movieID').
                       rename(columns={userID - 1: 'predictRating'}).
                       sort_values('predictRating', ascending=False).
                       iloc[:nums, :-1]
                       )
    return userRatedFullTable, recommendations


def runSVDPrediction(outputFilePath, ratings, movies, users):
    ratingsDataFrame, moviesDataFrame = createDataFrame(ratings, movies)
    # use factor = 90 to do prediction
    svdPredictDataFrame = svdPrediction(ratingsDataFrame, 90)
    # create recommended movie dictionary for each user
    recommendedResult = {}
    count = 0
    for user in users:
        userId = int(user[0])
        userRatedFullTable, recommendations = svdRecommender(svdPredictDataFrame, moviesDataFrame, ratings, userId, 10)
        movieIds = recommendations.ix[:, 0]
        recommendedResult[userId] = numpy.array(movieIds).tolist()
        writer = csv.writer(open(outputFilePath, 'a'))
        writer.writerow([userId, numpy.array(movieIds)])
        count += 1
        if count % 10 == 0:
            print("count: {0}".format(count))
    return recommendedResult


if __name__ == '__main__':
    db = DatabaseHelper(password='asdfghjkl')
    ratings = db.get_all_ratings_raw()
    movies = db.get_all_movies_svd()
    users = db.get_all_users()
    print("numOfUsers: {}, numOfMovies: {}, numOfRatings: {}".format(len(users), len(movies), len(ratings)))
    print("start svd: {}".format(time.ctime()))
    result = runSVDPrediction("svdOutput.csv", ratings, movies, users)
    print("end svd: {}".format(time.ctime()))
    db.save_svd_recommend_result(result)
