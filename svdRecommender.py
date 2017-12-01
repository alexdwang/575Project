import numpy
import time
import pandas
from scipy.sparse.linalg import svds
import csv
from sklearn.metrics import mean_squared_error
from math import sqrt


def loadFiles(filePath):
    lines = open(filePath, 'r').readlines()
    records = []
    for line in lines:
        records.append(line.split("::")[:3])
    return records


def createDataFrame(ratings, movies):
    # use pandas frame to create ratingDataFrame
    ratingsDataFrame = pandas.DataFrame(ratings, columns=["userID", "movieID", "rating"], dtype=int)
    ratingsDataFrame = ratingsDataFrame.pivot(index="userID", columns="movieID", values="rating").fillna(0)
    moviesDataFrame = pandas.DataFrame(movies, columns=["movieID", "name", "genres"])
    moviesDataFrame['movieID'] = moviesDataFrame['movieID'].apply(pandas.to_numeric)
    return ratingsDataFrame, moviesDataFrame


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# not used at all
def naiveSVDPrediction(ratingsDataFrame, factor=50):
    ratingsDataMatrix = ratingsDataFrame.as_matrix()
    # normalize users rating, not really understand why to do this?
    meanRating = numpy.mean(ratingsDataMatrix, axis=1)
    demeanedRatingsMatrix = ratingsDataMatrix - meanRating.reshape(meanRating.shape[0], 1)
    userFeature, sigma, itemFeature = svds(demeanedRatingsMatrix, factor)
    # do prediction and add meanRating back
    predictDataMatrix = numpy.dot(numpy.dot(userFeature, numpy.diag(sigma)), itemFeature) + meanRating.reshape(meanRating.shape[0], 1)
    predictDataFrame = pandas.DataFrame(predictDataMatrix, columns=ratingsDataFrame.columns)
    return predictDataFrame


def findNotShownIDs(movies):
    moviesDataFrame = pandas.DataFrame(movies, columns=["movieID", "name", "genres"])
    moviesDataFrame['movieID'] = moviesDataFrame['movieID'].apply(pandas.to_numeric)
    movieIDs = moviesDataFrame['movieID'].values
    notShownIDs = []
    for x in range(3952):
        if x + 1 not in movieIDs:
            notShownIDs.append(x + 1)
    return notShownIDs


def getUserRatedFullTable(moviesDataFrame, ratings, userID):
    ratingsDataFrame = pandas.DataFrame(ratings, columns=["userID", "movieID", "rating"], dtype=int)
    userRated = ratingsDataFrame[ratingsDataFrame.userID == userID]
    userRatedFullTable = (userRated.merge(moviesDataFrame, how='left', left_on='movieID', right_on='movieID').
                 sort_values(['rating'], ascending=False))
    return userRatedFullTable


def getRecommendedTable(moviesDataFrame, recommendations):
    rec = []
    for movieID in recommendations:
        movie = moviesDataFrame[moviesDataFrame.movieID == movieID].values
        rec.append(movie)
    return rec


def svdRecommender(svdPredictedMatrix, moviesDataFrame, ratings, userID, nums, notShownIDs):
    curUserPrediction = numpy.array(svdPredictedMatrix[userID - 1])
    curUserSorted = curUserPrediction.argsort()[::-1]
    userRatedFullTable = getUserRatedFullTable(moviesDataFrame, ratings, userID)
    userRatedIDs = userRatedFullTable['movieID'].values
    recommendations = []
    for x in range(len(curUserSorted)):
        if x not in notShownIDs and x not in userRatedIDs:
            recommendations.append(curUserSorted[x])
            if len(recommendations) == nums:
                break
    return userRatedFullTable, recommendations


def initializeSVDParameters(numOfUsers, numOfMovies):
    # initialize parameters
    userBias = [0.0 for x in range(numOfUsers)]
    movieBias = [0.0 for x in range(numOfMovies)]
    pu, qi = [], []
    factor = 100
    for user in range(numOfUsers):
        p = numpy.random.normal(0.0, 0.1, factor)
        pu.append(p)
    for movie in range(numOfMovies):
        p = numpy.random.normal(0.0, 0.1, factor)
        qi.append(p)
    loops = 20
    learnRate = 0.005
    regularize = 0.02
    return userBias, movieBias, pu, qi, factor, loops, learnRate, regularize


def updateSVDParameter(mean, trainDataFrame, predMatrix, userBias, movieBias, pu, qi, learnRate=0.005, regularize=0.02):
    trainRating = trainDataFrame.as_matrix()
    for rating in trainRating:
        userIndex = rating[0] - 1
        movieIndex = rating[1] - 1
        rate = rating[2]
        bias = mean + userBias[userIndex] + movieBias[movieIndex]
        pred = numpy.dot(pu[userIndex], qi[movieIndex]) + bias
        predMatrix[userIndex][movieIndex] = pred
        # update parameters
        errorUI = rate - pred
        userBias[userIndex] = userBias[userIndex] + learnRate * (errorUI - regularize * userBias[userIndex])
        movieBias[movieIndex] = movieBias[movieIndex] + learnRate * (errorUI - regularize * movieBias[movieIndex])
        pu[userIndex] = pu[userIndex] + learnRate * (errorUI * qi[movieIndex] - regularize * pu[userIndex])
        qi[movieIndex] = qi[movieIndex] + learnRate * (errorUI * pu[userIndex] - regularize * qi[movieIndex])
    return userBias, movieBias, pu, qi


def predict(predMatrix, numOfUsers, numOfMovies, userBias, movieBias, pu, qi, mean):
    # predict with updated parameter
    for i in range(numOfUsers):
        for j in range(numOfMovies):
            bias = mean + userBias[i] + movieBias[j]
            pred = numpy.dot(pu[i], qi[j]) + bias
            predMatrix[i][j] = pred
    return predMatrix


def trainSVD(numOfUsers, numOfMovies, ratings):
    # initialize parameters
    ratingsDataFrame = pandas.DataFrame(ratings, columns=["userID", "movieID", "rating"], dtype=int)
    mean = numpy.mean(ratingsDataFrame.as_matrix(), axis=0)[2]
    userBias, movieBias, pu, qi, factor, loops, learnRate, regularize = initializeSVDParameters(numOfUsers, numOfMovies)
    predMatrix = numpy.zeros((numOfUsers, numOfMovies))
    # train 20 times again
    for x in range(loops):
        print("x: {0} time: {1}".format(x, time.ctime()))
        userBias, movieBias, pu, qi = updateSVDParameter(mean, ratingsDataFrame, predMatrix, userBias, movieBias, pu, qi)
    predMatrix = predict(predMatrix, numOfUsers, numOfMovies, userBias, movieBias, pu, qi, mean)
    return predMatrix


def runSVDPrediction(outputFilePath, ratings, movies, users, notShownIDs):
    pivotedRatingsDataFrame, moviesDataFrame = createDataFrame(ratings, movies)
    # get svdPredictedMatrix here
    numOfUsers = len(users)
    numOfMovies = 3952
    svdPredictedMatrix = trainSVD(numOfUsers, numOfMovies, ratings)
    # create recommended movie dictionary for each user
    recommendedResult = {}
    count = 0
    for user in users:
        userId = int(user[0])
        userRatedFullTable, recommendations = svdRecommender(svdPredictedMatrix, moviesDataFrame, ratings, userId, 10, notShownIDs)
        recommendedResult[userId] = numpy.array(recommendations)
        writer = csv.writer(open(outputFilePath, 'a'))
        writer.writerow([userId, numpy.array(recommendations)])
        count += 1
        if count % 10 == 0:
            print("count: {}".format(count))
    return recommendedResult


if __name__ == '__main__':
    ratings = loadFiles("data/ratings.dat")
    movies = loadFiles("data/movies.dat")
    users = loadFiles("data/users.dat")
    notShownIDs = findNotShownIDs(movies)
    print(notShownIDs)
    # fill up movies table
    print("numOfUsers: {}, numOfMovies: {}, numOfRatings: {}".format(len(users), len(movies), len(ratings)))
    print("start svd: {}".format(time.ctime()))
    runSVDPrediction("svdOutput.csv", ratings, movies, users, notShownIDs)
    print("end svd: {}".format(time.ctime()))
