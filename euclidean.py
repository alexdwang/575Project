'''
Movie recommendation to users and find similar movies using euclidean
'''

#  A method to get the 10 most recommended movies for a user
#  return a dictionary whose key set is userid
#  content corresponds to each key is movieids of top 10 recommendations using euclidean

def getRecomDict_User(model):
    recommendtouser_dict = {}
    for userid in model.reviews.keys():
        mymovies = []
        for mid in model.predict_all_rankings(userid, n = 20):
            mymovies.append(model.movies[mid[0]]['movieid'])
        #     print(mid)
        # print("userid = ", userid)
        # print(mymovies)
        recommendtouser_dict[userid] = mymovies
    return recommendtouser_dict

#  A method to get the 10 most similar movies for given movies
#  return a dictionary whose key set is userid
#  content corresponds to each key is movieids of top 10 recommendations using euclidean

def getRecomDict_Movie(model):
    recommendtomovie_dict = {}
    for movieid in model.movies.keys():
        mymovies = []
        for movie in model.similar_items(movieid, n = 10):
            mymovies.append(model.movies[movie[0]]['movieid'])
        print("movie = ", movieid)
        print(mymovies)
        recommendtomovie_dict[movieid] = mymovies
    return recommendtomovie_dict