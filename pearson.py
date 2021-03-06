'''
Movie recommendation to users and find similar movies using pearson
'''

#  A method to get the 10 most recommended movies for a user
#  return a dictionary whose key set is userid
#  content corresponds to each key is movieids of top 10 recommendations using pearson

def getRecomDict_User(model):
    recommendtouser_dict = {}
    for userid in model.reviews.keys():
        mymovies = []
        for mid in model.predict_all_rankings(userid, 'pearson', 20):
            mymovies.append(model.movies[mid[0]]['movieid'])
        recommendtouser_dict[userid] = mymovies
    return recommendtouser_dict


#  A method to get the 10 most similar movies for given movies
#  return a dictionary whose key set is userid
#  content corresponds to each key is movieids of top 10 recommendations using pearson

def getRecomDict_Movie(model):
    recommendtomovie_dict = {}
    for movieid in model.movies.keys():
        mymovies = []
        for movie in model.similar_items(movieid, 'pearson', 10):
            mymovies.append(model.movies[movie[0]]['movieid'])
        recommendtomovie_dict[movieid] = mymovies
    return recommendtomovie_dict