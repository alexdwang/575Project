from flask import Flask, request, jsonify
from dbhelper import DatabaseHelper


app = Flask(__name__)


def get_recommend_by_user(userid, algo='euclidean'):
    db = DatabaseHelper(password='asdfghjkl')
    if algo == 'euclidean':
        return {str(userid): db.get_recommend_by_user_euclidean(userid)}
    elif algo == 'pearson':
        return {str(userid): db.get_recommend_by_user_pearson(userid)}
    elif algo == 'svd':
        return {str(userid): db.get_recommend_by_user_svd(userid)}
    else:
        return None


def get_recommend_by_movie(movieid, algo='euclidean'):
    db = DatabaseHelper(password='asdfghjkl')
    if algo == 'euclidean':
        return {str(movieid): db.get_recommend_by_movie_euclidean(movieid)}
    elif algo == 'pearson':
        return {str(movieid): db.get_recommend_by_movie_pearson(movieid)}
    else:
        return None


def return_err():
    return jsonify({'error': 'Invalid argument'})


@app.route('/api/getInfo', methods=['POST'])
def router():
    request_data = request.get_json()
    if not request_data:
        return return_err()
    algorithm = request_data.get('algorithm', None)
    if 'userid' in request_data:
        # Recommend by user
        result = get_recommend_by_user(request_data['userid'], algo=algorithm)
        if result:
            return jsonify(result)
        else:
            return return_err()
    elif 'movieid' in request_data:
        # Recommend by movies
        result = get_recommend_by_movie(request_data['movieid'], algo=algorithm)
        if result:
            return jsonify(result)
        else:
            return return_err()
    else:
        # Exception handling
        return_err()
