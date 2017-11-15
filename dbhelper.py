from collections import OrderedDict

import psycopg2


class DatabaseHelper(object):

    def __init__(self, host='localhost', port=5432, user='projectdb', password='', db='projectdb'):
        self.conn = psycopg2.connect(host=host, port=str(port), user=user, password=password, dbname=db)

    def __del__(self):
        self.conn.close()

    def get_all_users(self):
        uids = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT * FROM users')
            uids = [[x[0], x[1], x[2]] for x in cursor]
        return uids

    def get_all_movies_svd(self):
        mids = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT * FROM movies')
            mids = [[x[0], x[1], x[2]] for x in cursor]
        return mids

    def get_all_ratings_raw(self):
        ratings = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT * FROM ratings')
            ratings = [[x[0], x[1], float(x[2])] for x in cursor]
        return ratings

    def get_all_movies(self):
        movies = OrderedDict()
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT id,name FROM movies ORDER BY id')
            for movie in cursor:
                movies[int(movie[0])] = { 'movieid': movie[0], 'title': movie[1] }
        return movies

    def get_all_reviews(self):
        reviews = {}
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT uid FROM ratings')
            # Create dicts for users
            for u in cursor:
                reviews[int(u[0])] = {}
            cursor.execute('SELECT * FROM ratings')
            for rating in cursor:
                reviews[int(rating[0])][int(rating[1])] = {
                    'userid': rating[0],
                    'movieid': rating[1],
                    'rating': float(rating[2])
                }
        return reviews

    def save_knn_pearson_recommend_result(self, result, type='user'):
        table_name = b'recommend_by_user_pearson' if type == 'user' else b'recommend_by_movie_pearson'
        recommends = []
        for _id, movies in result.items():
            for mid in movies:
                recommends.append((_id, mid))
        print(recommends)
        with self.conn.cursor() as cursor:
            rcmd_data_text = b','.join(cursor.mogrify(b'(%s,%s)', row) for row in recommends)
            cursor.execute(b'INSERT INTO ' + table_name + b' VALUES ' + rcmd_data_text)
            self.conn.commit()

    def save_knn_euclidean_recommend_result(self, result, type='user'):
        table_name = b'recommend_by_user_euclidean' if type == 'user' else b'recommend_by_movie_euclidean'
        recommends = []
        for _id, movies in result.items():
            for mid in movies:
                recommends.append((_id, mid))
        with self.conn.cursor() as cursor:
            rcmd_data_text = b','.join(cursor.mogrify(b'(%s,%s)', row) for row in recommends)
            cursor.execute(b'INSERT INTO ' + table_name + b' VALUES ' + rcmd_data_text)
            self.conn.commit()


def test():
    db = DatabaseHelper(password='asdfghjkl')
    result = {1: [989, 1830, 3172, 3233, 3382, 3607, 3656, 3881, 787, 3245, 53, 2503, 2905, 3888, 3517, 527, 2019, 318, 1178, 922], 2: [787, 989, 1830, 2480, 3172, 3233, 3280, 3382, 3607, 3656, 3881, 3245, 53, 3888, 2503, 2905, 318, 2019, 1148, 922], 3: [787, 989, 2480, 3172, 3233, 3280, 3382, 3607, 3656, 3881, 3245, 53, 3888, 2503, 2905, 2019, 670, 318, 922, 50], 4: [787, 989, 2480, 3172, 3233, 3607, 3656, 3881, 3245, 2503, 2905, 3888, 2019, 318, 53, 1148, 527, 50, 2444, 922], 5: [787, 989, 1830, 3172, 3233, 3280, 3382, 3607, 3656, 3881, 53, 3245, 2503, 3888, 2905, 2309, 2019, 578, 2444, 922], 6: [787, 989, 3172, 3233, 3382, 3607, 3656, 3881, 3245, 53, 2503, 2905, 598, 3888, 318, 2019, 745, 1148, 2309, 1178], 7: [787, 989, 1830, 3172, 3233, 3607, 3656, 3881, 3888, 3245, 53, 2503, 2905, 2019, 318, 3517, 745, 50, 527, 578], 8: [787, 989, 1830, 3172, 3233, 3382, 3607, 3656, 3881, 53, 3888, 3245, 2503, 1787, 2905, 318, 2019, 578, 670, 2444], 9: [787, 989, 1830, 3172, 3233, 3280, 3382, 3607, 3656, 3881, 53, 3245, 2503, 3888, 2905, 2019, 318, 2309, 527, 670], 10: [989, 1830, 3172, 3233, 3280, 3382, 3607, 3656, 3881, 787, 3245, 53, 2503, 3888, 2905, 318, 3232, 2019, 1148, 50]}
    db.save_knn_euclidean_recommend_result(result, type='user')


if __name__ == '__main__':
    test()
