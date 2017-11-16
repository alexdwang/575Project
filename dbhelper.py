from collections import OrderedDict

import psycopg2


class DatabaseHelper(object):

    def __init__(self, host='localhost', port=5432, user='projectdb', password='', db='projectdb'):
        self.conn = psycopg2.connect(host=host, port=str(port), user=user, password=password, dbname=db)
        self.categories = ['Documentary', 'Comedy', 'Adventure', 'Musical', 'Crime', 'War',
                           'Fantasy', 'Childrens', 'Western', 'FilmNoir', 'Horror', 'SciFi',
                           'Animation', 'Action', 'Mystery', 'Thriller', 'Drama', 'Romance'
                          ]

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

    def save_svd_recommend_result(self, result):
        recommends = []
        for id, movies in result.items():
            for mid in movies:
                recommends.append((id, mid))

        with self.conn.cursor() as cursor:
            rcmd_data_text = b','.join(cursor.mogrify(b'(%s,%s)', row) for row in recommends)
            cursor.execute(b'INSERT INTO svd_recommend VALUES ' + rcmd_data_text)
            self.conn.commit()

    def get_recommend_by_user_euclidean(self, uid):
        recommends = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT name,documentary,comedy,adventure,musical,crime,war,fantasy,childrens,western,filmnoir,horror,scifi,animation,action,mystery,thriller,drama,romance FROM movies,recommend_by_user_euclidean WHERE recommend_by_user_euclidean.uid=' + str(uid) + ' AND movies.id=recommend_by_user_euclidean.mid')
            for row in cursor:
                genres = []
                for i in range(1, len(row)):
                    if row[i]:
                        genres.append(self.categories[i-1])
                recommends.append({'name': row[0], 'genre': ', '.join(genres)})
        return recommends

    def get_recommend_by_user_pearson(self, uid):
        recommends = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT name,documentary,comedy,adventure,musical,crime,war,fantasy,childrens,western,filmnoir,horror,scifi,animation,action,mystery,thriller,drama,romance FROM movies,recommend_by_user_pearson WHERE recommend_by_user_pearson.uid=' + str(uid) + ' AND movies.id=recommend_by_user_pearson.mid')
            for row in cursor:
                genres = []
                for i in range(1, len(row)):
                    if row[i]:
                        genres.append(self.categories[i-1])
                recommends.append({'name': row[0], 'genre': ', '.join(genres)})
        return recommends

    def get_recommend_by_user_svd(self, uid):
        recommends = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT name,documentary,comedy,adventure,musical,crime,war,fantasy,childrens,western,filmnoir,horror,scifi,animation,action,mystery,thriller,drama,romance FROM movies,svd_recommend WHERE svd_recommend.uid=' + str(uid) + ' AND movies.id=svd_recommend.mid')
            for row in cursor:
                genres = []
                for i in range(1, len(row)):
                    if row[i]:
                        genres.append(self.categories[i-1])
                recommends.append({'name': row[0], 'genre': ', '.join(genres)})
        return recommends

    def get_recommend_by_movie_euclidean(self, mid):
        recommends = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT name,documentary,comedy,adventure,musical,crime,war,fantasy,childrens,western,filmnoir,horror,scifi,animation,action,mystery,thriller,drama,romance FROM movies,recommend_by_movie_euclidean WHERE recommend_by_movie_euclidean.mid=' + str(mid) + ' AND movies.id=recommend_by_movie_euclidean.to_mid')
            for row in cursor:
                genres = []
                for i in range(1, len(row)):
                    if row[i]:
                        genres.append(self.categories[i-1])
                recommends.append({'name': row[0], 'genres': ', '.join(genres)})
        return recommends

    def get_recommend_by_movie_pearson(self, mid):
        recommends = []
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT name,documentary,comedy,adventure,musical,crime,war,fantasy,childrens,western,filmnoir,horror,scifi,animation,action,mystery,thriller,drama,romance FROM movies,recommend_by_movie_pearson WHERE recommend_by_movie_pearson.mid=' + str(mid) + ' AND movies.id=recommend_by_movie_pearson.to_mid')
            for row in cursor:
                genres = []
                for i in range(1, len(row)):
                    if row[i]:
                        genres.append(self.categories[i-1])
                recommends.append({'name': row[0], 'genres': ', '.join(genres)})
        return recommends


def test():
    db = DatabaseHelper(password='asdfghjkl')
    print(db.get_recommend_by_user_pearson(1))


if __name__ == '__main__':
    test()
