#!/usr/bin/env python3

import psycopg2


HOST = 'localhost'
PORT = '5432'
USER = 'projectdb'
PASSWORD = 'asdfghjkl'
DB = 'projectdb'


def main():
    conn = psycopg2.connect(dbname=DB, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()

    # Read users data into database
    users = []
    with open('data/users.dat', 'r') as userfile:
        for line in userfile:
            u = line.split(sep='::')
            users.append((u[0], u[1], u[2]))
    users_data_text = b','.join(cur.mogrify(b'(%s,%s,%s)', row) for row in users)
    cur.execute(b'INSERT INTO users VALUES ' + users_data_text)
    conn.commit()

    # Read movies data into database
    category_list = ['Documentary', 'Comedy', 'Adventure', 'Musical', 'Crime', 'War',
                     'Fantasy', 'Childrens', 'Western', 'FilmNoir', 'Horror', 'SciFi',
                     'Animation', 'Action', 'Mystery', 'Thriller', 'Drama', 'Romance'
                    ]
    movies = []
    with open('data/movies.dat', 'r') as moviefile:
        for line in moviefile:
            mline = line.split(sep='::')
            m = [mline[0], mline[1]]
            curr_cat_list = list(map(lambda x: x.strip().replace("'", '').replace('-', ''),
                                     mline[2].split(sep='|'))
                                )
            for c in category_list:
                m.append('1' if c in curr_cat_list else '0')
            movies.append(tuple(m))
    movies_data_text = b','.join(cur.mogrify(b'(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', row) for row in movies)
    cur.execute(b'INSERT INTO movies VALUES ' + movies_data_text)
    conn.commit()

    # Read ratings data into database
    with open('data/ratings.dat', 'r') as ratingfile:
        count = 0
        ratings = []
        for line in ratingfile:
            rline = line.split(sep='::')
            r = (rline[0], rline[1], rline[2])
            ratings.append(r)
            count += 1
            if count >= 20000:
                ratings_data_text = b','.join(cur.mogrify(b'(%s,%s,%s)', row) for row in ratings)
                cur.execute(b'INSERT INTO ratings VALUES ' + ratings_data_text)
                conn.commit()
                ratings.clear()
                count = 0
        if ratings:
            ratings_data_text = b','.join(cur.mogrify(b'(%s,%s,%s)', row) for row in ratings)
            cur.execute(b'INSERT INTO ratings VALUES ' + ratings_data_text)
            conn.commit()

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
