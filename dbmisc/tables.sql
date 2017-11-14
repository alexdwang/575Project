CREATE TABLE users (
    id INT PRIMARY KEY,
    sex CHAR(1),
    age SMALLINT
);

CREATE TABLE movies (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    Documentary BOOL,
	Comedy BOOL,
	Adventure BOOL,
	Musical BOOL,
	Crime BOOL,
	War BOOL,
	Fantasy BOOL,
	Childrens BOOL,
	Western BOOL,
	FilmNoir BOOL,
	Horror BOOL,
	SciFi BOOL,
	Animation BOOL,
	Action BOOL,
	Mystery BOOL,
	Thriller BOOL,
	Drama BOOL,
	Romance BOOL
);

CREATE TABLE ratings (
    uid INT REFERENCES users (id),
    mid INT REFERENCES movies (id),
    rating NUMERIC(2,1)
);

CREATE TABLE recommend_by_user_pearson (
	uid INT REFERENCES users (id),
	mid INT REFERENCES movies (id)
);

CREATE TABLE recommend_by_user_euclidean (
	uid INT REFERENCES users (id),
	mid INT REFERENCES movies (id)
);

CREATE TABLE recommend_by_movie_pearson (
	mid INT,
	to_mid INT
);

CREATE TABLE recommend_by_movie_euclidean (
	mid INT,
	to_mid INT
);
