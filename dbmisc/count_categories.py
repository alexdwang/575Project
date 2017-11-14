#!/usr/bin/env python3


def main():
    categories = set()
    with open('data/movies.dat', 'r') as lines:
        for line in lines:
            cats = line.split(sep='::')[2]
            l_cat = cats.split(sep='|')
            for cat in l_cat:
                categories.add(cat.strip())
    for cat in categories:
        print(cat)


if __name__ == '__main__':
    main()
