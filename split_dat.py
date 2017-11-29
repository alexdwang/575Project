import os
import random
from collections import defaultdict, OrderedDict

def relative_path(path):
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return  os.path.normpath(path)


def split_dat_matrix():
    split_rate = 0.8
    path = relative_path('ratings.dat')
    lines = open(path, 'r', encoding='UTF-8').readlines()
    array = []
    for line in lines:
        array.append(line)
    test_num = int((1 - split_rate) * len(array))
    indexes = random.sample([i for i in range(len(array))], test_num)
    indexes.sort()
    indexes.reverse()
    test_array = []
    for index in indexes:
        test_array.append(array.pop(index))
    f = open("test.dat", "wb")
    start = 2
    for row in test_array:
        text1 = row.encode("UTF-8")
        f.write(text1)
    f.close()
    f = open("train.dat", "wb")
    start = 2
    for row in array:
        text1 = row.encode("UTF-8")
        f.write(text1)
    f.close()
    return

split_dat_matrix()
