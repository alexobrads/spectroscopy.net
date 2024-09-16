from spectroscopy_net import data_gen
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sqlite3
from scipy import spatial, special, optimize as op
from itertools import combinations
from tqdm import tqdm
from time import time
from IPython.display import Latex
import pystan
import pandas as pd

def from_scratch_db(db):
    create_db(db)
    build_db(db, n, q, ref)


def create_db(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('''CREATE TABLE ref
             (ratio, wl_1, wl_2, wl_3)''')
    conn.commit()

def build_db(db, n, q, ref):
    c, conn = connect_db(db)
    data = build_data(n, q, ref)
    insert_data(c, conn, data[0], data[1], data[2], data[3])

def build_data(n, q, ref):
    t = number_of_combinations(n, q)
    wl_1 = []
    wl_2 = []
    wl_3 = []
    ratios = []

    for i, idx in tqdm(enumerate(combinations(range(n), 3)), total=t):
            v = np.array(ref[list(idx)])
            ratios.append((v[1:-1] - v[0])/(v[-1] - v[0]))
            wl_1.append(v[0])
            wl_2.append(v[1])
            wl_3.append(v[2])

    data = (ratios, wl_1, wl_2, wl_3)

    return data

def insert_data(c, conn, ratios, wl_1, wl_2, wl_3):
    for i, j, k, w in zip(ratios, wl_1, wl_2, wl_3):
        data = [float(i), float(j), float(k), float(w)]
        c.execute("INSERT INTO ref(ratio, wl_1, wl_2, wl_3) values(?, ?, ?, ?)", data)

    conn.commit()

def connect_db(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    return c, conn

def number_of_combinations(n, q):

    C = np.exp(special.loggamma(n + 1) - special.loggamma(q + 1) - special.loggamma(n + 1 - q))
    return int(np.round(C))

def find_match(c, num):
    c.execute("SELECT * FROM ref ORDER BY ABS(? - ratio)", (num,))
    return c.fetchone()

def build_tree(ref ,n_each, q):

    n_times = np.size(ref)/n_each

    t = number_of_combinations(n_each, q)
    total_length = int(t*n_times)

    ref_data = np.split(np.sort(ref), n_times)


    wave_length = np.memmap("spectroscopy_net/data/wave_length.memmap", mode="w+", shape=(total_length, q), dtype=float)
    ratios = np.memmap("spectroscopy_net/data/ratios.memmap", mode="w+", shape=(total_length, 1), dtype=float)

    for n in range(0, np.shape(ref_data)[0]):

        for i, idx in tqdm(enumerate(combinations(range(np.size(ref_data[0])), q)), total=t, disable=True):

            v = np.sort(ref_data[n][list(idx)]);

            ratios[i] = (v[1:-1] - v[0])/(v[-1] - v[0]);
            wave_length[i] = v;

    tree = spatial.KDTree(ratios);

    return tree, ratios, wave_length
