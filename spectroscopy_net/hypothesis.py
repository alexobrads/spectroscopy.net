from spectroscopy_net import data_gen, data_base
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
import pystan
import pickle
import os
import logging

import random

def ll_background(obs_1, ref_wl_high, ref_wl_low):
    return -np.ones(len(obs_1)) * np.log(ref_wl_high - ref_wl_low)

def ll_foreground(theta, obs_1, ref_wl, obs_px_err=1e-2, ref_wl_err=1e-4):
    wl_obs = np.polyval(theta, obs_1).reshape(-1, 1)
    D = spatial.distance_matrix(wl_obs, ref_wl.reshape(-1, 1))
    chisq = np.min(D, axis=1)**2
    ivar = ref_wl_err**-2
    return -0.5 * chisq * ivar

def ln_prior(w, theta):
    lp = np.zeros(np.atleast_1d(w).size)
    lp[(w > 1) | (w < 0)] = -np.inf
    return lp

def ln_likelihood(w, theta, obs_1, ref_wl, ref_wl_high, ref_wl_low):
    w = np.atleast_2d(w).T
    bg = np.log(1-w) + np.atleast_2d(ll_background(obs_1, ref_wl_high, ref_wl_low))
    fg = np.log(w) + np.atleast_2d(ll_foreground(theta, obs_1, ref_wl))
    return np.sum(special.logsumexp([bg, fg], axis=0), axis=1)

def ln_probability(w, theta, obs_1, ref_wl, ref_wl_high, ref_wl_low):
    return ln_prior(w, theta) + ln_likelihood(w, theta, obs_1, ref_wl, ref_wl_high, ref_wl_low)


def number_of_combinations(n, q):

    C = np.exp(special.loggamma(n + 1) - special.loggamma(q + 1) - special.loggamma(n + 1 - q))
    return int(np.round(C))




def build_stan():

    line_model = """
    data {
        int<lower=1> N;
        real y[N];
        real y_err[N];
        real x[N];
        real bg_lower;
        real bg_upper;
        }

    parameters {
        real<lower=0.5, upper=1> theta;
        real m;
        real<lower=0> c;
        real<lower=0> bg_variance;
        }

    model {
        for (n in 1:N) {
        target += log_mix(theta,
                        normal_lpdf(y[n] | m * x[n] + c, y_err[n]),
                        normal_lpdf(y[n] | m * x[n] + c, pow(pow(y_err[n], 2) + bg_variance, 0.5)));
        }
        }
        """

    sm = pystan.StanModel(model_code=line_model)
    with open("spectroscopy_net/data/model.pkl", "wb") as fp:
        pickle.dump(sm, fp)

    return sm




def load_stan_model(path):

    with open(path, "rb") as fp:
        model = pickle.load(fp)


    return model


def run_stan(
    path,
    len_data,
    px_obs,
    ref_match,
    ref_wl_low,
    ref_wl_high,
    theta_t

    ):

    model = load_stan_model(path)


    linear_data = {'N': len_data,
               'y': ref_match,
               'y_err' : 1e-4*np.ones_like(px_obs),
               'x': px_obs,
               'bg_lower': ref_wl_low,
               'bg_upper': ref_wl_high

               }

    op2 = model.optimizing(data=linear_data);

    print("FROM OPTIMIZING")
    theta_op = [op2[("m")], op2[("c")]]
    print("Theta OP:", op2[("m")], op2[("c")])
    after = np.sum((theta_t - theta_op)**2)
    print("Theta diff after op:", after)
    print("***************")



def solve(
    path,
    ref,
    obs_px,
    obs_amps,
    theta_true,
    tree,
    ratios,
    wave_length,
    ref_wl_high,
    ref_wl_low,
    n,
    q
    ):

    for obs_1, amps_1, theta_t in zip(obs_px, obs_amps, theta_true):

        min_bayes_factor = 10**9

        ln_bg_prob = np.sum(ll_background(obs_1, ref_wl_high, ref_wl_low))
        best_bayes_factor, best_theta = (None, None)

        t = number_of_combinations(n, q)

        for i, obs_idx in tqdm(enumerate(combinations(np.argsort(amps_1), q)), total=t, disable=True):

            px_obs_ = np.sort(obs_1[list(obs_idx)])
            ratios = (px_obs_[1:-1] - px_obs_[0])/(px_obs_[-1] - px_obs_[0])

            ds, idxs = map(np.atleast_1d, tree.query(ratios, k=1))

            for d, idx in zip(ds, idxs):

                wl_ref_ = np.sort(wave_length[idx])

                theta = np.polyfit(px_obs_, wl_ref_, 1)
                wl_obs_guess = np.polyval(theta, obs_1)
                D = spatial.distance_matrix(wl_obs_guess.reshape(-1,1), ref.reshape(-1, 1))
                D_min = np.min(D, axis=1)
                ind = D.argmin(1)
                wl_ref_match = ref[ind]


                ws = np.linspace(0, 1, 100)[1:-1]
                w = ws[np.argmax(ln_probability(ws, theta, obs_1, ref, ref_wl_high, ref_wl_low))]

                bayes_factor = np.exp(ln_probability(w, theta, obs_1, ref, ref_wl_high, ref_wl_low) - ln_bg_prob)


                if best_bayes_factor is None or bayes_factor > best_bayes_factor:
                    best_bayes_factor, best_theta = (bayes_factor, theta)

                if bayes_factor > min_bayes_factor:
                    break

            if best_bayes_factor > min_bayes_factor:
                print("FROM HYPOTHESIS")
                print("Solution found")
                print("Bayes Factor:", best_bayes_factor)
                print("True Theta:", theta_t)
                print("Best Theta:", best_theta)
                before = np.sum((best_theta - theta_t)**2)
                print("Theta Diff:", before)
                break

        else:
            print("FROM HYPOTHESIS")
            print("***************")
            print("Unsolved problem")
            print("Best Bayes Factor:", best_bayes_factor)
            print("True Theta:", theta_t)
            print("Best Theta:", best_theta)
            before = np.sum((best_theta - theta_t)**2)
            print("Theta Diff:", before)


        run_stan(
            path,
            np.shape(obs_1)[0],
            obs_1,
            wl_ref_match,
            ref_wl_low,
            ref_wl_high,
            theta_t
            )
