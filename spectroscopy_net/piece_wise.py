import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, special, optimize as op
from itertools import combinations
from tqdm import tqdm
from time import time
from IPython.display import Latex
import pystan
import spectroscopy_net as sn
import math

import random

def generate_data(theta, N_ref, N_obs, N_pix, f=0.1, wl_lower=300, wl_upper=1200,
                  intrinsic_px_obs_err=1e-3, intrinsic_wl_ref_err=1e-2,
                  amp_obs_scale=10000, amp_ref_scale=65535):
    px_obs = np.random.uniform(0, N_pix, N_obs)
    wl_obs = np.polyval(theta, px_obs)
    K = int(N_obs * (1 - f))
    idx = np.random.choice(N_obs, K, replace=False)
    wl_ref = np.hstack([
        wl_obs[idx],
        np.random.uniform(wl_lower, wl_upper, N_ref - K)
    ])
    unscaled_amplitudes = np.random.multivariate_normal([0, 0],
                                                        [[1, 0.9], [0.9, 1]],
                                                        size=N_ref)
    amp_obs_scale, amp_ref_scale = (10000, 65535)
    amplitudes = (unscaled_amplitudes - np.min(unscaled_amplitudes, axis=0)) \
               / np.ptp(unscaled_amplitudes, axis=0) \
               * np.array([amp_obs_scale, amp_ref_scale])
    amp_obs_known, amp_ref = (amplitudes.T[0][:K], amplitudes.T[1])
    amp_obs = np.random.uniform(0, amp_obs_scale, size=N_obs)
    amp_obs[idx] = amp_obs_known
    px_obs_err = np.random.normal(intrinsic_px_obs_err, intrinsic_px_obs_err/10, size=N_obs)
    wl_ref_err = np.random.normal(intrinsic_wl_ref_err, intrinsic_wl_ref_err/10, size=N_ref)
    px_obs += np.random.normal(0, intrinsic_px_obs_err, size=N_obs)
    wl_ref += np.random.normal(0, intrinsic_wl_ref_err, size=N_ref)
    idx = np.argsort(px_obs)
    idx2 = np.argsort(wl_ref)
    obs = (px_obs[idx], amp_obs[idx], px_obs_err[idx], wl_obs[idx])
    ref = (wl_ref[idx2], amp_ref[idx2], wl_ref_err[idx2])
    return obs, ref

def number_of_combinations(n, q):
    C = np.exp(special.loggamma(n + 1) - special.loggamma(q + 1) - special.loggamma(n + 1 - q))
    return int(np.round(C))

def grow_tree(wl_ref, p=1):
    if p < 1:
        raise ValueError("p > 1")
    q = p + 2
    n = len(wl_ref)
    t = number_of_combinations(n, p + 2)
    indices = np.memmap("indices.memmap", mode="w+", shape=(t, q), dtype=int)
    ratios = np.memmap("ratios.memmap", mode="w+", shape=(t, p), dtype=float)
    for i, idx in tqdm(enumerate(combinations(range(n), q)), total=t, disable=True):
        v = np.sort(wl_ref[list(idx)])
        ratios[i] = (v[1:-1] - v[0])/(v[-1] - v[0])
        indices[i] = idx
    tree = spatial.KDTree(ratios)
    return (indices, ratios, tree)

def split_test_data(obs, n):
    px_obs, amp_obs, px_obs_err, px_sol = obs
    split = np.array_split(np.argsort(px_obs), n)
    data = []
    for i in range(0,n):
        idx = split[i]
        px = px_obs[idx]
        amp = amp_obs[idx]
        px_err = px_obs_err[idx]
        sol = px_sol[idx]
        d = (px, amp, px_err, sol)
        data.append(d)
    return data

def split_real_data(obs, n):
    px_obs, px_sol = obs
    split = np.array_split(np.argsort(px_obs), n)
    data = []
    for i in range(0,n):
        idx = split[i]
        px = px_obs[idx]
        sol = px_sol[idx]
        d = (px, sol)
        data.append(d)
    return data

def get_gradient(theta):
    if theta[0] > 0:
         grad = 1
    else:
        grad = -1
    return grad

def sol_range(theta, obs):
    min = np.polyval(theta, np.min(obs))
    max = np.polyval(theta, np.max(obs))
    return ((np.min(obs), min), (np.max(obs), max))

def check_grad(conditions):
    n = len(conditions)
    grad = []
    for i in conditions:
        grad.append(i[0])
    tot = np.sum(np.asarray(grad))
    if tot == n:
        a = tot == n
    elif tot == -n:
        a = tot == -n
    else:
        a = False
    return a

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def find_intersect(conditions):
    p = np.sort(np.array(conditions)[:, :], axis=0)
    conditions = p.tolist()
    points = []
    for i in conditions:
        a1 = np.array([i[1][0], i[1][1]])
        a2 = np.array([i[2][0], i[2][1]])
        points.append((a1, a2))
    intersect = []
    for i in range(0, len(points)-1):
        intersect1 = seg_intersect(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        intersect.append(intersect1)
    return intersect

def check_intersect_on_ccd(conditions):
    p = np.sort(np.array(conditions)[:, :], axis=0)
    conditions = p.tolist()
    intersects = find_intersect(conditions)
    result = []
    for i in intersects:
        if  1 < i[0] < 2048:
            a = True
        else:
            a = False
        result.append(a)
    if np.array(result).all() == np.ones((1, len(result)), dtype=bool)[0].all():
        return True
    else:
        return False

def check_ends(conditions):
    p = np.sort(np.array(conditions)[:, :], axis=0)
    conditions = p.tolist()
    grad = conditions[0][0]
    if grad == 1:
        x_values = list()
        y_values = list()
        for i in range(0, len(conditions)):
            for j in range(1, 3):
                x = conditions[i][j][0]
                y = conditions[i][j][1]
                x_values.append(x)
                y_values.append(y)
        idx = np.argsort(x_values)
        x_values = np.asarray(x_values)[idx]
        y_values = np.asarray(y_values)[idx]
        range_y = y_values[1] - y_values[0]
        result1 = lambda y_values: np.all(y_values[:-1] <= y_values[1:])
        results = result1(y_values)

    if grad == -1:
        x_values = list()
        y_values = list()
        for i in range(0, len(conditions)):
            for j in range(1, 3):
                x = conditions[i][j][0]
                y = conditions[i][j][1]
                x_values.append(x)
                y_values.append(y)
        idx = np.argsort(x_values)
        x_values = np.asarray(x_values)[idx]
        y_values = np.asarray(y_values)[idx]
        range_y = y_values[1] - y_values[0]
        result1 = lambda y_values: np.all(y_values[:-1] >= y_values[1:])
        results = result1(y_values)


    return results

def wl_p_pixel(conditions):
    res = []
    for i in range(0, len(conditions)):
        res1 = (conditions[i][2][0] - conditions[i][1][0])/(conditions[i][2][1] - conditions[i][1][1])
        res.append(res1)
    results = []
    for i in range(0, len(conditions) -1 ):
        margin = res[i]*0.2
        a = res[i] < res[i+1] < res[i]+margin
        results.append(a)
    if np.array(results).all() == np.ones((1, len(results)), dtype=bool)[0].all():
        return True
    else:
        return False

def check_intersect_between_lines(conditions):
    i = find_intersect(conditions)
    s = np.sort(np.array(conditions)[:, 1:np.shape(conditions)[0]], axis=0)
    e = np.sort(np.array(conditions)[:, 2:np.shape(conditions)[0]+1], axis=0)

    results = []
    for n in range(0, len(i)):
        if e[n][0][0] < i[n][0] < s[n+1][0][0]:
            results.append(True)


        else:
            results.append(False)


    if np.array(results).all() == np.ones((1, len(results)), dtype=bool)[0].all():
        return True

    else:
        return False



def do_these_predictions_make_sense(conditions):

    a = check_grad(conditions)
    b = check_intersect_on_ccd(conditions)
    c = check_ends(conditions)
    d = check_intersect_between_lines(conditions)
    e = wl_p_pixel(conditions)

    result = np.array((a, b, c, d, e))
    if np.array(result).all() == np.ones((1, len(result)), dtype=bool)[0].all():
        return True

    else:
        return False




def ll_background(px_obs, wl_upper, wl_lower):
    return -np.ones(len(px_obs)) * np.log(wl_upper - wl_lower)

def ln_prior(w):
    lp = np.zeros(np.atleast_1d(w).size)
    lp[(w > 1) | (w < 0)] = -np.inf
    return lp








def ll_foreground_linear(theta, px_obs, wl_ref, px_obs_err=1e-3, wl_ref_err=1e-4):

    wl_obs = np.polyval(theta, px_obs).reshape(-1, 1)
    D = spatial.distance_matrix(wl_obs, wl_ref.reshape(-1, 1))
    idx = np.argmin(D, axis=1)
    best_match = wl_ref[idx]
    chisq = np.min(D, axis=1)**2
    ivar = wl_ref_err**-2
    return -0.5 * chisq * ivar, best_match

def ln_likelihood_linear(w, theta, px_obs, wl_ref, wl_upper, wl_lower):
    w = np.atleast_2d(w).T
    bg = np.log(1-w) + np.atleast_2d(ll_background(px_obs, wl_upper, wl_lower))

    fg_ln, best_match = ll_foreground_linear(theta, px_obs, wl_ref)

    fg = np.log(w) + np.atleast_2d(fg_ln)
    return np.sum(special.logsumexp([bg, fg], axis=0), axis=1), best_match

def ln_probability_linear(w, theta, px_obs, wl_ref, wl_upper, wl_lower):
    like_ln, best_match = ln_likelihood_linear(w, theta, px_obs, wl_ref, wl_upper, wl_lower)
    ln_prob = ln_prior(w) + like_ln
    return ln_prob, best_match

def ll_foreground_side(theta, px_obs, wl_ref, px_obs_err=1e-3, wl_ref_err=1e-4):

    wl_obs = np.polyval(theta, px_obs).reshape(-1, 1)
    D = spatial.distance_matrix(wl_obs, wl_ref.reshape(-1, 1))
    idx = np.argmin(D, axis=1)
    best_match = wl_ref[idx]
    chisq = np.min(D, axis=1)**2
    ivar = 100
    return -0.5 * chisq * ivar, best_match

def ln_likelihood_side(w, theta, px_obs, wl_ref, wl_upper, wl_lower):
    w = np.atleast_2d(w).T
    bg = np.log(1-w) + np.atleast_2d(ll_background(px_obs, wl_upper, wl_lower))

    fg_ln, best_match = ll_foreground_side(theta, px_obs, wl_ref)

    fg = np.log(w) + np.atleast_2d(fg_ln)
    return np.sum(special.logsumexp([bg, fg], axis=0), axis=1), best_match

def ln_probability_side(w, theta, px_obs, wl_ref, wl_upper, wl_lower):
    like_ln, best_match = ln_likelihood_side(w, theta, px_obs, wl_ref, wl_upper, wl_lower)
    ln_prob = ln_prior(w) + like_ln
    return ln_prob, best_match

def calc_bayes(theta, px_obs, wl_ref, wl_upper, wl_lower):
    ln_bg_prob = np.sum(ll_background(px_obs, wl_upper, wl_lower))
    ws = np.linspace(0, 1, 100)[1:-1]
    ln_prob_best, best_match = ln_probability_side(ws, theta, px_obs, wl_ref, wl_upper, wl_lower)
    w = ws[np.argmax(ln_prob_best)]
    ln_prob, best_match = ln_probability_side(w, theta, px_obs, wl_ref, wl_upper, wl_lower)
    bayes_factor = np.exp(ln_prob - ln_bg_prob)
    return bayes_factor

def solve_linear(obs, ref, trees, min_bayes_factor=10**9, deg=1, k=10, p=1):

    px_obs, amp_obs, px_obs_err, px_sol = obs
    wl_ref, amp_ref, wl_ref_err = ref
    indices, ratios, tree = trees

    wl_lower = np.min(wl_ref)
    wl_upper = np.max(wl_ref)

    best_bayes_factor, best_theta = (None, None)

    C = number_of_combinations(len(wl_ref), p+2)
    ln_bg_prob = np.sum(ll_background(px_obs, wl_upper, wl_lower))


    for i, obs_idx in tqdm(enumerate(combinations(np.argsort(amp_obs), p + 2)), total=C):
        px_obs_ = np.sort(px_obs[list(obs_idx)])
        ratios = (px_obs_[1:-1] - px_obs_[0])/(px_obs_[-1] - px_obs_[0])
        ds, idxs = map(np.atleast_1d, tree.query(ratios, k=k))

        for d, idx in zip(ds, idxs):
            wl_ref_ = np.sort(wl_ref[indices[idx]])
            theta = np.polyfit(px_obs_, wl_ref_, deg)
            ws = np.linspace(0, 1, 100)[1:-1]

            ln_prob_best, best_match = sn.ln_probability_linear(ws, theta, px_obs, wl_ref, wl_upper, wl_lower)

            w = ws[np.argmax(ln_prob_best)]

            ln_prob, best_match = sn.ln_probability_linear(w, theta, px_obs, wl_ref, wl_upper, wl_lower)

            bayes_factor = np.exp(ln_prob - ln_bg_prob)

            if best_bayes_factor is None or bayes_factor > best_bayes_factor:
                best_bayes_factor, best_theta = (bayes_factor, theta)

            if bayes_factor > min_bayes_factor:
                break

        if best_bayes_factor > min_bayes_factor:
            print("Solution found")
            print("Bayes Factor:", best_bayes_factor)
            print("Best Theta:", best_theta)
            break

    else:
        print("Unsolved problem")
        print("Bayes Factor:", best_bayes_factor)
        print("Best Theta:", best_theta)

    return (best_bayes_factor, best_theta)

def solve_piecewise_test(obs, ref, trees, k=100, min_bayes_factor=10**9, deg=1, p=1, max_split_depth=2):

    wl_ref, amp_ref, wl_ref_err = ref
    indices, ratios, tree = trees
    wl_lower = np.min(wl_ref)
    wl_upper = np.max(wl_ref)

    split = sn.split_test_data(obs, max_split_depth)
    idx_range = np.shape(split)[0]

    start_idx = 0
    end_idx = idx_range
    data = np.vstack((split, split))
    end = 2*idx_range
    result = 0

    fig, axs = plt.subplots(nrows=max_split_depth, sharex=True, sharey=False, figsize=(10,10))

    t = 0

    while result == 0:
        conditions = []

        for x in range(start_idx, end_idx, 1):

            px_obs, amp_obs, px_obs_err, px_sol = data[x]

            best_bayes_factor, best_theta, best_conditions, best_intersection = (None, None, None, None)
            best_px, best_wl = (None, None)

            C = sn.number_of_combinations(len(wl_ref), p+2)

            for i, obs_idx in tqdm(enumerate(combinations(np.argsort(amp_obs), p + 2)), total=C, disable=True):

                px_obs_ = np.sort(px_obs[list(obs_idx)])
                ratios = (px_obs_[1:-1] - px_obs_[0])/(px_obs_[-1] - px_obs_[0])
                ds, idxs = map(np.atleast_1d, tree.query(ratios, k=k))

                for d, idx in zip(ds, idxs):

                    wl_ref_ = np.sort(wl_ref[indices[idx]])
                    theta = np.polyfit(px_obs_, wl_ref_, deg)
                    grad1 = sn.get_gradient(theta)
                    range1 = sn.sol_range(theta, px_obs)
                    conditions1 = list((grad1, range1[0], range1[1]))

                    D = spatial.distance_matrix(wl_ref_.reshape(-1, 1), wl_ref.reshape(-1, 1))
                    idx_match = np.argmin(D, axis=1)
                    best_match = wl_ref[idx_match]

                    if len(conditions) == 0:

                        bayes_factor = sn.calc_bayes(theta, px_obs, wl_ref, wl_upper, wl_lower)

                    if len(conditions) >= 1:

                        con = (*conditions, conditions1)
                        result = sn.do_these_predictions_make_sense(con)

                        if result == True:
                            bayes_factor = sn.calc_bayes(theta, px_obs, wl_ref, wl_upper, wl_lower)

                        else:
                            bayes_factor = 10


                    if best_bayes_factor is None or bayes_factor > best_bayes_factor:
                        best_bayes_factor, best_theta, best_conditions = (bayes_factor, theta, conditions1)
                        best_px, best_wl = (px_obs_, best_match)

                    if bayes_factor > min_bayes_factor:
                        break

                if best_bayes_factor > min_bayes_factor:
                    print("Solution found")
                    print("Bayes Factor:", best_bayes_factor)
                    conditions.append(best_conditions)
                    axs[t].scatter(px_obs, np.polyval(best_theta, px_obs))
                    axs[t].plot(obs[0], obs[3])
                    axs[t].set(adjustable='box')
                    axs[t].set_ylim([4700, 5000])
                    break

            else:
                print("Unsolved problem")
                print("Bayes Factor:", best_bayes_factor)
                conditions.append(best_conditions)
                axs[t].scatter(px_obs, np.polyval(best_theta, px_obs))
                axs[t].plot(obs[0], obs[3])
                axs[t].set(adjustable='box')
                axs[t].set_ylim([4700, 5000])

        if sn.check_ends(conditions) == False:
            print("No match")
            start_idx = start_idx + 1
            end_idx = end_idx + 1
            result = 0
            t = t + 1

        if sn.check_ends(conditions) == True:
            print("likely solution")
            result = 1
            t = t + 1

        if end_idx == end:
            result = 1

    return conditions

def solve_piecewise_mike(obs, ref, trees, k=100, min_bayes_factor=10**9, deg=1, p=1, max_split_depth=2):

    wl_ref = np.sort(ref)
    indices, ratios, tree = trees
    wl_lower = np.min(wl_ref)
    wl_upper = np.max(wl_ref)

    split = sn.split_real_data(obs, max_split_depth)
    idx_range = np.shape(split)[0]

    start_idx = 0
    end_idx = idx_range
    data = np.vstack((split, split))
    end = 2*idx_range
    result = 0

    fig, axs = plt.subplots(nrows=max_split_depth, sharex=True, sharey=False, figsize=(10,10))

    t = 0

    while result == 0:
        conditions = []

        for x in range(start_idx, end_idx, 1):

            px_obs, px_sol = data[x]

            best_bayes_factor, best_theta, best_conditions, best_intersection = (None, None, None, None)
            best_px, best_wl = (None, None)

            C = sn.number_of_combinations(len(wl_ref), p+2)

            for i, obs_idx in tqdm(enumerate(combinations(range(0, len(px_obs)), p + 2)), total=C):

                px_obs_ = np.sort(px_obs[list(obs_idx)])
                ratios = (px_obs_[1:-1] - px_obs_[0])/(px_obs_[-1] - px_obs_[0])
                ds, idxs = map(np.atleast_1d, tree.query(ratios, k=k))

                for d, idx in zip(ds, idxs):

                    wl_ref_ = np.sort(wl_ref[indices[idx]])
                    theta = np.polyfit(px_obs_, wl_ref_, deg)
                    grad1 = sn.get_gradient(theta)
                    range1 = sn.sol_range(theta, px_obs)
                    conditions1 = list((grad1, range1[0], range1[1]))

                    D = spatial.distance_matrix(wl_ref_.reshape(-1, 1), wl_ref.reshape(-1, 1))
                    idx_match = np.argmin(D, axis=1)
                    best_match = wl_ref[idx_match]

                    if len(conditions) == 0:

                        bayes_factor = sn.calc_bayes(theta, px_obs, wl_ref, wl_upper, wl_lower)

                    if len(conditions) >= 1:

                        con = (*conditions, conditions1)
                        result = sn.do_these_predictions_make_sense(con)

                        if result == True:
                            bayes_factor = sn.calc_bayes(theta, px_obs, wl_ref, wl_upper, wl_lower)

                        else:
                            bayes_factor = 10


                    if best_bayes_factor is None or bayes_factor > best_bayes_factor:
                        best_bayes_factor, best_theta, best_conditions = (bayes_factor, theta, conditions1)
                        best_px, best_wl = (px_obs_, best_match)

                    if bayes_factor > min_bayes_factor:
                        break

                if best_bayes_factor > min_bayes_factor:
                    print("Solution found")
                    print("Bayes Factor:", best_bayes_factor)
                    conditions.append(best_conditions)
                    axs[t].scatter(best_px, best_wl)
                    axs[t].plot(obs[0], obs[1])
                    axs[t].set(adjustable='box')
                    axs[t].set_ylim([4700, 5000])
                    break

            else:
                print("Unsolved problem")
                print("Bayes Factor:", best_bayes_factor)
                conditions.append(best_conditions)
                axs[t].scatter(px_obs, np.polyval(best_theta, px_obs))
                axs[t].plot(obs[0], obs[1])
                axs[t].set(adjustable='box')
                axs[t].set_ylim([4700, 5000])

        if sn.check_ends(conditions) == False:
            print("No match")
            start_idx = start_idx + 1
            end_idx = end_idx + 1
            result = 0
            t = t + 1

        if sn.check_ends(conditions) == True:
            print("likely solution")
            result = 1
            t = t + 1

        if end_idx == end:
            result = 1

    return conditions
