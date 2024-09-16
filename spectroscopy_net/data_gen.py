import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def build_amps(
    ref_size
    ):

    unscaled_amplitudes = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]],size=(ref_size))
    obs_scale = 10000
    ref_scale = 65535
    amplitudes = (unscaled_amplitudes - np.min(unscaled_amplitudes, axis=0))/ np.ptp(unscaled_amplitudes, axis=0) * np.array([obs_scale, ref_scale])
    amp_ref = amplitudes.T[1]
    amp_obs = amplitudes.T[0]
    amps = (amp_obs, amp_ref)
    return  amps


def build_ref(
    ref_wl_low,
    ref_wl_high,
    ref_noise,
    ref_size
    ):

    wl_ref = np.random.uniform(ref_wl_low, ref_wl_high, size=(ref_size))
    wl_err = np.random.normal(0, ref_noise, ref_size)
    amps = build_amps(ref_size)
    ref = (wl_ref, wl_err, amps[1])
    return ref


def trim_ref(
    n_echelle,
    ref,
    ref_size,
    ccd_wl_low,
    ccd_wl_high
    ):

    amps = build_amps(ref_size)

    idx = np.where(ref[0] < ccd_wl_low)
    ref_range = np.delete([ref[0]], idx)
    amp = np.delete([amps[0]], idx)

    idx2 = np.where(ref_range > ccd_wl_high)
    ref_range = np.delete(ref_range, idx2)
    amp = np.delete(amp, idx2)

    ref_stack = np.repeat([ref_range], repeats=n_echelle, axis=0)
    amp_stack = np.repeat([amp], repeats=n_echelle, axis=0)

    return ref_stack, amp_stack

def create_theta(
    n_echelle,
    ccd_wl_low,
    ccd_wl_high,
    overlap,
    ref_stack
    ):

    nech = np.arange(0,n_echelle, 1)
    ccd_range = (ccd_wl_high - ccd_wl_low)
    split = ccd_range/(n_echelle)
    wl_split_low = nech*split + ccd_wl_low
    wl_split_high = np.sort(((-nech*split +ccd_wl_high)- overlap*(wl_split_low[1]-wl_split_low[0])))


    combine = np.sort(
                np.concatenate((wl_split_low, wl_split_high))
                )

    scaling  = np.array([combine[4] - combine[1]])
    shift = np.arange(ccd_wl_low, ccd_wl_high, (ccd_range/n_echelle))
    nech2 = np.arange(0,70, 70/n_echelle)

    m = np.squeeze(np.array([(-1048/scaling)]*n_echelle).transpose()*-(0.8**(nech2/70)))
    c = np.squeeze(np.array([(-shift+1)+1.5*np.flip(nech2),]))

    m = np.around(m, 2)
    c = np.around(c, 2)

    theta = (m, c)

    return theta

def create_outliers(
    n_echelle,
    px_echelle1,
    px_err_echelle1,
    py_echelle1,
    py_err_echelle1,
    wl_echelle1,
    amp_echelle1,
    n_outliers_low,
    n_outliers_high,
    ccd_noise
    ):

    outliers_px1 = []
    outliers_px_err1 = []
    outliers_py1 = []
    outliers_py_err1 = []
    outliers_wl1 = []
    outliers_amp1 = []

    nech = np.arange(0,70, 70/n_echelle)

    for j in nech:

        n = np.random.randint(n_outliers_low, n_outliers_high)


        outliers_px_1 = np.random.uniform(0, 2048, size=n)

        outliers_px_err_1 = np.random.normal(0,  ccd_noise, size=n)

        outliers_py_1 = create_y_pixels_for_outliers(outliers_px_1, n, j)

        outliers_py_err_1 = np.random.normal(0,  ccd_noise, size=n)


        outliers_ref1 = np.random.uniform(0, 0, size=n)
        outliers_amp_1 = np.random.uniform(0, 10000, size=n)

        outliers_px1.append(outliers_px_1)
        outliers_px_err1.append(outliers_px_err_1)

        outliers_py1.append(outliers_py_1)
        outliers_py_err1.append(outliers_py_err_1)

        outliers_wl1.append(outliers_ref1)
        outliers_amp1.append(outliers_amp_1)

    outliers = (outliers_px1, outliers_px_err1, outliers_py1, outliers_py_err1, outliers_wl1, outliers_amp1)

    return outliers



def add_outliers(
    n_echelle,
    px_echelle1,
    px_err_echelle1,
    py_echelle1,
    py_err_echelle1,
    wl_echelle1,
    amp_echelle1,
    n_outliers_low,
    n_outliers_high,
    ccd_noise
    ):


    outliers = create_outliers(
        n_echelle,
        px_echelle1,
        px_err_echelle1,
        py_echelle1,
        py_err_echelle1,
        wl_echelle1,
        amp_echelle1,
        n_outliers_low,
        n_outliers_high,
        ccd_noise
        )

    px_echelle2 = []
    px_err_echelle2 = []
    py_echelle2 = []
    py_err_echelle2 = []
    wl_echelle2 = []
    amp_echelle2 = []

    for a, b, c, d, e, f, g, h, i, j, k, l in zip(
        px_echelle1,
        outliers[0],
        px_err_echelle1,
        outliers[1],

        py_echelle1,
        outliers[2],
        py_err_echelle1,
        outliers[3],

        wl_echelle1,
        outliers[4],

        amp_echelle1,
        outliers[5]
        ):

        a1 = np.concatenate((a, b))
        idx = np.argsort(a1)
        a1 = a1[idx]
        px_echelle2.append(a1)


        b1 = np.concatenate((c, d))
        b1 = b1[idx]
        px_err_echelle2.append(b1)


        c1 = np.concatenate((e, f))
        c1 = c1[idx]
        py_echelle2.append(c1)


        d1 = np.concatenate((g, h))
        d1 = d1[idx]
        py_err_echelle2.append(d1)


        e1 = np.concatenate((i, j))
        e1 = e1[idx]
        wl_echelle2.append(e1)


        f1 = np.concatenate((k, l))
        f1 = f1[idx]
        amp_echelle2.append(f1)

        obs = (px_echelle2, px_err_echelle2, py_echelle2, py_err_echelle2, amp_echelle2, wl_echelle2)

    return obs

def solve(theta, values):
    solution = []
    theta_true = []
    for i, j, k in zip(theta[0], theta[1], values):
        m = np.round(i, 3)
        c = np.round(j*m, 3)
        m_t = np.round(1/i, 3)
        c_t = np.round(-j, 3)
        solution1 = np.polyval([m, c], k)
        solution.append(solution1)
        theta_true.append((m_t, c_t))

    sol = np.squeeze(np.asarray(solution))
    theta_true = np.squeeze(np.asarray(theta_true))

    return sol, theta_true


def create_x_pixels(
    ref,
    ref_size,
    n_echelle,
    ccd_wl_low,
    ccd_wl_high,
    overlap
    ):

    ref_stack, amp_echelle = trim_ref(
        n_echelle,
        ref,
        ref_size,
        ccd_wl_low,
        ccd_wl_high
        )

    theta = create_theta(
        n_echelle,
        ccd_wl_low,
        ccd_wl_high,
        overlap,
        ref_stack
        )

    px_echelle, theta_true = solve(theta, ref_stack)

    return ref_stack, amp_echelle, theta_true, px_echelle



def create_y_pixels_for_outliers(
    i,
    n,
    j
    ):

    c = 20*((np.ones((n,1))*j).transpose())+10 + ((np.ones((n,1))*j).transpose())*8
    y = 80*np.sin((i)/(250*math.pi))
    py = y + np.squeeze(c, axis=0)

    return py

def create_y_pixels(
    px_echelle,
    n_echelle
    ):

    nech = np.arange(0,70, 70/n_echelle)
    py_echelle = []
    for i, j in zip(px_echelle, nech):

        n = np.size(i)
        c = 20*((np.ones((n,1))*j).transpose())+10 + ((np.ones((n,1))*j).transpose())*8
        y = 80*np.sin((i)/(250*math.pi))
        py = np.add(y, c)
        py_echelle.append(py)

    return py_echelle


def build_obs(
    n_echelle,
    px_echelle,
    py_echelle,
    ref_stack,
    amp_echelle,
    n_each_echelle,
    ccd_noise,
    n_outliers_low,
    n_outliers_high
    ):

    px_echelle1 = []
    px_err_echelle1 = []
    py_echelle1 = []
    py_err_echelle1 = []
    wl_echelle1 = []
    amp_echelle1 = []

    for i, j, k, w in zip(px_echelle, py_echelle, ref_stack, amp_echelle):
        idx = np.where(i < 0)[0]
        x = np.delete(i, idx)
        y = np.delete(j, idx)
        z = np.delete(k, idx)
        a = np.delete(w, idx)
        idx1 = np.where(x > 2048)[0]
        x1 = np.delete(x, idx1)
        y1 = np.delete(y, idx1)
        z1 = np.delete(z, idx1)
        a1 = np.delete(a, idx1)
        idx2 = np.where(y < 0)[0]
        x2 = np.delete(x1, idx2)
        y2 = np.delete(y1, idx2)
        z2 = np.delete(z1, idx2)
        a2 = np.delete(a1, idx2)
        idx3 = np.where(y > 2048)[0]
        x3 = np.delete(x2, idx3)
        y3 = np.delete(y2, idx3)
        z3 = np.delete(z2, idx3)
        a3 = np.delete(a2, idx3)

        try:
            idx4 = np.random.choice(
                np.arange(0, np.size(x3)),
                size=np.size(x3)-np.random.randint(
                    n_each_echelle-3,
                    n_each_echelle+3,1),
                replace=False
            )

            x4 = np.delete(x3, idx4)
            y4 = np.delete(y3, idx4)
            z4 = np.delete(z3, idx4)
            a4 = np.delete(a3, idx4)

            x_err = np.random.normal(0,  ccd_noise, size=np.size(x4))
            y_err = np.random.normal(0,  ccd_noise, size=np.size(y4))

            x5 = x_err + x4
            y5 = y_err + y4

            px_echelle1.append(x5)
            px_err_echelle1.append(x_err)
            py_echelle1.append(y5)
            py_err_echelle1.append(y_err)
            wl_echelle1.append(z4)
            amp_echelle1.append(a4)

        except ValueError:

            x_err = np.random.normal(0,  ccd_noise, size=np.size(x3))
            y_err = np.random.normal(0,  ccd_noise, size=np.size(y3))
            x5 = x_err + x3
            y5 = y_err + y3

            px_echelle1.append(x5)
            px_err_echelle1.append(x_err)
            py_echelle1.append(y5)
            py_err_echelle1.append(y_err)
            wl_echelle1.append(z3)
            amp_echelle1.append(a3)

    obs1 = add_outliers(
        n_echelle,
        px_echelle1,
        px_err_echelle1,
        py_echelle1,
        py_err_echelle1,
        wl_echelle1,
        amp_echelle1,
        n_outliers_low,
        n_outliers_high,
        ccd_noise
        )


    obs = (
        obs1[0],
        obs1[1],
        obs1[2],
        obs1[3],
        obs1[4],
        obs1[5]
        )

    return obs




def build_ccd(
        ref,
        ref_size,
        n_echelle,
        n_each_echelle,
        ccd_wl_low,
        ccd_wl_high,
        ccd_noise,
        overlap,
        n_outliers_low,
        n_outliers_high
        ):

    ref_stack, amp_echelle, theta_true, px_echelle,  = create_x_pixels(
        ref,
        ref_size,
        n_echelle,
        ccd_wl_low,
        ccd_wl_high,
        overlap
        )

    py_echelle = create_y_pixels(
        px_echelle,
        n_echelle
        )

    obs = build_obs(
        n_echelle,
        px_echelle,
        py_echelle,
        ref_stack,
        amp_echelle,
        n_each_echelle,
        ccd_noise,
        n_outliers_low,
        n_outliers_high
        )

    return obs, theta_true






def graph_data3d(x_1, y_1, z_1, c_1, ccd_wl_low, ccd_wl_high, colour, file):

    fig = plt.figure()
    ax1 = plt.axes(projection="3d")

    ax1.set_xlabel('X Pixel')
    ax1.set_ylabel('Y Pixel')
    ax1.set_zlabel('Angstrom')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax1.set_xlim(0, 2048)
    ax1.set_ylim(0, 2048)
    ax1.set_zlim(ccd_wl_high, ccd_wl_low)
    p1 = ax1.scatter3D(y_1, x_1, z_1, c=c_1, cmap=colour, s=6)


def graph_data(x_1, y_1, z_1, c_1, ccd_wl_low, ccd_wl_high, colour, file):

    fig = plt.figure(figsize=plt.figaspect(0.4), constrained_layout=False)
    ax1 = fig.add_subplot(222, projection='3d')

    ax1.set_ylabel('X Pixel')
    ax1.set_xlabel('Y Pixel')
    ax1.set_zlabel('Angstrom')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(4))
    mask = z_1 > 1
    ax1.set_xlim(-80, 2100)
    ax1.set_ylim(-80, 2100)
    ax1.set_zlim(ccd_wl_low, ccd_wl_high)
    p1 = ax1.scatter3D(x_1[mask], y_1[mask], z_1[mask], c=c_1[mask], cmap=colour, s=4)

    ax2 = fig.add_subplot(221, aspect='equal')

    ax2.set_xlabel('X Pixel')
    ax2.set_ylabel('Y Pixel')
    ax2.set_xlim(-80, 2100)
    ax2.set_ylim(-80, 2100)
    p2 = ax2.scatter(x_1, y_1, c=c_1, cmap=colour, s=4)

    cbaxes = fig.add_axes([0.7, -0.51, 0.015, 1.3])
    fig.colorbar(p1, ax=ax2, cax = cbaxes, label='Angstrom')


    ax3 = fig.add_subplot(212)
    ax3.set_xlabel('Angstrom')
    ax3.set_ylabel("X Pixel")
    ax3.set_xlim(np.max(z_1), ccd_wl_low-10)
    ax3.set_ylim(-80, 2100)

    p3 = ax3.scatter(z_1, x_1, c=c_1, cmap=colour, s=4)


    plt.subplots_adjust(top=0.8, bottom=-0.5, left=0, right=0.65, hspace=0.25,
                    wspace=0.1)


    plt.savefig(file, bbox_inches="tight")
