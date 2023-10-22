import numpy as np
import matplotlib.pyplot as plt


def quality_factor(r, l, c):
    q = 1 / r * np.sqrt(l / c)
    return q


def mutual_inductance(coil_1, coil_2, d, po, fi, N=60, K=60):
    """
    Function for calculating mutual inductance between two flat inductors.
    :param coil_1:
    :param coil_2:
    :param d:
    :param po:
    :param fi:
    :param N:
    :param K:
    :return:
    """
    # vacuum permeability
    mu0 = 4 * np.pi * 10 ** (-7)
    n = np.arange(N)
    k = n.reshape((K, 1))
    df1 = 2 * np.pi / N
    df2 = 2 * np.pi / K

    mi = np.zeros((d.shape[0], po.shape[0], fi.shape[0]))

    for ind_d in range(len(d)):
        for ind_p in range(len(po)):
            for ind_f in range(len(fi)):

                mi_turns = np.zeros((coil_1.shape[0], coil_2.shape[0]))
                for ri in range(coil_1.shape[0]):
                    for rj in range(coil_2.shape[0]):
                        m = 0
                        xk_xn = po[ind_p] + coil_1[ri] * np.cos(df2 * k) * np.cos(fi[ind_f]) - coil_2[rj] * np.cos(df1 * n)
                        yk_yn = coil_1[ri] * np.sin(df2 * k) * np.cos(fi[ind_f]) - coil_2[rj] * np.sin(df1 * n)
                        zk_zn = d[ind_d] + coil_1[ri] * np.cos(df2 * k) * np.sin(fi[ind_f])
                        r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5
                        m += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
                        m *= mu0 * coil_1[ri] * coil_2[rj] / (4 * np.pi)
                        mi_turns[ri][rj] = np.sum(m)
                mi[ind_d][ind_p][ind_f] = np.sum(mi_turns)

    return mi


def self_inductance_turn(r, r_turn):
    """
    Function for calculating the self-inductance of a coil turn.
    :param r:
    :param r_turn:
    :return:
    """
    # vacuum permeability
    mu0 = 4 * np.pi * 10 ** (-7)
    a = np.log(8 * r / r_turn)
    b = r_turn ** 2 / (8 * r ** 2)
    return mu0 * r * (a - 7 / 4 + b * (a + 1 / 3))


def self_inductance_coil(coil, r_turn, N=60, K=60):
    """
    Function for calculating the coil's own inductance.
    :param coil:
    :param r_turn:
    :param N:
    :param K:
    :return:
    """
    # vacuum permeability
    mu0 = 4 * np.pi * 10 ** (-7)
    l = np.sum(self_inductance_turn(r=coil, r_turn=r_turn))
    d, ro, fi = 0, 0, 0
    n = np.arange(N)
    k = n.reshape((K, 1))
    df1 = 2 * np.pi / N
    df2 = 2 * np.pi / K
    mutual_inductance = 0
    for ri in range(len(coil)):
        for rj in range(len(coil)):
            if ri != rj:
                M = 0

                xk_xn = ro + coil[ri] * np.cos(df2 * k) * np.cos(fi) - coil[rj] * np.cos(df1 * n)
                yk_yn = coil[ri] * np.sin(df2 * k) * np.cos(fi) - coil[rj] * np.sin(df1 * n)
                zk_zn = d + coil[ri] * np.cos(df2 * k) * np.sin(fi)

                r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5

                M += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
                M *= mu0 * coil[ri] * coil[rj] / (4 * np.pi)
                mutual_inductance += np.sum(M)
    l += mutual_inductance
    return l


def coupling_coefficient(coil_1, r1_turn, coil_2, r2_turn, d, po, fi):
    """
    Function for calculating the coupling coefficient between inductors.
    :param coil_1:
    :param r1_turn:
    :param coil_2:
    :param r2_turn:
    :param d:
    :param po:
    :param fi:
    :return:
    """
    l_1 = self_inductance_coil(coil_1, r1_turn)
    l_2 = self_inductance_coil(coil_2, r2_turn)
    m = mutual_inductance(coil_1, coil_2, d, po, fi)
    return m / np.sqrt(l_1 * l_2)


def calculation_r_in(coil_t, coil_r, distance, range_m):
    start = 1e-3
    finish = coil_r[1] - start * (coil_r[2] - 1)

    pogr = 5e-4
    eps = (start + finish) / 2
    kof = 0
    x1 = start
    x2 = finish

    x0 = (x1 + x2) / 2
    m_x0 = mutual_inductance(
        coil_1=np.linspace(x0, coil_t[1], coil_t[2]),
        coil_2=np.linspace(x0, coil_r[1], coil_r[2]),
        d=distance[0], po=distance[1], fi=distance[2]
    )
    i = 0
    while eps >= pogr or np.max(m_x0) > range_m[1]:

        i += 1
        if np.max(m_x0) > range_m[1]:
            x2 = x0
            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(x0, coil_t[1], coil_t[2]),
                coil_2=np.linspace(x0, coil_r[1], coil_r[2]),
                d=distance[0], po=distance[1], fi=distance[2]
            )
        else:
            x1 = x0
            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(x0, coil_t[1], coil_t[2]),
                coil_2=np.linspace(x0, coil_r[1], coil_r[2]),
                d=distance[0], po=distance[1], fi=distance[2]
            )
        eps = (x2 - x1) / 2
        if i > 15:
            kof = 1
            break

    return kof, x0


def calculation_r_out_t_max(coil_t, coil_r,
                            distance, range_m):
    m = mutual_inductance(
            coil_1=np.linspace(*coil_t),
            coil_2=np.linspace(*coil_r),
            d=distance[0], po=distance[1], fi=distance[2]
    )
    m_prev = m.copy()
    r_out_t = 0
    kof = 0
    i = 0
    while np.min(m) < range_m[0]:
        i += 1
        r_out_t += coil_r[1]
        m = mutual_inductance(
            coil_1=np.linspace(coil_t[0], r_out_t, coil_t[2]),
            coil_2=np.linspace(*coil_r),
            d=distance[0], po=distance[1], fi=distance[2]
        )
        if np.min(m_prev) > np.min(m) and i > 10:
            kof = 1
            break
        m_prev = m.copy()

    return kof, r_out_t


def calculation_r_out_t(coil_t, coil_r,
                        distance, range_m):
    a = coil_r[1]
    b = coil_t[1]
    pogr = 5e-4

    eps = (a + b) / 2

    x1 = a
    x2 = b

    x0 = (x1 + x2) / 2
    m_x0 = mutual_inductance(
        coil_1=np.linspace(coil_t[0], x0, coil_t[2]),
        coil_2=np.linspace(*coil_r),
        d=distance[0], po=distance[1], fi=distance[2]
    )

    while eps >= pogr or np.min(m_x0) < range_m[0]:
        if np.min(m_x0) > range_m[0]:
            x2 = x0

            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(coil_t[0], x0, coil_t[2]),
                coil_2=np.linspace(*coil_r),
                d=distance[0], po=distance[1], fi=distance[2]
            )
        else:
            x1 = x0

            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(coil_t[0], x0, coil_t[2]),
                coil_2=np.linspace(*coil_r),
                d=distance[0], po=distance[1], fi=distance[2]
            )

        eps = (x2 - x1) / 2

    return x0


def debug(x, y, y_max=None, y_min=None, title=None, x_label="ro, м", y_label="M, Гн"):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_max is not None and y_min is not None:
        plt.plot(x, y_max * np.ones(x.shape), "k--", )
        plt.plot(x, y_min * np.ones(x.shape), "k--", )
    if y is not None:
        plt.plot(x, y, label="Оптимизированный случай")
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    plt.show()


def mutation_lb(start, finish, x=None, dr_min=0.001, dr_max=0.025):
    if x is None:
        res = np.random.uniform(start, finish)
    elif np.random.choice([-1, 1]) > 0:
        res = np.random.uniform(low=finish if x+dr_min > finish else x+dr_min,
                                high=finish if x+dr_max > finish else x+dr_max)
    else:
        res = np.random.uniform(low=start if x-dr_min < start else x-dr_min,
                                high=start if x-dr_max < start else x-dr_max)
    return np.round(res, 3)
