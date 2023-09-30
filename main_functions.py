import numpy as np


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