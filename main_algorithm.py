from main_functions import *

import numpy as np
import csv

DATASET = "datasets/dataset.csv"


def read(csv_filename):
    with open(csv_filename, "r") as file:
        data = list(csv.DictReader(file))
    return data


def coil_optimization_algorithm(**kwargs):
    """
    :param kwargs:
    :return:
    """
    # get parameters of output power and its differential
    p = float(kwargs["power"])
    n = float(kwargs["n"])

    # get parameter of frequency
    f = float(kwargs["f"])

    # get capacity parameters of the transmitting and receiving parts
    c_t, c_r = float(kwargs["c_t"]), float(kwargs["c_r"])

    # get the resistance parameters of parts of the electrical circuit
    r_l, r_t, r_r = float(kwargs["r_l"]), float(kwargs["r_t"]), float(kwargs["r_r"])

    # get the geometric characteristics of the coils
    r_turn = float(kwargs["r_turn"])
    r_out_r = float(kwargs["r_out_r"])

    # get axial distance
    d_min, d_max = float(kwargs["d_min"]), float(kwargs["d_max"])
    d = np.linspace(d_min, d_max, 1) if d_min == d_max else np.linspace(d_min, d_max, 35)

    # get lateral misalignment
    po_min, po_max = float(kwargs["po_min"]), float(kwargs["po_max"])
    po = np.linspace(po_min, po_max, 1) if po_min == po_max else np.linspace(po_min, po_max, 35)

    # get angular misalignment
    fi_min, fi_max = float(kwargs["fi_min"]), float(kwargs["fi_max"])
    fi = np.linspace(fi_min, fi_max, 1) if po_min == po_max else np.linspace(fi_min, fi_max, 35)
    fi = np.radians(fi)

    # Step 1. Assignment of the initial values.
    w = 2 * np.pi * f
    p_max, p_min = p * (1 + n), p * (1 - n)

    # Step 2. Calculation of Lt and Lr.
    l_t, l_r = 1 / (c_t * w ** 2), 1 / (c_r * w ** 2)

    # Step 3. Calculation of N and K.
    r_in_t = r_out_t = r_in_r = r_out_r
    n_t = int(np.ceil(np.sqrt(l_t / self_inductance_turn(r_out_t, r_turn))))
    k_r = int(np.ceil(np.sqrt(l_r / self_inductance_turn(r_out_r, r_turn))))

    # Step 4. Calculation of Vs.
    # calculation quality factor
    q_t = quality_factor(r_t, l_t, c_t)
    q_r = quality_factor(r_r + r_l, l_r, c_r)
    k_crit = 1 / np.sqrt(q_t * q_r)

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r

    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = np.abs((a + b) * np.sqrt(p_max / r_l))

    # Step 5. Calculation of Mmin and Mmax.
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = np.abs((vs * np.sqrt(r_l) + a)) / (2 * w * np.sqrt(p_min))
    m_min = np.abs((vs * np.sqrt(r_l) - a)) / (2 * w * np.sqrt(p_min))

    # Step 6. Calculation of R_in_t and R_in_r.
    r_in_t = r_in_r = calculation_r_in(coil_t=(r_in_t, r_out_t, n_t),
                                       coil_r=(r_in_r, r_out_r, k_r),
                                       distance=(d, po, fi),
                                       range_m=(m_min, m_max))

    # Step 7. Calculation of Mcalcmin and Mcalcmax
    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        d=d, po=po, fi=fi
    )

    kof = 0
    while np.min(m) < m_min or np.max(m) > m_max and kof == 0:

        # Step 8. Calculation of R outT max.
        kof, r_out_t_max = calculation_r_out_t_max(coil_t=(r_in_t, r_out_t, n_t),
                                                   coil_r=(r_in_r, r_out_r, k_r),
                                                   distance=(d, po, fi),
                                                   range_m=(m_min, m_max))
        while kof == 1:
            n_t += 1

            kof, r_out_t_max = calculation_r_out_t_max(coil_t=(r_in_t, r_out_t, n_t),
                                                       coil_r=(r_in_r, r_out_r, k_r),
                                                       distance=(d, po, fi),
                                                       range_m=(m_min, m_max))

        # Step 9. Calculation of R outT.




def main():
    for data in read(DATASET):
        print(data)
        coil_optimization_algorithm(**data)


if __name__ == "__main__":
    main()

