from operator import itemgetter
from deterministic_algorithm import *

import numpy as np

from sahc.sahc import fitness_func


NAME_ALGORITHM = "AHC"


def adaptive_hill_climbing(
        coil_t,
        coil_r,
        distance,
        min_max_p,
        system_param
):
    # unpacking system parameters for calculate fitness function
    w, r_l, r_t, r_r = system_param

    p_min, p_max = min_max_p[0], min_max_p[1]

    # unpacking transmit coil variables
    r_in_t, r_out_t, n_t, r_turn_t = coil_t

    # unpacking receive coil variables
    r_in_r, r_out_r, k_r, r_turn_r = coil_r

    po_0, p_l, p = fitness_func(
        w=w,
        r_l=r_l, r_t=r_t, r_r=r_r,
        p_max=p_max,
        coil_t=(r_in_t, r_out_t, n_t, r_turn_t),
        coil_r=(r_in_r, r_out_r, k_r, r_turn_r),
        distance=distance
    )

    po_max = distance[1][-1]
    i_useless = 0

    # probability of mutation
    p_m = 0.9

    flag = False

    while i_useless <= 50 and po_0 != po_max:
        r_out_tq = r_out_t
        r_in_tq = r_in_t
        n_tq = n_t
        r_in_rq = r_in_r
        k_rq = k_r

        # mutate r_out_t
        if np.random.random() < p_m:
            r_out_tq = np.random.uniform(r_turn_t * (n_t + 4), 2 * r_out_r)

        # mutate r_in_t
        if np.random.random() < p_m:
            r_in_tq = np.random.uniform(4 * r_turn_t, r_out_t - r_turn_t * (n_t - 1))

        # mutate n_t
        if np.random.random() < p_m:
            n_tq = np.random.randint(1, (r_out_t - r_in_t) // r_turn_t + 1)

        # mutate r_in_t
        if np.random.random() < p_m:
            r_in_rq = np.random.uniform(0.001, 0.5 * r_out_r)

        # mutate k_r
        if np.random.random() < p_m:
            k_r_max = (r_out_t - r_in_t) // (2 * r_turn_r)
            k_rq = np.random.randint(1, k_r_max)

        # calculate fitness function
        po_q, p_l, p = fitness_func(
            w=w,
            r_l=r_l, r_t=r_t, r_r=r_r,
            p_max=p_max,
            coil_t=(r_in_tq, r_out_tq, n_tq, r_turn_t),
            coil_r=(r_in_rq, r_out_r, k_rq, r_turn_r),
            distance=distance
        )

        if p_min < np.min(p) and np.max(p) < p_max:

            if po_q > po_0:
                flag = True  # there was at least one mutation
                i_useless = 0 # update count useless operation
                print("Find better value...")
                # save parameter
                r_out_t = r_out_tq
                r_in_t = r_in_tq
                n_t = n_tq
                r_in_r = r_in_rq
                k_r = k_rq

                # update the distance at which the maximum power falls
                po_0 = po_q
            else:
                i_useless += 1
                if i_useless % 10 == 0:
                    print(f"i_useless = {i_useless}")
        else:
            i_useless += 1
            if i_useless % 10 == 0:
                print(f"i_useless = {i_useless}")

    coil_t = (r_in_t, r_out_t, n_t, r_turn_t)
    coil_r = (r_in_r, r_out_r, k_r, r_turn_r)

    return flag, coil_t, coil_r


def geometric_optimization_algorithm(**kwargs):
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
    fi = np.linspace(fi_min, fi_max, 1) if fi_min == fi_max else np.linspace(fi_min, fi_max, 35)
    fi = np.radians(fi)

    # Step 1. Assignment of the initial values.
    print("Running step 1.")
    w = 2 * np.pi * f
    p_max, p_min = p * (1 + n), p * (1 - n)

    # Step 2. Calculation of Lt and Lr.
    print("Running step 2.")
    l_t, l_r = 1 / (c_t * w ** 2), 1 / (c_r * w ** 2)

    # Step 3. Calculation of N and K.
    print("Running step 3.")
    r_in_t = r_out_t = r_in_r = r_out_r
    n_t = int(np.ceil(np.sqrt(l_t / self_inductance_turn(r_out_t, r_turn))))
    k_r = int(np.ceil(np.sqrt(l_r / self_inductance_turn(r_out_r, r_turn))))

    # Step 4. Calculation of Vs.
    # calculation quality factor
    print("Running step 4.")
    q_t = quality_factor(r_t, l_t, c_t)
    q_r = quality_factor(r_r + r_l, l_r, c_r)
    k_crit = 1 / np.sqrt(q_t * q_r)

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = np.abs((a + b) * np.sqrt(p_max / r_l))

    # Step 5. Calculation of Mmin and Mmax.
    print("Running step 5.")
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = np.abs((vs * np.sqrt(r_l) + a)) / (2 * w * np.sqrt(p_min))
    m_min = np.abs((vs * np.sqrt(r_l) - a)) / (2 * w * np.sqrt(p_min))

    # Step 6. Calculation of R_in_t and R_in_r.
    print("Running step 6.")
    kof_2, r_in_r = calculation_r_in(coil_t=(r_in_t, r_out_t, n_t),
                                     coil_r=(r_in_r, r_out_r, k_r),
                                     distance=(d, po, fi),
                                     range_m=(m_min, m_max))
    r_in_t = r_in_r

    # Step 8. Calculation of R outT max.
    print("Running step 8.")
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

    flag, coil_t, coil_r = adaptive_hill_climbing(
        system_param=(w, r_l, r_t, r_r),
        coil_t=(r_in_t, r_out_t, n_t, r_turn),
        coil_r=(r_in_r, r_out_r, k_r, r_turn),
        distance=(d, po, fi),
        min_max_p=(p_min, p_max)
    )

    # Step 12. Recalculation of L_t, L_r and C_t, C_r
    print("Running step 12.")
    coil_t = np.linspace(*coil_t[:-1])
    coil_r = np.linspace(*coil_r[:-1])

    l_t = self_inductance_coil(coil_t, r_turn)
    c_t = 1 / (l_t * w ** 2)
    q_t = quality_factor(r_t, l_t, c_t)
    print(f"Transmitting part:\n r in_t={coil_t[0] * 1e3} мм"
          f"                  \n r out_t={coil_t[-1] * 1e3} мм"
          f"                  \n Nt={len(coil_t)}",
          f"                  \n Lt={l_t * 1e6} мкГн",
          f"                  \n Ct={c_t * 1e9} нФ"
          f"                  \n Qt={q_t}\n")

    l_r = self_inductance_coil(coil_r, r_turn)
    c_r = 1 / (l_r * w ** 2)
    q_r = quality_factor(r_l + r_r, l_r, c_r)
    print(f"Receiving part:\n r in_r={coil_r[0] * 1e3} мм"
          f"               \n r out_r={coil_r[-1] * 1e3} мм"
          f"               \n Kr={len(coil_r)}"
          f"               \n Lr={l_r * 1e6} мкГн",
          f"               \n Cr={c_r * 1e9} нФ"
          f"               \n Qr={q_r}\n")

    k = coupling_coefficient(
        coil_1=coil_t, r1_turn=r_turn,
        coil_2=coil_r, r2_turn=r_turn,
        d=d, po=po, fi=fi
    )

    # show plot coupling coefficient
    # debug(x=po, y=k[0, :, 0],
    #       y_label="k",
    #       title="Коэффициент связи")

    dk = np.round((np.max(k) - np.min(k)) / np.max(k) * 100, 3)
    print(f"The resulting difference in coupling coefficient: dk="
          f"{dk} %\n")

    m = mutual_inductance(
        coil_1=coil_t,
        coil_2=coil_r,
        d=d, po=po, fi=fi
    )

    dm_req = np.round((m_max - m_min) / m_max * 100, 3)
    print(f"Permissible difference in mutual inductance: dM="
          f"{dm_req} %")

    dm = np.round((np.max(m) - np.min(m)) / np.max(m) * 100, 3)
    print(f"The resulting difference in mutual inductance: dM="
          f"{dm} %\n")

    # calculation power transfer efficiency
    efficiency = ((w ** 2) * (m ** 2) * r_l) / ((r_l + r_r) * (r_t * (r_r + r_l) + (w ** 2) * (m ** 2)))
    # show plot power transfer efficiency
    debug(x=po, y_label="η",
          y=efficiency[0, :, 0], title="Эффективность передачи энергии")

    # show plot mutual inductance
    # debug(x=po, y_max=m_max, y_min=m_min,
    #       y=m[0, :, 0], title="Взаимная индуктивность")

    k_crit = 1 / np.sqrt(q_t * q_r)
    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = np.abs((a + b) * np.sqrt(p_max / r_l))
    p_l = (w ** 2) * (m ** 2) * (vs ** 2) * r_l / (np.abs(z_t * z_r) + (w ** 2) * (m ** 2)) ** 2

    dpl_req = np.round((p_max - p_min) / p_max * 100, 3)
    print(f"Permissible difference in mutual inductance: dP="
          f"{dpl_req} %")

    dpl = np.round((np.max(p_l) - np.min(p_l)) / np.max(p_l) * 100, 3)
    print(f"The resulting difference in mutual inductance: dP="
          f"{dpl} %\n")

    # show plot output power
    debug(x=po, y_max=p_max, y_min=p_min,
          y=p_l[0, :, 0], title="Выходная мощность",
          x_label="po, м", y_label="P, Вт")

    result = {
        "result": flag,
        "test_name": kwargs["test_name"], "algorithm_name": NAME_ALGORITHM,
        "power": p, "n": n, "f": f,
        "r_l": r_l, "r_t": r_t, "r_r": r_r,
        "r_turn": r_turn,
        "coil_t": list(coil_t), "l_t": l_t * 1e6, "c_t": c_t * 1e9, "q_t": q_t,
        "coil_r": list(coil_r), "l_r": l_r * 1e6, "c_r": c_r * 1e9, "q_r": q_r,
        "m_min": m_min, "m_max": m_max, "dm_req": dm_req,
        "p_min": p_min, "p_max": p_max, "dpl_req": dpl_req,
        "k": list(k[0, :, 0]), "dk": dk,
        "m": list(m[0, :, 0]), "dm": dm,
        "efficiency": list(efficiency[0, :, 0]),
        "p_l": list(p_l[0, :, 0]), "dpl": dpl,
        "d_min": d_min, "d_max": d_max,
        "po_min": po_min, "po_max": po_max,
        "fi_min": fi_min, "fi_max": fi_max
    }
    return result


def run_idle(test_name):
    dataset = "../" + DATASET

    for data in read(dataset):
        if data["test_name"] == test_name:
            geometric_optimization_algorithm(**data)
            break


def main():
    run_idle(
        test_name="test1"
    )


if __name__ == "__main__":
    main()
