from deterministic_algorithm import *

NAME_ALGORITHM = "Алгоритм 3 (изменяется кол-во витков, внешний и внутренний радиусы)"



def hill_climbing(coil_t, coil_r, distance, min_max_m):
    print("Running hill climbing algorithm...")

    r_in_t = coil_t[0]
    r_out_t = coil_t[1]
    n_t = coil_t[2]

    r_in_r = coil_r[0]
    k_r = coil_r[2]

    # mutate the transmitting coil
    r_out_tq = mutation_lb(
        start=coil_t[0] + 2 * coil_t[3], finish=coil_t[1] - 2 * coil_t[3]
    )
    r_in_tq = mutation_lb(
        start=1e-3, finish=r_out_tq - 2 * coil_t[3]
    )
    n_t_max = int((r_out_tq - r_in_tq) / (2 * coil_t[3]))
    n_tq = np.random.randint(2, 3 if n_t_max <= 2 else n_t_max)

    # mutate the receiving coil
    r_in_rq = mutation_lb(
        start=1e-3, finish=coil_r[1] - 2 * coil_t[3]
    )
    k_r_max = int((r_in_rq - coil_r[0]) / (2 * coil_r[3]) + 1)
    k_rq = np.random.randint(2, 3 if k_r_max <= 2 else k_r_max)

    m_q = mutual_inductance(
        coil_1=np.linspace(r_in_tq, r_out_tq, n_tq),
        coil_2=np.linspace(r_in_rq, coil_r[1], k_rq),
        d=distance[0], po=distance[1], fi=distance[2]
    )

    iterations = 1000
    for _ in range(iterations):

        if np.max(m_q) < min_max_m[1] and np.min(m_q) > min_max_m[0]:
            r_out_t = r_out_tq
            r_in_t = r_in_tq
            n_t = n_tq

            r_in_r = r_in_rq
            k_r = k_rq

            print(f"Find best combination n_t={n_t} r_in_t={r_in_t} r_out_t={r_out_t} r_in_r={r_in_r} k_r={k_r}")
            break

        # mutate the transmitting coil
        r_out_tq = mutation_lb(
            start=coil_t[0] + 2 * coil_t[3], finish=coil_t[1] - 2 * coil_t[3]
        )
        r_in_tq = mutation_lb(
            start=1e-3, finish=r_out_tq - 2 * coil_t[3]
        )
        n_t_max = int((r_out_tq - r_in_tq) / (2 * coil_t[3]))
        n_tq = np.random.randint(2, 3 if n_t_max <= 2 else n_t_max)

        # mutate the receiving coil
        r_in_rq = mutation_lb(
            start=1e-3, finish=coil_r[1] - 2 * coil_t[3]
        )
        k_r_max = int((r_in_rq - coil_r[0]) / (2 * coil_r[3]) + 1)
        k_rq = np.random.randint(2, 3 if k_r_max <= 2 else k_r_max)

        m_q = mutual_inductance(
            coil_1=np.linspace(r_in_tq, r_out_tq, n_tq),
            coil_2=np.linspace(r_in_rq, coil_r[1], k_rq),
            d=distance[0], po=distance[1], fi=distance[2]
        )

    if np.max(m_q) < min_max_m[1] and np.min(m_q) > min_max_m[0]:
        r_out_t = r_out_tq
        r_in_t = r_in_tq
        n_t = n_tq

        r_in_r = r_in_rq
        k_r = k_rq

        print(f"Find best combination n_t={n_t} r_in_t={r_in_t} r_out_t={r_out_t} r_in_r={r_in_r} k_r={k_r}")

    return r_in_t, r_out_t, n_t, r_in_r, k_r


def stochastic_optimization_algorithm_2(**kwargs):
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

    r_in_t, r_out_t, n_t, r_in_r, k_r = hill_climbing(
        coil_t=(r_in_t, r_out_t_max, n_t, r_turn),
        coil_r=(r_in_r, r_out_r, k_r, r_turn),
        distance=(d, po, fi),
        min_max_m=(m_min, m_max)
    )

    # m = mutual_inductance(
    #     coil_1=np.linspace(r_in_r, r_out_r, k_r),
    #     coil_2=np.linspace(r_in_t, r_out_t, n_t),
    #     d=d, po=po, fi=fi
    # )
    # if np.max(m) < m_max and np.min(m) > m_min:
    #     r_out_t, n_t, k_r = hill_climbing(
    #         coil_t=(r_in_t, r_out_t_max, n_t, r_turn),
    #         coil_r=(r_in_r, r_out_r, k_r, r_turn),
    #         distance=(d, po, fi),
    #         min_max_m=(m_min, m_max)
    #     )
    # else:
    #     print("Skip hill climbing...")

    # Step 12. Recalculation of L_t, L_r and C_t, C_r
    print("Running step 12.")
    l_t = self_inductance_coil(np.linspace(r_in_t, r_out_t, n_t), r_turn)
    c_t = 1 / (w ** 2 * l_t)

    l_r = self_inductance_coil(np.linspace(r_in_r, r_out_r, k_r), r_turn)
    c_r = 1 / (w ** 2 * l_r)

    m = mutual_inductance(
        coil_1=np.linspace(r_in_r, r_out_r, k_r),
        coil_2=np.linspace(r_in_t, r_out_t, n_t),
        d=d, po=po, fi=fi
    )

    print(f"Transmitting part:\n r in_t={r_in_t * 1e3} мм"
          f"                  \n r out_t={r_out_t * 1e3} мм"
          f"                  \n Nt={n_t}",
          f"                  \n Lt={l_t * 1e6} мкГн",
          f"                  \n Ct={c_t * 1e9} нФ\n")

    print(f"Receiving part:\n r in_r={r_in_r * 1e3} мм"
          f"               \n r out_r={r_out_r * 1e3} мм"
          f"               \n Kr={k_r}"
          f"               \n Lr={l_r * 1e6} мкГн",
          f"               \n Cr={c_r * 1e9} нФ\n")

    debug(x=po, y_max=m_max, y_min=m_min,
          y=m[0, :, 0], title="Взаимная индуктивность")

    dm_req = np.round((m_max - m_min) / m_max * 100, 3)
    print(f"Permissible difference in mutual inductance: dM="
          f"{dm_req} %")

    dm = np.round((np.max(m) - np.min(m)) / np.max(m) * 100, 3)
    print(f"The resulting difference in mutual inductance: dM="
          f"{dm} %")

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
    p_l = (w ** 2) * (m ** 2) * (vs ** 2) * r_l / (np.abs(z_t * z_r) + (w ** 2) * (m ** 2)) ** 2

    debug(x=m[0, :, 0], y_max=p_max, y_min=p_min,
          y=p_l[0, :, 0], title="Выходная мощность",
          x_label="M, Гн", y_label="P, Вт")

    dpl_req = np.round((p_max - p_min) / p_max * 100, 3)
    print(f"Permissible difference in mutual inductance: dP="
          f"{dpl_req} %")

    dpl = np.round((np.max(p_l) - np.min(p_l)) / np.max(p_l) * 100, 3)
    print(f"The resulting difference in mutual inductance: dP="
          f"{dpl} %")

    result = {"test_name": kwargs["test_name"], "algorithm_name": NAME_ALGORITHM,
              "power": p, "n": n, "f": f,
              "r_l": r_l, "r_t": r_t, "r_r": r_r,
              "r_turn": r_turn,
              "coil_t": (r_in_t, r_out_t, n_t), "l_t": l_t * 1e6, "c_t": c_t * 1e9,
              "coil_r": (r_in_r, r_out_r, k_r), "l_r": l_r * 1e6, "c_r": c_r * 1e9,
              "m_min": m_min, "m_max": m_max, "dm_req": dm_req,
              "p_min": p_min, "p_max": p_max, "dpl_req": dpl_req,
              "m": m[0, :, 0], "dm": dm,
              "p_l": p_l[0, :, 0], "dpl": dpl,
              "d_min": d_min, "d_max": d_max,
              "po_min": po_min, "po_max": po_max,
              "fi_min": fi_min, "fi_max": fi_max
              }
    return result


def main():

    dataset = "../" + DATASET

    # an array of geometry optimization results for each test
    res = []
    for data in read(dataset):
        if data["test_name"] == "test1":
            res.append(stochastic_optimization_algorithm_2(**data))
            break

    # save result of geometry optimization for each test
    result = f"../result/algorithm_3_result.csv"
    write(result, res)


if __name__ == "__main__":
    main()
