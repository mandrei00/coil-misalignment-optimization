from main_functions import *

import numpy as np
import csv

DATASET = "dataset/dataset.csv"
NAME_ALGORITHM = "Deterministic algorithm"


def read(csv_filename):
    with open(csv_filename, "r") as file:
        data = list(csv.DictReader(file))
    return data


def write(csv_filename, data):
    if data is []:
        return
    try:
        fields_name = list(data[0].keys())
    except Exception as err:
        print(f"Error {err}")
        return
    with open(csv_filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields_name)
        writer.writeheader()
        writer.writerows(data)


def coil_optimization_algorithm(**kwargs):
    # initial guess - algorithm found solution
    flag = True

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

    # Step 7. Calculation of Mcalcmin and Mcalcmax.
    print("Running step 7.")
    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        d=d, po=po, fi=fi
    )

    kof = 0
    while np.min(m) < m_min or np.max(m) > m_max and kof == 0:

        # Step 8. Calculation of R outT max.
        print("Running step 8.")
        kof, r_out_t_max = calculation_r_out_t_max(coil_t=(r_in_t, r_out_t, n_t),
                                                   coil_r=(r_in_r, r_out_r, k_r),
                                                   distance=(d, po, fi),
                                                   range_m=(m_min, m_max))

        while kof == 1:
            n_t += 1
            print("Running step 8.")
            kof, r_out_t_max = calculation_r_out_t_max(coil_t=(r_in_t, r_out_t, n_t),
                                                       coil_r=(r_in_r, r_out_r, k_r),
                                                       distance=(d, po, fi),
                                                       range_m=(m_min, m_max))

        # Step 9. Calculation of R outT.
        print("Running step 9.")
        r_out_t = calculation_r_out_t(coil_t=(r_in_t, r_out_t_max, n_t),
                                      coil_r=(r_in_r, r_out_r, k_r),
                                      distance=(d, po, fi),
                                      range_m=(m_min, m_max))

        # Step 10. Recalculation R_in.
        print("Running step 10.")
        kof_2, r_in = calculation_r_in(coil_t=(r_in_t, r_out_t, n_t),
                                       coil_r=(r_in_r, r_out_r, k_r),
                                       distance=(d, po, fi),
                                       range_m=(m_min, m_max))

        r_in_t = r_in_r = r_in

        while kof_2 == 1:
            if k_r >= 2:
                k_r -= 1

                kof_2, r_in_r = calculation_r_in(coil_t=(r_in_t, r_out_t, n_t),
                                                 coil_r=(r_in_r, r_out_r, k_r),
                                                 distance=(d, po, fi),
                                                 range_m=(m_min, m_max))
                r_in_t = r_in_r
            else:
                print("Optimization is not Possible.")
                flag = False
                kof_2 = 0
                kof = 1

        m = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, po=po, fi=fi
        )

    # Step 11. Recalculation of L_t, L_r and C_t, C_r
    print("Running step 11.\n")

    # ToDo: add round for radius coil
    coil_t = np.linspace(r_in_t, r_out_t, n_t)
    coil_r = np.linspace(r_in_r, r_out_r, k_r)

    l_t = self_inductance_coil(coil_t, r_turn)
    c_t = 1 / (w ** 2 * l_t)
    q_t = quality_factor(r_t, l_t, c_t)
    print(f"Transmitting part:\n r in_t={coil_t[0] * 1e3} мм"
          f"                  \n r out_t={coil_t[-1] * 1e3} мм"
          f"                  \n Nt={len(coil_t)}",
          f"                  \n Lt={l_t * 1e6} мкГн",
          f"                  \n Ct={c_t * 1e9} нФ"
          f"                  \n Qt={q_t}\n")

    l_r = self_inductance_coil(coil_r, r_turn)
    c_r = 1 / (w ** 2 * l_r)
    q_r = quality_factor(r_t + r_r, l_r, c_r)
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
    debug(x=po, y=k[0, :, 0],
          y_label="k",
          title="Коэффициент связи")

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

    # show plot mutual inductance
    debug(x=po, y_max=m_max, y_min=m_min,
          y=m[0, :, 0], title="Взаимная индуктивность")

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
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
          x_label="M, Гн", y_label="P, Вт")

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
        "p_l": list(p_l[0, :, 0]), "dpl": dpl,
        "d_min": d_min, "d_max": d_max,
        "po_min": po_min, "po_max": po_max,
        "fi_min": fi_min, "fi_max": fi_max
    }
    return result


def run_all_test():
    res = []

    for data in read(DATASET):
        print("Running test " + data["test_name"])
        res.append(coil_optimization_algorithm(**data))

    # save result of geometry optimization for each test
    result = "result/deterministic_algorithm_result.csv"
    write(result, res)


def run_test(test_name):
    # an array of geometry optimization results for each test
    res = []
    for data in read(DATASET):
        if data["test_name"] == test_name:
            res.append(coil_optimization_algorithm(**data))

    # save result of geometry optimization for each test
    result = "result/deterministic_algorithm_result.csv"
    write(result, res)


def main():
    run_all_test()
    # run_test(
    #     test_name="test1"
    # )


if __name__ == "__main__":
    main()
