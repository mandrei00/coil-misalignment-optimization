import csv
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

import numpy as np

RESULTS = [
    # "result/algorithm_1_result.csv",
    # "result/algorithm_2_result.csv",
    "result/algorithm_3_result.csv",
    "result/algorithm_4_result.csv",
    "result/deterministic_algorithm_result.csv"
]


def read_result(name_file, name_set):
    with open(name_file, "r") as file:
        data = list(csv.DictReader(file))
        for d in data:
            if "test_name" in d and d["test_name"] == name_set:
                ret = d
    return ret


def plot_diff(x, ys, x_label, y_label, labels=None, y_max=None, y_min=None, title=None):

    if x_label is not None and y_label is not None:
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    # draw a boundary line
    if y_max is not None and y_min is not None:
        plt.plot(x, y_max * np.ones(x.shape), "k--", )
        plt.plot(x, y_min * np.ones(x.shape), "k--", )

    if labels is not None:
        # draw each function graph
        for ind in range(len(labels)):
            plt.plot(x, ys[ind], label=labels[ind])
            plt.legend(loc="best")
    else:
        # draw each function graph
        for ind in range(len(ys)):
            plt.plot(x, ys[ind])

    plt.grid()
    if title is not None:
        plt.title(title)

    plt.show()


def plot_coil(coil, title=None):
    if title is not None:
        plt.title(title)
    for r_in in coil:
        plt.gca().add_artist(ptc.Circle((0, 0), radius=r_in, fill=False))
    plt.xlim([-coil[-1] - 0.01, coil[-1] + 0.01])
    plt.ylim([-coil[-1] - 0.01, coil[-1] + 0.01])
    plt.show()


def main():
    diff_results = []
    name_set = "test1"

    for res in RESULTS:
        diff_results.append(read_result(
            name_file=res, name_set=name_set
        ))

    labels = []
    power = []
    mutual_inductance = []
    po = []
    min_max_m = []
    min_max_p = []
    for res in diff_results:
        if "p_l" in res and "m" in res and "algorithm_name" in res:
            labels.append(res["algorithm_name"])

            p_l = res["p_l"].replace("\n", "").replace("[", "").replace("]", "")
            p_l = np.fromstring(p_l, sep=" ", dtype=float)
            power.append(p_l)
            min_max_p.append(float(res["p_min"]))
            min_max_p.append(float(res["p_max"]))

            # plot transmitting coil
            coil_t = res["coil_t"]
            if "[" in coil_t:
                coil_t = coil_t[coil_t.find("[") + 1:coil_t.find("]")]
                coil_t = np.fromstring(coil_t, sep=", ", dtype=float)
            else:
                coil_t = coil_t.replace("(", "").replace(")", "")
                coil_t = np.fromstring(coil_t, sep=", ", dtype=float)
                coil_t = np.linspace(coil_t[0], coil_t[1], int(coil_t[2]))
            plot_coil(coil=coil_t, title="Передающая катушка")

            # plot coil receiving
            coil_t = res["coil_r"]
            if "[" in coil_t:
                coil_t = coil_t[coil_t.find("[") + 1:coil_t.find("]")]
                coil_t = np.fromstring(coil_t, sep=", ", dtype=float)
            else:
                coil_t = coil_t.replace("(", "").replace(")", "")
                coil_t = np.fromstring(coil_t, sep=", ", dtype=float)
                coil_t = np.linspace(coil_t[0], coil_t[1], int(coil_t[2]))
            plot_coil(coil=coil_t, title="Принимающая катушка")

            m = res["m"].replace("\n", "").replace("[", "").replace("]", "")
            m = np.fromstring(m, sep=" ", dtype=float)
            mutual_inductance.append(m)
            min_max_m.append(float(res["m_min"]))
            min_max_m.append(float(res["m_max"]))

            po = np.linspace(float(res["po_min"]), float(res["po_max"]), len(p_l))

    plot_diff(
        x=po, ys=mutual_inductance,
        # labels=labels,
        y_min=min_max_m[0], y_max=min_max_m[1],
        x_label="po, м", y_label="M, Гн",
        title="Сравнение распределения взаимной индуктивности\n для оптимизированных геометрий"
    )

    plot_diff(
        x=po, ys=power,
        # labels=labels,
        y_min=min_max_p[0], y_max=min_max_p[1],
        x_label="po, м", y_label="P, Вт",
        title="Сравнение распределения выходной мощности\n для оптимизированных геометрий"
    )


if __name__ == "__main__":
    main()
