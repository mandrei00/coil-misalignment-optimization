import csv
import matplotlib.pyplot as plt

import numpy as np

RESULTS = [
    "result/algorithm_1_result.csv",
    "result/algorithm_2_result.csv",
    "result/determine_algorithm_result.csv"
]


def read_result(name_file, name_set):
    with open(name_file, "r") as file:
        data = list(csv.DictReader(file))
        for d in data:
            if "name_test" in d and d["name_test"] == name_set:
                ret = d
    return ret


def plot_diff(x, ys, labels, x_label, y_label, y_max=None, y_min=None, title=None):
    if x_label is not None and y_label is not None:
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    if y_max is not None and y_min is not None:
        plt.plot(x, y_max * np.ones(x.shape), "k--", )
        plt.plot(x, y_min * np.ones(x.shape), "k--", )

    for ind in range(len(labels)):
        plt.plot(x, ys[ind], label=labels[ind])

    plt.grid()
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
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
        if "p_l" in res and "m" in res and "name_algorithm" in res:
            labels.append(res["name_algorithm"])

            p_l = res["p_l"].replace("\n", "").replace("[", "").replace("]", "")
            p_l = np.fromstring(p_l, sep=" ", dtype=float)
            power.append(p_l)
            min_max_p.append(float(res["p_min"]))
            min_max_p.append(float(res["p_max"]))

            m = res["m"].replace("\n", "").replace("[", "").replace("]", "")
            m = np.fromstring(m, sep=" ", dtype=float)
            mutual_inductance.append(m)
            min_max_m.append(float(res["m_min"]))
            min_max_m.append(float(res["m_max"]))

            po = np.linspace(float(res["po_min"]), float(res["po_max"]), len(p_l))

    plot_diff(
        x=po, ys=mutual_inductance, labels=labels,
        y_min=min_max_m[0], y_max=min_max_m[1],
        x_label="po, м", y_label="M, Гн",
        title="Сравнение распределения взаимной индуктивности\n для оптимизированных геометрий"
    )

    plot_diff(
        x=po, ys=power, labels=labels,
        y_min=min_max_p[0], y_max=min_max_p[1],
        x_label="po, м", y_label="P, Вт",
        title="Сравнение распределения выходной мощности\n для оптимизированных геометрий"
    )


if __name__ == "__main__":
    main()
