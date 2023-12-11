import csv
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import pandas as pd

import numpy as np

RESULTS = [
    # "result/algorithm_1_result.csv",
    # "result/algorithm_2_result.csv",
    # "result/algorithm_3_result.csv",
    # "result/algorithm_4_result.csv",
    "result/sahc.csv",
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


def plot_coil(coil, title=None, fig_name=None, show=False):
    if title is not None:
        plt.title(title)

    for r_in in coil:
        plt.gca().add_artist(ptc.Circle((0, 0), radius=r_in, fill=False))

    plt.xlim([-coil[-1], coil[-1]])
    plt.ylim([-coil[-1], coil[-1]])

    plt.axis("off")

    if fig_name is not None:
        plt.savefig(f"{fig_name}.svg")

    plt.show()


def show_table_system(name_csv_file, indices, name_excel_file=None):

    if indices is None:
        indices = []

    info = pd.read_csv(
        name_csv_file
    )

    for c in ["coil_t", "coil_r"]:
        coil_indices = ["r_out", "r_in", "n"]
        data_coils = []
        if c in indices:
            indices_tabel = indices.copy()
            indices_tabel.remove(c)
            coil_indices = [name_ind + c[-2:] for name_ind in coil_indices]

            for val in info[c]:

                columns_coil = dict.fromkeys(coil_indices, None)

                coil = np.fromstring(val[1:-1], sep=", ", dtype=float)

                columns_coil[coil_indices[0]] = np.round(np.max(coil) * 1000, 3)
                columns_coil[coil_indices[1]] = np.round(np.min(coil) * 1000, 3)
                columns_coil[coil_indices[2]] = len(coil)

                data_coils.append(columns_coil.copy())
            break

    table_coils = pd.DataFrame(data_coils)
    info_table = pd.concat([info[indices_tabel], table_coils], sort=False, axis=1)

    if name_excel_file is not None:
        info_table.to_excel(name_excel_file)

    # print(f"table name: {name_csv_file}")
    # print(info[indices])


def main():
    diff_results = []
    name_set = "test1"

    for res in RESULTS:
        diff_results.append(read_result(
            name_file=res, name_set=name_set
        ))

        # show_table_system(
        #     name_csv_file=res,
        #     indices=["c_t", "l_t", "q_t", "coil_t"],
        #     name_excel_file=rf"result_table/{name_set}/" + diff_results[-1]["algorithm_name"] + "_transmit" + ".xlsx"
        # )
        #
        # show_table_system(
        #     name_csv_file=res,
        #     indices=["c_r", "l_r", "q_r", "coil_r"],
        #     name_excel_file=rf"result_table/{name_set}/" + diff_results[-1]["algorithm_name"]+ "_receive" + ".xlsx"
        # )

    labels = []

    power = []
    min_max_p = []

    mutual_inductance = []
    min_max_m = []

    couple_coefficient = []
    po = []

    for res in diff_results:
        if "p_l" in res and "m" in res and "algorithm_name" in res:
            # get name of algorithms
            labels.append(res["algorithm_name"])

            # get power
            power.append(np.fromstring(res["p_l"][1:-1], sep=", ", dtype=float))
            # get range of power
            min_max_p.append(float(res["p_min"]))
            min_max_p.append(float(res["p_max"]))

            # get mutual inductance
            mutual_inductance.append(np.fromstring(res["m"][1:-1], sep=", ", dtype=float))
            # get range of mutual inductance
            min_max_m.append(float(res["m_min"]))
            min_max_m.append(float(res["m_max"]))

            # get coupling coefficient
            couple_coefficient.append(np.fromstring(res["k"][1:-1], sep=", ", dtype=float))

            # get lateral misalignment
            po = np.linspace(float(res["po_min"]), float(res["po_max"]), len(power[-1]))

            # # plot transmitting coil
            # coil_t = np.fromstring(res["coil_t"][1:-1], sep=", ", dtype=float)
            # plot_coil(
            #     coil=coil_t,
            #     # title="Передающая катушка",
            #     fig_name=rf"graphics\{labels[-1]}_transmitting_coil"
            # )
            # # plot coil receiving
            # coil_r = np.fromstring(res["coil_r"][1:-1], sep=", ", dtype=float)
            # plot_coil(
            #     coil=coil_r,
            #     # title="Принимающая катушка",
            #     fig_name=rf"graphics\{labels[-1]}_receiving_coil"
            # )

    plot_diff(
        x=po * 1e3, ys=mutual_inductance,
        # labels=labels,
        y_min=min_max_m[0], y_max=min_max_m[1],
        x_label="ρ, мм", y_label="M, Гн",
        title="Сравнение распределения взаимной индуктивности\n для оптимизированных геометрий"
    )

    plot_diff(
        x=po * 1e3, ys=power,
        # labels=labels,
        y_min=min_max_p[0], y_max=min_max_p[1],
        x_label="ρ, мм", y_label="P, Вт",
        # title="Сравнение распределения выходной мощности\n для оптимизированных геометрий"
    )

    plot_diff(
        x=po * 1e3, ys=couple_coefficient,
        # labels=labels,
        x_label="ρ, мм", y_label="k",
        title="Сравнение распределения коэффициента связи\n для оптимизированных геометрий"
    )


if __name__ == "__main__":
    main()
