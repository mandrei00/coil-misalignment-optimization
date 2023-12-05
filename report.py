import matplotlib.pyplot as plt
import pandas as pd

from compare_result import *

RESULTS = [
    # "result/algorithm_1_result.csv",
    # "result/algorithm_2_result.csv",
    ("result/algorithm_3_result.csv", "blue"),
    ("result/algorithm_4_result.csv", "orange"),
    ("result/deterministic_algorithm_result.csv", "green")
]


def read_result(name_file, name_set):
    with open(name_file, "r") as file:
        data = list(csv.DictReader(file))
        for d in data:
            if "test_name" in d and d["test_name"] == name_set:
                ret = d
    return ret


def get_data_coil(coil):
    """
    Retrieves the number of turns, outer and inner radius of the coil
    from an array presented in string format.
    :param coil: str
    :return: float, float, float
    """
    coil = np.fromstring(coil[1:-1], sep=", ", dtype=float)

    r_out = np.round(np.max(coil) * 1000, 3)
    r_in = np.round(np.min(coil) * 1000, 3)
    n = len(coil)

    return r_in, r_out, n


def make_column(indices, data):
    """
    Processes data from a csv file and makes a table of it.
    :param indices:
    :param data:
    :return:
    """
    test_data = np.array([])
    for ind in indices:

        if ind in data:
            if "coil" in ind:
                r_in, r_out, n = get_data_coil(
                    coil=data[ind]
                )

                test_data = np.append(test_data, r_in)
                test_data = np.append(test_data, r_out)
                test_data = np.append(test_data, n)
            else:
                v = float(data[ind])
                test_data = np.append(test_data, v)

    return test_data.copy()


def test_result_table(name_test, data, transmitting_part=True, receiving_part=True):
    ind_column = []

    table_transmit = np.array([])
    table_receive = np.array([])
    for d in data:

        res = read_result(
            name_file=d[0], name_set=name_test
        )

        ind_column.append(res["algorithm_name"])

        if transmitting_part:
            indices_t = ["c_t", "l_t", "q_t", "coil_t"]
            column_t = make_column(
                indices=indices_t,
                data=res
            )
            if not np.any(table_transmit):
                table_transmit = np.array([column_t])
            else:
                table_transmit = np.concatenate((table_transmit, np.array([column_t])))

        if receiving_part:
            indices_r = ["c_r", "l_r", "q_r", "coil_r"]
            column_r = make_column(
                indices=indices_r,
                data=res
            )
            if not np.any(table_receive):
                table_receive = np.array([column_r])
            else:
                table_receive = np.concatenate((table_receive, np.array([column_r])))

    table_transmit = table_transmit.T
    table_receive = table_receive.T

    with pd.ExcelWriter(f"{name_test}.xlsx") as writer:
        if transmitting_part:
            pd.DataFrame(
                table_transmit,
                index=["c_t", "l_t", "q_t", "r_in_t", "r_out_t", "n_t"],
                columns=ind_column
            ).to_excel(
                writer, sheet_name="transmitting part"
            )

        if receiving_part:
            pd.DataFrame(
                table_receive,
                index=["c_r", "l_r", "q_r", "r_in_r", "r_out_r", "k_r"],
                columns=ind_column
            ).to_excel(
                writer, sheet_name="receiving part"
            )

    return table_transmit, table_receive


def draw_two_coil(coil_1, coil_2, color=None, title=None, name_fig=None):
    if title is not None:
        plt.title(title)

    d = 0.01

    for r_in in coil_1:
        plt.gca().add_artist(
            ptc.Circle((0, 0),
                       radius=r_in,
                       color=color,
                       fill=False)
        )

    for r_in in coil_2:
        plt.gca().add_artist(
            ptc.Circle((np.max(coil_1) + np.max(coil_2) + d, 0),
                       radius=r_in,
                       color=color,
                       fill=False)
        )

    y_max = np.max(np.concatenate((coil_1, coil_2)))
    plt.xlim([-np.max(coil_1) - 0.01, np.max(coil_1) + 2 * np.max(coil_2) + 2 * 0.0015 + 0.01])
    plt.ylim([-y_max - 0.01, y_max + 0.01])

    plt.axis("off")
    # plt.grid()

    if name_fig is not None:
        plt.savefig(name_fig)
    plt.show()


def save_figure_two_coil(name_test, data):
    for d in data:
        res = read_result(
            name_file=d[0], name_set=name_test
        )

        coil_t = np.fromstring(res["coil_t"][1:-1], sep=", ", dtype=float)
        coil_r = np.fromstring(res["coil_r"][1:-1], sep=", ", dtype=float)

        name = "algorithm_name"
        draw_two_coil(
            coil_1=coil_t,
            coil_2=coil_r,
            color=d[1],
            # title="Передающая и принимающая катушки индуктивности\n(слева-направо)",
            name_fig=f"{name_test}_{res[name]}.svg"
        )


def main():
    name_test = "test8"

    test_result_table(
        name_test=name_test,
        data=RESULTS,

    )

    save_figure_two_coil(
        name_test=name_test,
        data=RESULTS
    )


if __name__ == "__main__":
    main()
