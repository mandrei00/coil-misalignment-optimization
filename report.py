import pandas as pd

from compare_result import *

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
            name_file=d, name_set=name_test
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

        print(table_receive)

        if receiving_part:
            pd.DataFrame(
                table_receive,
                index=["c_r", "l_r", "q_r", "r_in_r", "r_out_r", "k_r"],
                columns=ind_column
            ).to_excel(
                writer, sheet_name="receiving part"
            )

    return table_transmit, table_receive


def main():
    test_result_table(
        name_test="test1",
        data=RESULTS
    )


if __name__ == "__main__":
    main()
