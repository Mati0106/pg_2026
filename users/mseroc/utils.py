from pathlib import Path
import pandas as pd


def load_data(input_list: list,
              cond: str,
              index_output: int,
              extension: str = '.csv') -> pd.DataFrame():
    """

    Args:
        input_list: list

        cond: str
        index_output: int
        extension: str

    Returns:

    """
    output_list = [el for el in input_list if el == cond]

    cwd = Path.cwd()
    path_datasets = cwd.joinpath("datasets")
    try:
        path_output_list = path_datasets.joinpath(output_list[index_output] + extension)
    except IndexError:
        raise ("Not able to found selected file in provided list which is passing condition")

    data = pd.read_csv(path_output_list)
    return data
