from pathlib import Path

def loadData(
        elements: list,
        cond: str,
        index_output: int,
        extension=".csv"
    ):
    """


    """

    output_list = [el for el in elements if el==cond]

    print(output_list)
    try:
        file_name = output_list[index_output]+extension
    except IndexError:
        raise("Wrong index ZMIANA")


    cwd = Path.cwd()
    PATH = cwd.parents[1]

    print(PATH)
    path_datasets = PATH.joinpath("datasets")
    path_output_list = path_datasets.joinpath(file_name)
    print(path_output_list)

    return path_output_list

