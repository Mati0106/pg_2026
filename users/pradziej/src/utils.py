from pathlib import Path
import pandas as pd

def load_data(input_list: list, cond: str, idx : int = 0, extension: str = '.csv'):
    """
        It's a function for loading data set for provided input list with condition achieved
        Args:
            input_list: List of datasets.
            cond: Expected to load dataset.
            index: The index to be taken if there are more than one element in the input list that satisfies the condition. (default is 0).
            extension: The file extension of the dataset to be loaded (default is '.csv').

        Returns: 
            Pandas DataFrame with loaded dataset.
        Raises:
            IndexError: If the provided index is out of range for the filtered list of datasets.
            Exception: If any other exception occurs during the dataset loading process.
    """
    output_list = [el for el in input_list if el == cond]
    path_datasets = Path.cwd().joinpath("datasets")
    try:
        path_output_list = path_datasets.joinpath(output_list[idx] + extension)
    except IndexError as ie:
        print("No such index in given list. " + ie)
        raise ie
    except Exception as e:
        print("Exception occurs during data set load" + e)
        raise e
    
    return pd.read_csv(path_output_list)
