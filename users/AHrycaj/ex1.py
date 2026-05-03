

def load_data(input_list,cond,extension='.csv'):
    df = pd.read_csv(input_list)
    if cond:
"""

"""
# Proszę utworzyć liste [,titanic',,bostom,,Iris']
elements = ['iris', 'booston', 'titanic']
# Dla danej listy wybrać tylko boston za pomoca list comprehension
output_list = [el for el in elements if el == 'boston']
# Używając ścieżek z biblioteki pathlib
from pathlib import Path

cwd = Path.cwd()
path_datasets = cwd.joinpath('datasets')
path_output_list = path_datasets.joinpath(output_list[0])
cwd.parents[0]
import pandas as pd

print(path_output_list)

