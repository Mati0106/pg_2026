from users.pszarlowski.settings import settings
from users.pszarlowski.utility import loadData
import pandas as pd


elements = settings.get('elements', [])
cond = settings.get('cond', str)
index_output = settings.get('index_output', int)
    # ['titanic-full', 'boston', 'Iris']



path = loadData(elements,cond, index_output)
print(path)

df = pd.read_csv(path)
print(df)