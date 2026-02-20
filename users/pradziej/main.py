from src.utils import load_data
from src.settings import settings

elements = settings.get('elements', [])
boston = load_data(input_list=elements, cond='boston')
print(boston.head())