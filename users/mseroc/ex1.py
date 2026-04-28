# internal
from users.mseroc.utils import load_data
from users.mseroc.settings import settings
from users.mseroc.credentials import credentials

elements = settings.get("elements", [])
target = settings.get("target", "")
user = credentials.get("user", "")
password = credentials.get("password", "")


def connection_to_db(user, password):
    print(f"User is: {user}, Password is: {password}")
    return ""


con = connection_to_db(user, password)
data = load_data(input_list=elements, index_output=0, cond='iris')
