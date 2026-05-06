from pathlib import Path
import pandas as pd
import kagglehub
import logging as log

LOCAL_DATA_DIRECTORY = "datasets"
DEFAULT_EXTENSION = '.csv'


class DataLoader:
    def _load_data_from_kaggle(self, dataset_name: str, extension: str = DEFAULT_EXTENSION,
                               force_download: bool = False) -> str:
        access_token_path = Path('~/.kaggle/access_token').expanduser()
        log.debug(f'getting kaggle credentials from: {access_token_path}')
        if not Path.exists(access_token_path):
            log.debug("Logging into kaggle...")
            kagglehub.login()

        [ds_owner, ds_name] = dataset_name.split('/')
        log.debug(f"Owner {ds_owner}, name {ds_name}")
        output_file = ds_name + extension
        log.info(
            f"Downloading data '{dataset_name}' from kaggle into '{LOCAL_DATA_DIRECTORY}/{ds_owner}/{output_file}' ...")
        return kagglehub.dataset_download(dataset_name, output_dir=f'{LOCAL_DATA_DIRECTORY}/{ds_owner}',
                                          force_download=force_download)

    def get_data_file(self, dir_path: Path, extension: str = DEFAULT_EXTENSION) -> Path:
        files = Path(dir_path).glob(f'./*{extension}')
        f = list(files)[0]
        if f.is_file():
            return f.resolve()
        raise Exception(f'Can not find file with extension {extension} in {dir_path}')

    def load_data_from_local_cache(self, dir_path: Path, extension: str = '.csv') -> pd.DataFrame:
        path: Path = self.get_data_file(dir_path, extension=extension)
        log.debug(f"Reading local version of dataset from '{path}'...")
        self.df: pd.DataFrame = pd.read_csv(path)
        log.info(self.df.describe())
        log.info(self.df.head())
        return self.df

    def load_data(self, dataset_name: str, force_download: bool = False) -> pd.DataFrame:
        path = Path(self._load_data_from_kaggle(dataset_name, force_download=force_download))
        return self.load_data_from_local_cache(path)

    def print_report(self):
        # Data understanding
        print(f'Columns, ', self.df.columns)
        print(f'RiskLevel: ', self.df['RiskLevel'].unique())
        print('Head of the data', self.df.head())
        print('Tail of the data', self.df.tail())
        print('Types of the columns', self.df.dtypes)

    def clean_data(self):
        print('Amount of rows(observations), columns(features) [raw data]', self.df.shape)

        self.df.drop_duplicates(inplace=True)
        print('Amount of rows(observations), columns(features) [without duplicates data]', self.df.shape)

        self.df.dropna(inplace=True)
        print('Amount of rows(observations), columns(features) [without null values]', self.df.shape)
