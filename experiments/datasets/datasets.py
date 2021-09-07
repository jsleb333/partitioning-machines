import os, sys
sys.path.append(os.getcwd())
from urllib import request
import pandas as pd
import numpy as np
from zipfile import ZipFile
import re

dataset_list = []


def load_datasets(datasets=None):
    """
    Args:
        datasets (list of str): Lists of datasets to load by name. If None, all available datasets are loaded iteratively.
    """
    if datasets is None:
        datasets = [d.name for d in dataset_list]
    for dataset in dataset_list:
        if dataset.name in datasets:
            yield dataset.load()


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class Dataset(type):
    def __init__(cls, cls_name, bases, attrs):
        cls.dataset_name = camel_to_snake(cls_name)
        cls._dataframe = None
        # Each dataset is a singleton, but initializing will load into memory the dataset
        def cls_new(self, *args, **kwargs):
            if cls._dataframe is None:
                cls.load()
            return cls
        cls.__new__ = cls_new

        cls.build_dataframe = classmethod(cls.build_dataframe) # Make the method accessible to the class and not only the instance

        dataset_list.append(cls) # Append to the list of available datasets

        super().__init__(cls, bases, attrs)

    def __repr__(cls):
        return f'Dataset {cls.dataset_name} with {cls.n_examples} examples and {cls.n_features} features'

    @property
    def path_to_raw_file(cls):
        return os.path.dirname(__file__) + '/raw/' + cls.dataset_name + '.raw'

    def load(cls):
        if not os.path.exists(cls.path_to_raw_file):
            cls.download_dataset()

        cls.dataframe = cls.build_dataframe()

    def download_dataset(cls):
        content = request.urlopen(cls.url)
        os.makedirs(os.path.dirname(__file__) + '/raw/', exist_ok=True)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    def build_dataframe(cls):
        raise NotImplementedError

    @property
    def dataframe(cls):
        if cls._dataframe is None:
            cls.load()
        return cls._dataframe

    @dataframe.setter
    def dataframe(cls, value):
        cls._dataframe = value

    @property
    def data(cls):
        return cls.dataframe.loc[:, cls.dataframe.columns != 'class'].to_numpy(dtype=float)

    @property
    def target(cls):
        return cls.dataframe.loc[:, 'class'].to_numpy()

    @property
    def n_examples(cls):
        return cls.data.shape[0]

    @property
    def n_features(cls):
        return cls.data.shape[1]

    @property
    def n_classes(cls):
        return len(set(cls.target))


class BreastCancerWisconsinDiagnostic(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    bibtex_label = "street1993nuclear"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = ['id', 'class'] + [f'attr {i}' for i in range(30)]
            df = pd.read_csv(file, names=col_names, header=None)
            df.drop(columns=col_names[0], inplace=True)
        return df

class Cardiotocography10(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
    bibtex_label = "ayres2000sisporto"
    def build_dataframe(cls):
        with pd.ExcelFile(cls.path_to_raw_file) as file:
            df = pd.read_excel(file, sheet_name=file.sheet_names[1], header=0, skiprows=[0] + [i for i in range(2128, 2131)])
            cols = list(df)
            cols_to_drop = cols[:10] + cols[31:43] + cols[-2:]
            df.drop(columns=cols_to_drop, inplace=True)
            df.rename(columns={'CLASS':'class'}, inplace=True)
        return df

class ClimateModelSimulationCrashes(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    bibtex_label = "lucas2013failure"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class ConnectionistBenchSonar(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    bibtex_label = "gorman1988analysis"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class DiabeticRetinopathyDebrecen(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    bibtex_label = "antal2014ensemble"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Fertility(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    bibtex_label = "gil2012predicting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class HabermansSurvival(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    bibtex_label = "haberman1976generalized"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class ImageSegmentation(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[0]:'class'}, inplace=True)
        return df

class Ionosphere(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    bibtex_label = "sigillito1989classification"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Iris(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    bibtex_label = "fisher1936use"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        return df

class Parkinson(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    bibtex_label = "little2007exploiting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0)
            df.rename(columns={'status':'class'}, inplace=True)
            df.drop(columns='name', inplace=True)
        return df

class PlanningRelax(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt"
    bibtex_label = "bhatt2012planning"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep='\t', )
            df.drop(columns=list(df)[-1], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class QSARBiodegradation(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
    bibtex_label = "mansouri2013quantitative"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep=';')
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Seeds(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    bibtex_label = "charytanowicz2010complete"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Spambase(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class VertebralColumn3C(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    bibtex_label = "berthonnaud2005analysis"
    def build_dataframe(cls):
        with ZipFile(cls.path_to_raw_file, 'r') as zipfile:
            with zipfile.open('column_3C.dat') as file:
                df = pd.read_csv(file, header=None, delim_whitespace=True)
                df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class WallFollowingRobot24(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data"
    bibtex_label = "freire2009short"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Wine(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    bibtex_label = "aeberhard1994comparative"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = [
                'class',
                'Alcohol',
                'Malic acid',
                'Ash',
                'Alcalinity of ash',
                'Magnesium',
                'Total phenols',
                'Flavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color intensity',
                'Hue',
                'OD280/OD315 of diluted wines',
                'Proline'
            ]
            df = pd.read_csv(file, names=col_names, header=None)
        return df

class Yeast(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    bibtex_label = "horton1996probabilistic"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.drop(columns=list(df)[0], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df


if __name__ == "__main__":
    for i, d in enumerate(dataset_list):
        print(d)
        assert not np.isnan(d.data.sum())
        print(i, d.dataset_name, d.n_examples, d.n_classes)


