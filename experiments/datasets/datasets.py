import os, sys

from pandas.core.indexes import category
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
    for dataset in dataset_list:
        if datasets is not None and dataset.name in datasets:
            yield dataset
            dataset.unload()


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class LazyDataset(type):
    """Dataset metaclass which makes every datasets implemented 'lazy' in the sense that it will be loaded in memory only when its attributes are accessed. Each dataset is a singleton and can be unloaded from memory using the 'unload' method.
    """
    def __init__(cls, cls_name, bases, attrs):
        super().__init__(cls, bases, attrs)
        cls.name = camel_to_snake(cls_name)
        cls.path_to_raw_file = os.path.dirname(__file__) + '/raw/' + cls.name + '.raw'
        cls._is_loaded = False
        dataset_list.append(cls) # Append to the list of available datasets

    def load(cls):
        cls._is_loaded == True
        cls.nominal_features = []
        if not os.path.exists(cls.path_to_raw_file):
            cls.download_dataset()
        cls.dataframe = cls.build_dataframe(cls)
        cls.data = cls.dataframe.loc[:, cls.dataframe.columns != 'class'].to_numpy()
        cls.target = cls.dataframe.loc[:, 'class'].to_numpy()
        cls.n_examples = cls.data.shape[0]
        cls.n_features = cls.data.shape[1]
        cls.n_classes = len(set(cls.target))
        cls.nominal_feat_dist = cls._compute_nominal_feature_distribution()
        return cls

    def _compute_nominal_feature_distribution(cls):
        tmp_cat_dist = [len(cls.dataframe.iloc[:,i].cat.categories) for i in cls.nominal_features]
        nominal_feat_dist = [0]*max(tmp_cat_dist)
        for n_nom_feat in tmp_cat_dist:
            nominal_feat_dist[n_nom_feat-1] += 1
        return nominal_feat_dist

    def unload(cls):
        del cls.dataframe
        del cls.data
        del cls.target
        cls._is_loaded = False

    def __call__(cls, *args, **kwds): # Makes the dataset a singleton
        if not cls._is_loaded:
            cls.load()
        return cls

    def __getattr__(cls, attr): # Loads the dataset on-demand (i.e. lazily)
        if not cls._is_loaded:
            cls.load()
            return getattr(cls, attr)
        raise AttributeError

    def __repr__(cls):
        return f'Dataset {cls.name} with {cls.n_examples} examples and {cls.n_features} features'

    def download_dataset(cls):
        content = request.urlopen(cls.url)
        os.makedirs(os.path.dirname(__file__) + '/raw/', exist_ok=True)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    def build_dataframe(cls) -> pd.DataFrame:
        raise NotImplementedError


class Adult(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    bibtex_label = "kohavi1996scaling"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'age': float,
                'workclass': 'category',
                'fnlwgt': float,
                'education': 'category',
                'education-num': float,
                'marital-status': 'category',
                'occupation': 'category',
                'relationship': 'category',
                'race': 'category',
                'sex': 'category',
                'capital-gain': float,
                'capital-loss': float,
                'hours-per-week': float,
                'native-country': 'category',
                'class': 'category'
            }
            cls.nominal_features = [i for i, v in enumerate(features.values()) if v == 'category']
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
            for col_name, dtype in features.items():
                if dtype == 'category':
                    df[col_name].cat.categories = list(i for i, _ in enumerate(set(df[col_name])))
        return df

class BreastCancerWisconsinDiagnostic(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    bibtex_label = "street1993nuclear"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = ['id', 'class'] + [f'attr {i}' for i in range(30)]
            df = pd.read_csv(file, names=col_names, header=None)
            df.drop(columns=col_names[0], inplace=True)
        return df

class Cardiotocography10(metaclass=LazyDataset):
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

class ClimateModelSimulationCrashes(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    bibtex_label = "lucas2013failure"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class ConnectionistBenchSonar(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    bibtex_label = "gorman1988analysis"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class DiabeticRetinopathyDebrecen(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    bibtex_label = "antal2014ensemble"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Fertility(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    bibtex_label = "gil2012predicting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class HabermansSurvival(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    bibtex_label = "haberman1976generalized"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class HeartDiseaseClevelandProcessed(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    bibtex_label = "detrano1989international"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'age': float,
                'sex': 'category',
                'cp': 'category',
                'trestbps': float,
                'chol': float,
                'fbs': float,
                'restecg': 'category',
                'thalach': float,
                'exang': 'category',
                'oldpeak': float,
                'slope': 'category',
                'ca': 'category',
                'thal': 'category',
                'class': 'category'
            }
            cls.nominal_features = [i for i, v in enumerate(features.values()) if v == 'category']
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
            for col_name, dtype in features.items():
                if dtype == 'category':
                    df[col_name].cat.categories = list(i for i, _ in enumerate(set(df[col_name])))
        return df

class ImageSegmentation(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[0]:'class'}, inplace=True)
        return df

class Ionosphere(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    bibtex_label = "sigillito1989classification"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Iris(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    bibtex_label = "fisher1936use"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        return df

class Parkinson(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    bibtex_label = "little2007exploiting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0)
            df.rename(columns={'status':'class'}, inplace=True)
            df.drop(columns='name', inplace=True)
        return df

class PlanningRelax(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt"
    bibtex_label = "bhatt2012planning"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep='\t', )
            df.drop(columns=list(df)[-1], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class QSARBiodegradation(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
    bibtex_label = "mansouri2013quantitative"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep=';')
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Seeds(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    bibtex_label = "charytanowicz2010complete"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Spambase(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class VertebralColumn3C(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    bibtex_label = "berthonnaud2005analysis"
    def build_dataframe(cls):
        with ZipFile(cls.path_to_raw_file, 'r') as zipfile:
            with zipfile.open('column_3C.dat') as file:
                df = pd.read_csv(file, header=None, delim_whitespace=True)
                df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class WallFollowingRobot24(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data"
    bibtex_label = "freire2009short"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df

class Wine(metaclass=LazyDataset):
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

class Yeast(metaclass=LazyDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    bibtex_label = "horton1996probabilistic"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.drop(columns=list(df)[0], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df


if __name__ == "__main__":
    for i, d in enumerate(load_datasets(['heart_disease_cleveland_processed'])):
        print(d)
        assert not np.isnan(d.data.sum())
        print(i, d.name, d.nominal_features, d.nominal_feat_dist)


