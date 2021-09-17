import os, sys

sys.path.append(os.getcwd())
from urllib import request
import pandas as pd
import numpy as np
from zipfile import ZipFile

from utils import camel_to_snake


dataset_list = []

def load_datasets(datasets=None):
    """
    Args:
        datasets (list of str): Lists of datasets to load by name. If None, all available datasets are loaded iteratively.
    """
    for dataset in dataset_list:
        if (not datasets) or (datasets and dataset.name in datasets):
            yield dataset
            dataset.unload()

class Nominal(pd.CategoricalDtype):
    def __init__(self, categories=None):
        super().__init__(categories=categories, ordered=False)
    def __repr__(self):
        return 'nominal'

class Ordinal(pd.CategoricalDtype):
    def __init__(self, categories=None):
        super().__init__(categories=categories, ordered=True)
    def __repr__(self):
        return 'ordinal'

class Label(pd.CategoricalDtype):
    def __init__(self, categories=None):
        super().__init__(categories=categories, ordered=False)


class Dataset(type):
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
        if not os.path.exists(cls.path_to_raw_file):
            cls.download_dataset()
        cls.dataframe = cls.build_dataframe(cls)
        cls._convert_categorical_str_to_int(cls.dataframe)

        cls.data = cls.examples = cls.dataframe.loc[:, cls.dataframe.columns != 'label'].to_numpy()
        cls.target = cls.labels = cls.dataframe.loc[:, 'label'].to_numpy()
        cls.n_examples = cls.data.shape[0]
        cls.n_features = cls.data.shape[1]
        cls.n_classes = len(set(cls.target))

        cls.nominal_features, cls.ordinal_features = [], []
        for i, (col_name, col) in enumerate(cls.dataframe.items()):
            if isinstance(col.dtype, pd.CategoricalDtype) and col_name != 'label':
                if col.dtype.ordered: cls.ordinal_features.append(i)
                else: cls.nominal_features.append(i)
        cls.nominal_feat_dist = cls._compute_feature_distribution(cls.nominal_features)
        cls.ordinal_feat_dist = cls._compute_feature_distribution(cls.ordinal_features)

        return cls

    def _compute_feature_distribution(cls, features):
        tmp_cat_dist = [len(cls.dataframe.iloc[:,i].cat.categories) for i in features]
        if not tmp_cat_dist:
            return [0, 0]
        feat_dist = [0]*max(tmp_cat_dist)
        for n_feat in tmp_cat_dist:
            feat_dist[n_feat-1] += 1
        return feat_dist

    def unload(cls):
        if cls._is_loaded:
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

    def __str__(cls):
        return cls.name

    def _convert_categorical_str_to_int(cls, df):
        for col_name, col in df.items():
            if isinstance(col.dtype, pd.CategoricalDtype):
                col.cat.categories = range(len(col.cat.categories))

    def download_dataset(cls):
        content = request.urlopen(cls.url)
        os.makedirs(os.path.dirname(__file__) + '/raw/', exist_ok=True)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    def build_dataframe(cls) -> pd.DataFrame:
        raise NotImplementedError


class AcuteInflammation(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data"
    bibtex_label = "czerniak2003application"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r', encoding='utf-16') as file:
            features = {
                'Temperature' : float,
                'Occurrence' : Nominal(),
                'Lumbar pain' : Nominal(),
                'Urine pushing' : Nominal(),
                'Micturition pains' : Nominal(),
                'Burning' : Nominal(),
                'Inflammation' : Label(),
                'label' : Label(), #'Nephritis' : Label(),
            }
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features, delimiter='\t', decimal=',', encoding='utf-16le')
            df.drop(columns=['Inflammation'])
        return df

class Adult(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    bibtex_label = "kohavi1996scaling"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'age': float,
                'workclass': Nominal(),
                'fnlwgt': float,
                'education': Nominal(),
                'education-num': float,
                'marital-status': Nominal(),
                'occupation': Nominal(),
                'relationship': Nominal(),
                'race': Nominal(),
                'sex': Nominal(),
                'capital-gain': float,
                'capital-loss': float,
                'hours-per-week': float,
                'native-country': Nominal(),
                'label': Nominal()
            }
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
        return df

class Amphibians(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00528/dataset.csv"
    bibtex_label = "blachnik2019predicting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'ID': float,
                'MV': Nominal(),
                'SR': float,
                'NR': float,
                'TR': Nominal(),
                'VR': Ordinal(),
                'SUR1': Nominal(),
                'SUR2': Nominal(),
                'SUR3': Nominal(),
                'UR': Nominal(),
                'FR': Nominal(),
                'OR': Ordinal([25, 50, 75, 80, 99, 100]),
                'RR': Ordinal([0, 1, 2, 5, 9, 10]),
                'BR': Ordinal([0, 1, 2, 5, 9, 10]),
                'MR': Ordinal(),
                'CR': Nominal(),
                'label': Label(), # 'Green frogs': Nominal(),
                'Brown frogs': Nominal(),
                'Common toad': Nominal(),
                'Fire-bellied toad': Nominal(),
                'Tree frog': Nominal(),
                'Common newt': Nominal(),
                'Great crested newt': Nominal()
            }
            df = pd.read_csv(file, names=features.keys(), skiprows=[0,1], dtype=features, delimiter=';')
            df.drop(columns=['ID', 'MV', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog', 'Common newt', 'Great crested newt'], inplace=True)
        return df

class BreastCancerWisconsinDiagnostic(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    bibtex_label = "street1993nuclear"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = ['id', 'label'] + [f'attr {i}' for i in range(30)]
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
            df.rename(columns={'CLASS':'label'}, inplace=True)
        return df

class ClimateModelSimulationCrashes(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    bibtex_label = "lucas2013failure"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class ConnectionistBenchSonar(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    bibtex_label = "gorman1988analysis"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class DiabeticRetinopathyDebrecen(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    bibtex_label = "antal2014ensemble"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Fertility(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    bibtex_label = "gil2012predicting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class HabermansSurvival(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    bibtex_label = "haberman1976generalized"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class HeartDiseaseClevelandProcessed(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    bibtex_label = "detrano1989international"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'age': float,
                'sex': Nominal(),
                'cp': Nominal(),
                'trestbps': float,
                'chol': float,
                'fbs': float,
                'restecg': Nominal(),
                'thalach': float,
                'exang': Nominal(),
                'oldpeak': float,
                'slope': Nominal(),
                'ca': Nominal(),
                'thal': Nominal(),
                'label': Nominal()
            }
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
        return df

class ImageSegmentation(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[0]:'label'}, inplace=True)
        return df

class Ionosphere(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    bibtex_label = "sigillito1989classification"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Iris(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    bibtex_label = "fisher1936use"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'label'])
        return df

class Mushroom(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    bibtex_label = "lincoff1997field"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = { f: Nominal() for f in [
                'label',
                'cap-shape',
                'cap-surface',
                'cap-color',
                'bruises',
                'odor',
                'gill-attachment',
                'gill-spacing',
                'gill-size',
                'gill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat',
            ]}
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
        return df

class Parkinson(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    bibtex_label = "little2007exploiting"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0)
            df.rename(columns={'status':'label'}, inplace=True)
            df.drop(columns='name', inplace=True)
        return df

class PlanningRelax(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt"
    bibtex_label = "bhatt2012planning"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep='\t', )
            df.drop(columns=list(df)[-1], inplace=True)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class QSARBiodegradation(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
    bibtex_label = "mansouri2013quantitative"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep=';')
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Seeds(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    bibtex_label = "charytanowicz2010complete"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Spambase(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    bibtex_label = None
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class StatlogGerman(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    bibtex_label = "hofmann94statloggerman"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {i+1 : dtype for i, dtype in enumerate([
                Ordinal(['A14', 'A11', 'A12', 'A13']),
                float,
                Nominal(),
                Nominal(),
                float, #5
                Ordinal(['A65'] + [f'A{i}' for i in range(61, 65)]),
                Ordinal(),
                float,
                Nominal(),
                Nominal(), #10
                float,
                Nominal(),
                float,
                Nominal(),
                Nominal(), #15
                float,
                Ordinal(),
                Ordinal(),
                Nominal(),
                Nominal(), #20
            ])} | {'label': Label()}
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features, delim_whitespace=True)
        return df

class VertebralColumn3C(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    bibtex_label = "berthonnaud2005analysis"
    def build_dataframe(cls):
        with ZipFile(cls.path_to_raw_file, 'r') as zipfile:
            with zipfile.open('column_3C.dat') as file:
                df = pd.read_csv(file, header=None, delim_whitespace=True)
                df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class WallFollowingRobot24(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data"
    bibtex_label = "freire2009short"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Wine(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    bibtex_label = "aeberhard1994comparative"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = [
                'label',
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
            df.rename(columns={list(df)[-1]:'label'}, inplace=True)
        return df

class Zoo(metaclass=Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
    bibtex_label = "forsyth90zoo"
    def build_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            features = {
                'animal': Nominal(),
                'hair': Nominal(),
                'feathers': Nominal(),
                'eggs': Nominal(),
                'milk': Nominal(),
                'airborne': Nominal(),
                'aquatic': Nominal(),
                'predator': Nominal(),
                'toothed': Nominal(),
                'backbone': Nominal(),
                'breathes': Nominal(),
                'venomous': Nominal(),
                'fins': Nominal(),
                'legs': Ordinal([0,2,4,5,6,8]),
                'tail': Nominal(),
                'domestic': Nominal(),
                'catsize': Nominal(),
                'label': Label()
            }
            df = pd.read_csv(file, names=features.keys(), header=None, dtype=features)
            df.drop(columns=['animal'], inplace=True)
        return df



if __name__ == "__main__":
    for i, d in enumerate(load_datasets(['acute_inflammation'])):
        print(d)
        assert not np.isnan(d.data.sum())
        print(i, d.name, d.nominal_features, d.nominal_feat_dist)


