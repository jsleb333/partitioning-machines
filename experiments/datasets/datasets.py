import os, sys
sys.path.append(os.getcwd())
import pickle as pkl
from urllib import request
import pandas as pd


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


def classproperty(method):
    class ClassPropertyDescriptor:
        def __init__(self, method):
            self.method = method
        def __get__(self, obj, objtype=None):
            return self.method(objtype)
    return ClassPropertyDescriptor(method)


class Dataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.n_examples, self.n_features = self.data.shape

    @property
    def data(self):
        return self.dataframe.loc[:, self.dataframe.columns != 'class'].to_numpy(dtype=float)

    @property
    def target(self):
        return self.dataframe.loc[:, 'class'].to_numpy()

    def __repr__(self):
        return f'Dataset f{type(self).name} with {self.n_examples} examples and {self.n_features} features'

    @classproperty
    def path_to_raw_file(cls):
        return os.path.dirname(__file__) + '/raw/' + cls.name + '.raw'

    @classmethod
    def load(cls):
        if not os.path.exists(cls.path_to_raw_file):
            cls.download_dataset()

        return cls(cls.create_dataframe())

    @classmethod
    def download_dataset(cls):
        content = request.urlopen(cls.url)
        os.makedirs(os.path.dirname(__file__) + '/raw/', exist_ok=True)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    @classmethod
    def create_dataframe(cls):
        raise NotImplementedError


class BreastCancerWisconsinDiagnostic(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    name = "breast_cancer_wisconsin_diagnostic"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = ['id', 'class'] + [f'attr {i}' for i in range(30)]
            df = pd.read_csv(file, names=col_names, header=None)
            df.drop(columns=col_names[0], inplace=True)
        return df
dataset_list.append(BreastCancerWisconsinDiagnostic)

class ClimateModelSimulationCrashes(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    name = "climate_model_simulation_crashes"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(ClimateModelSimulationCrashes)

class ConnectionistBenchSonar(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    name = "connectionist_bench_sonar"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(ConnectionistBenchSonar)

class DiabeticRetinopathyDebrecen(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    name = "diabetic_retinopathy_debrecen"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(DiabeticRetinopathyDebrecen)

class Fertility(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    name = "fertility"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Fertility)

class HabermansSurvival(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    name = "habermans_survival"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(HabermansSurvival)

class ImageSegmentation(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
    name = "image_segmentation"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[0]:'class'}, inplace=True)
        return df
dataset_list.append(ImageSegmentation)

class IndianLiverPatient(Dataset):
    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    name = "indian_liver_patient"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
            df.rename(columns={list(df)[1]:'gender'}, inplace=True)
            for i, gender in enumerate(df['gender']):
                if gender == 'Male':
                    df.at[i, 'gender'] = 0
                else:
                    df.at[i, 'gender'] = 1
        return df
dataset_list.append(IndianLiverPatient)

class Iris(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    name = "iris"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        return df
dataset_list.append(Iris)

class Wine(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    name = "wine"
    @classmethod
    def create_dataframe(cls):
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
dataset_list.append(Wine)


if __name__ == "__main__":
    # dataset = IndianLiverPatient
    # dataset.download_dataset()
    # df = dataset.create_dataframe()
    # print(df)

    for d in load_datasets():
        print(d.n_examples, d.target)
