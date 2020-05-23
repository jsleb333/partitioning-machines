import os, sys
sys.path.append(os.getcwd())
import pickle as pkl
from urllib import request
import pandas as pd


dataset_list = []

def load_datasets():
    for dataset in dataset_list:
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
        return self.dataframe.loc[:,self.dataframe.columns[:-1]].to_numpy(dtype=float)
    
    @property
    def target(self):
        return self.dataframe.loc[:,self.dataframe.columns[-1]].to_numpy()
    
    def __repr__(self):
        return f'Dataset f{type(self).name} with {self.n_examples} examples and {self.n_features} features'
        
    @classproperty
    def path_to_raw_file(cls):
        return os.path.dirname(__file__) + '/raw/' + cls.name + '.raw'
    
    @classproperty
    def path_to_processed_file(cls):
        return os.path.dirname(__file__) + '/processed/' + cls.name + '.pkl'
    
    @classmethod
    def load(cls):
        if not os.path.exists(cls.path_to_processed_file):
            cls.download_dataset()
        
        return cls(cls.create_dataframe())
    
    @classmethod
    def download_dataset(cls):
        content = request.urlopen(cls.url)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    @classmethod
    def create_dataframe(cls):
        raise NotImplementedError


# class BreastCancerWisconsinOriginal(Dataset):
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#     name = "breast_cancer_wisconsin_original"
#     @classmethod
#     def create_dataframe(cls):
#         with open(cls.path_to_raw_file, 'r') as file:
#             col_names = [
#                 'id number',
#                 'Clump Thickness',
#                 'Uniformity of Cell Size',
#                 'Uniformity of Cell Shape',
#                 'Marginal Adhesion',
#                 'Single Epithelial Cell Size',
#                 'Bare Nuclei',
#                 'Bland Chromatin',
#                 'Normal Nucleoli',
#                 'Mitoses',
#                 'Class',
#             ]
#             df = pd.read_csv(file, header=None, names=col_names)
#             df.drop(columns=col_names[0], inplace=True)
#         return df
# dataset_list.append(BreastCancerWisconsinOriginal)

class ClimateModelSimulationCrashes(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    name = "climate_model_simulation_crashes"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
        return df
dataset_list.append(ClimateModelSimulationCrashes)

class ConnectionistBenchSonar(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    name = "connectionist_bench_sonar"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
        return df
dataset_list.append(ConnectionistBenchSonar)

class DiabeticRetinopathyDebrecen(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    name = "diabetic_retinopathy_debrecen"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
        return df
dataset_list.append(DiabeticRetinopathyDebrecen)
    
class Fertility(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    name = "fertility"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
        return df
dataset_list.append(Fertility)
    
class Iris(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    name = "iris"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'flower type'])
        return df
dataset_list.append(Iris)
    
class Wine(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    name = "wine"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
        return df
dataset_list.append(Wine)

    
if __name__ == "__main__":
    # for dataset in dataset_list:
    #     dataset.download_dataset()
    # dataset = Wine
    # d = dataset.load()
    # print(d.n_examples, d.n_features, d.data, d.target)
    for d in load_datasets():
        d