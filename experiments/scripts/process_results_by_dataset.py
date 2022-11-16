from ctypes import alignment
import python2latex as p2l
import sys, os

from xarray import align
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from scipy.stats import t as students_t

from experiments.datasets.datasets import dataset_list

from partitioning_machines import func_to_cmd

class MeanWithCI(float, p2l.TexObject):
    def __new__(cls, array, confidence_level=.6827):
        mean = np.mean(array)
        instance = super().__new__(cls, mean)
        instance.mean = mean
        instance.std = array.std(ddof=1)
        instance.ci = MeanWithCI.confidence_interval(instance, confidence_level, len(array))
        return instance

    def confidence_interval(self, confidence_level, n_samples):
        return self.std * students_t.ppf((1+confidence_level)/2, n_samples-1)

    def __format__(self, format_spec):
        return f'${format(self.mean, format_spec)} ({format(self.std, format_spec)})$'
        # return f'${format(self.mean, format_spec)} \pm {format(self.ci, format_spec)}$'
        # return f'${format(self.mean, format_spec)}$'


def build_table(dataset, exp_name):
    dataset_name = dataset.name
    models = {
        'no_pruning': 'OG',
        'cart_pruning': 'CC',
        'reduced_error_pruning': 'RE',
        'kearns_mansour_pruning': 'KM',
        'ours_shawe_taylor_pruning': 'Ours',
        'oracle_pruning': 'Oracle',
    }

    stats = {
        'train_accuracy': 'Train acc.',
        'val_accuracy': 'Val. acc.',
        'test_accuracy': 'Test acc.',
        'n_leaves': 'Leaves',
        'height': 'Height',
        'time': 'Time $[s]$',
        'bound': 'Bound'
    }

    significance = 0.1/100

    if isinstance(exp_name, str):
        exp_names = [exp_name]*len(models)
    else:
        exp_names = exp_name

    alignment = r'@{\hspace{6pt}}'.join('l'+'c'*len(models))
    table = p2l.Table((len(stats)+1, len(models)+1),
                      float_format='.3f',
                      alignment=alignment)
    table.body.insert(0, '\\small')

    table[0,1:] = list(models.values())
    table[0,1:].add_rule()
    table[1:,0] = list(stats.values())

    for i, (model, exp) in enumerate(zip(models, exp_names)):
        path = f'./experiments/results/{exp}/{dataset_name}/{model}.csv'
        with open(path, 'r', newline='') as file:
            df = pd.read_csv(file, header=0)

        for j, stat in enumerate(stats):
            if df[stat].isna().any():
                table[j+1,i+1] = 'NA'
            else:
                table[j+1, i+1] = MeanWithCI(df[stat])

    dataset_name = dataset_name.replace('_', ' ').title()

    dataset_citation = '' if dataset.bibtex_label is None else f'\\citep{{{dataset.bibtex_label}}}'

    table.caption = f'Statistics for the {dataset_name} data set {dataset_citation}. There are {dataset.n_examples} examples, {dataset.n_features} features ({dataset.n_real_valued_features} real-valued, {len(dataset.ordinal_features)} ordinal and {len(dataset.nominal_features)} nominal), and {dataset.n_classes} classes.'

    table[3,1:-1].highlight_best(best=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=significance, rtol=0)
    table[4:6,1:].format_spec = '.1f'

    return table


@func_to_cmd
def process_results(exp_name='exp06-bst10.5'):
    """
    Produces Tables 2 to 20 from the paper (Appendix E). Will try to call pdflatex if installed.

    Args:
        exp_name (str): Name of the experiment used when the experiments were run. If no experiments by that name are found, entries are set to 'nan'.

    Prints in the console some compiled statistics used in the paper and the tex string used to produce the tables, and will compile it if possible.
    """

    path = './experiments/results/' + exp_name
    doc = p2l.Document(exp_name + '_results_by_dataset', path)
    doc.add_package('natbib')

    tables = [build_table(dataset, exp_name) for dataset in dataset_list]

    doc.body.extend(tables)
    # print(doc.build(save_to_disk=False))

    doc.build(show_pdf=False)


if __name__ == "__main__":
    process_results()
