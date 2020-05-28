import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np

from experiments.datasets.datasets import dataset_list


class MeanWithStd(float, p2l.TexObject):
    def __new__(cls, array):
        mean = array.mean()
        instance = super().__new__(cls, mean)
        instance.mean = mean
        instance.std = array.std()
        return instance

    def __format__(self, format_spec):
        return f'${format(self.mean, format_spec)} \pm {format(self.std, format_spec)}$'


if __name__ == "__main__":

    exp_name = 'test_date'
    # exp_files = os.listdir(exp_dir)
    # datetime_format = "%Y-%m-%d_%Hh%Mm"
    # exp_name = max(exp_files, key=lambda exp_date: datetime.strptime(exp_date, datetime_format))
    doc = p2l.Document(exp_name + '_results', '.')


    model_names = ['original_tree',
                'vapnik_tree',
                'breiman_tree',
                'modified_breiman_tree',
                ]

    table = doc.new(p2l.Table((len(dataset_list)+2, 5), float_format='.3f', alignment='lcccc'))

    table[0:2,0].multicell('Dataset', v_shift='-3pt')
    table[0,1:] = 'Model'
    table[1,1:] = ['Original tree', 'Ours', 'Breiman', 'Modified Breiman']
    table[0,1:].add_rule()
    table[2:,0] = [d.name.replace('_', ' ').title() + f' ({d.load().n_examples})' for d in dataset_list]
    table[1].add_rule()

    for d, dataset in enumerate(dataset_list):
        for i, model in enumerate(model_names):
            ts_acc = []
            path = './experiments/results/' + dataset.name + '/' + exp_name + '/'
            try:
                with open(path + model + '.csv', 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    for row in reader:
                        ts_acc.append(row[3])
            except FileNotFoundError:
                ts_acc.append(np.nan)
            
            table[d+2, i+1] = MeanWithStd(np.array(ts_acc, dtype=float))

        table[d+2,1:].highlight_best(highlight=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=0.0025, rtol=0)

    table.caption = """Mean test accuracy and its standard deviation on 25 random splits of 19 selected datasets taken from the UCI Machine Learning Repository. The train-test split ratio was $75\\%$ - $25\\%$. The total number of examples of each dataset is in parenthesis. The column ``Original tree'' presents the result of the full unpruned tree, the ``Ours'' column is the original tree pruned with Vapnik's bound with our results, the ``Breiman'' column is the original tree pruned with the CART pruning algorithm and ``Modified Breiman'' is the original tree pruned with the same algorithm as CART, but the complexity term is changed to reflect the dependencies found in our work. For Breiman and Modified Breiman, the number of folds in the cross-validation was 10."""

    doc.build()
