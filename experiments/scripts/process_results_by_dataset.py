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


def build_table(path, dataset_name):

    model_names = ['original_tree',
                'vapnik_tree',
                'breiman_tree',
                'modified_breiman_tree',
                ]

    table = p2l.Table((6, 5), float_format='.3f')
    table[0,1:] = ['Original tree', 'Vapnik', 'Breiman', 'Modified Breiman']
    table[0,1:].add_rule()
    table[1:,0] = ['Train acc.', 'Test acc.', 'Leaves', 'Height', 'Bound']

    for i, model in enumerate(model_names):
        tr_acc = []
        ts_acc = []
        leaves = []
        height = []
        bound = []
        with open(path + model + '.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                tr_acc.append(row[2])
                ts_acc.append(row[3])
                leaves.append(row[4])
                height.append(row[5])
                if i == 1:
                    bound.append(row[6])

        tr_acc = np.array(tr_acc, dtype=float)
        ts_acc = np.array(ts_acc, dtype=float)
        leaves = np.array(leaves, dtype=float)
        height = np.array(height, dtype=float)
        if i == 1:
            bound = np.array(bound, dtype=float)

        table[0+1, i+1] = MeanWithStd(tr_acc)
        table[1+1, i+1] = MeanWithStd(ts_acc)
        table[2+1, i+1] = MeanWithStd(leaves)
        table[3+1, i+1] = MeanWithStd(height)
        if i == 1:
            table[4+1, i+1] = MeanWithStd(bound)

    table.caption = dataset_name.replace('_', ' ').title() + f' Dataset, {len(tr_acc)} runs'
    table[2,1:].highlight_best(highlight=lambda content: '$\\mathbf{' + content[1:-1] + '}$')
    print(table.highlights)
    table[3:,1:].change_format('.1f')

    return table


if __name__ == "__main__":

    exp_name = 'test_date'
    # exp_files = os.listdir(exp_dir)
    # datetime_format = "%Y-%m-%d_%Hh%Mm"
    # exp_name = max(exp_files, key=lambda exp_date: datetime.strptime(exp_date, datetime_format))
    doc = p2l.Document(exp_name + '_results', '.')

    for dataset in dataset_list:
        path = './experiments/results/' + dataset.name + '/' + exp_name + '/'

        doc += build_table(path, dataset.name)


    doc.build()
