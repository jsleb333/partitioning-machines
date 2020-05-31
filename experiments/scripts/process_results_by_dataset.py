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


def build_table(dataset_name):

    model_names = [
        'original_tree',
        'breiman_tree',
        'modified_breiman_tree',
        'ours',
        ]
    
    exp_names = [
        'test_date',
        'test_date',
        'test_date',
        'shawe-taylor_bound'
    ]

    table = p2l.Table((7, 5), float_format='.3f')
    table[0,1:] = ['Original', 'CART', 'M-CART', 'Ours']
    table[0,1:].add_rule()
    table[1:,0] = ['Train acc.', 'Test acc.', 'Leaves', 'Height', 'Bound', 'Time $[s]$']

    for i, (model, exp) in enumerate(zip(model_names, exp_names)):
        tr_acc = []
        ts_acc = []
        leaves = []
        height = []
        bound = []
        time = []
        path = './experiments/results/' + dataset_name + '/' + exp + '/' + model + '.csv'
        with open(path, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                tr_acc.append(row[2])
                ts_acc.append(row[3])
                leaves.append(row[4])
                height.append(row[5])
                if i == 3:
                    bound.append(row[6])
                    time.append(row[7])

        tr_acc = np.array(tr_acc, dtype=float)
        ts_acc = np.array(ts_acc, dtype=float)
        leaves = np.array(leaves, dtype=float)
        height = np.array(height, dtype=float)
        if i == 3:
            bound = np.array(bound, dtype=float)
            time = np.array(time, dtype=float)

        table[0+1, i+1] = MeanWithStd(tr_acc)
        table[1+1, i+1] = MeanWithStd(ts_acc)
        table[2+1, i+1] = MeanWithStd(leaves)
        table[3+1, i+1] = MeanWithStd(height)
        if i == 3:
            table[4+1, i+1] = MeanWithStd(bound)
            table[5+1, i+1] = MeanWithStd(time)

    table.caption = dataset_name.replace('_', ' ').title() + f' Dataset, {len(tr_acc)} runs'
    table[2,1:].highlight_best(highlight=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=.0025, rtol=0)
    table[3:,1:].change_format('.1f')

    return table


if __name__ == "__main__":

    exp_name = 'shawe-taylor_bound'
    # exp_files = os.listdir(exp_dir)
    # datetime_format = "%Y-%m-%d_%Hh%Mm"
    # exp_name = max(exp_files, key=lambda exp_date: datetime.strptime(exp_date, datetime_format))
    doc = p2l.Document(exp_name + '_results', '.')

    tables = [build_table(dataset.name) for dataset in dataset_list]

    times_leaves_cart = [table[3,4].data[0][0] / table[3,2].data[0][0] for table in tables]
    print('cart leaves', sum(times_leaves_cart)/len(times_leaves_cart))
    acc_gain_vs_cart = [table[2,4].data[0][0] - table[2,2].data[0][0] for table in tables]
    print('acc gain ours vs cart', sum(acc_gain_vs_cart)/len(acc_gain_vs_cart))

    times_leaves_mcart = [table[3,3].data[0][0] / table[3,2].data[0][0] for table in tables]
    print('mcart leaves', sum(times_leaves_mcart)/len(times_leaves_mcart))
    acc_gain_vs_mcart = [table[2,3].data[0][0] - table[2,2].data[0][0] for table in tables]
    print('acc gain cart vs m-cart', sum(acc_gain_vs_mcart)/len(acc_gain_vs_mcart))

    doc.body.extend(tables)

    doc.build()
