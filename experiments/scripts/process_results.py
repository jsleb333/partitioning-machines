import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np

from experiments.datasets.datasets import load_datasets
from experiments.datasets.datasets import BreastCancerWisconsinDiagnostic, ClimateModelSimulationCrashes, ConnectionistBenchSonar, DiabeticRetinopathyDebrecen, Fertility, HabermansSurvival, ImageSegmentation, IndianLiverPatient, Iris, Wine

dataset = Fertility

exp_dir = './experiments/results/' + dataset.name + '/'

exp_files = os.listdir(exp_dir)

datetime_format = "%Y-%m-%d_%Hh%Mm"
most_recent = max(exp_files, key=lambda exp_date: datetime.strptime(exp_date, datetime_format))

path = exp_dir + most_recent + '/'

model_names = ['original_tree',
               'vapnik_tree',
               'breiman_tree',
               'modified_breiman_tree',
               ]

doc = p2l.Document(dataset.name + '_results', path)
table = doc.new(p2l.Table((5, 5), float_format='.3f'))
table[0,1:] = ['Original tree', 'Vapnik', 'Breiman', 'Modified Breiman']
table[0,1:].add_rule()
table[1:,0] = ['Train acc.', 'Test acc.', 'Leaves', 'Height']

for i, model in enumerate(model_names):
    tr_acc = []
    ts_acc = []
    leaves = []
    height = []
    with open(path + model + '.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            tr_acc.append(row[2])
            ts_acc.append(row[3])
            leaves.append(row[4])
            height.append(row[5])
    
    tr_acc = np.array(tr_acc, dtype=float)
    ts_acc = np.array(ts_acc, dtype=float)
    leaves = np.array(leaves, dtype=float)
    height = np.array(height, dtype=float)

    table[1, i+1] = tr_acc.mean()
    table[2, i+1] = ts_acc.mean()
    table[3, i+1] = leaves.mean()
    table[4, i+1] = height.mean()

table.caption = dataset.name.replace('_', ' ') + f' Dataset, {len(tr_acc)} runs'

table[3:,1:].change_format('.1f')
table[2,1:].highlight_best()

doc.build()