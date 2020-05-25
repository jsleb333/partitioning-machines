import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime

from experiments.datasets.datasets import load_datasets, BreastCancerWisconsinDiagnostic

exp_files = [exp[0] for exp in os.walk('./experiments/results/' + BreastCancerWisconsinDiagnostic.name + '/')]
print(exp_files)

datetime_format = "%Y-%m-%d_%Hh%Mm"

most_recent = max(exp_files, key=lambda exp_date: datetime.strptime(exp_date, datetime_format))

print(most_recent)