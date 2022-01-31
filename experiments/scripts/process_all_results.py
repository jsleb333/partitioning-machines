import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np
from scipy.stats import t as students_t

from experiments.datasets import dataset_list as d_list
from experiments.models import model_dict
from experiments.utils import camel_to_snake

from partitioning_machines import func_to_cmd


class MeanWithCI(float, p2l.TexObject):
    def __new__(cls, array, confidence_level=.6827):
        mean = array.mean()
        instance = super().__new__(cls, mean)
        instance.mean = mean
        instance.std = array.std(ddof=1)
        instance.ci = MeanWithCI.confidence_interval(instance, confidence_level, len(array))
        return instance

    def confidence_interval(self, confidence_level, n_samples):
        return self.std * students_t.ppf((1+confidence_level)/2, n_samples-1)

    def __format__(self, format_spec):
        # return f'${format(self.mean, format_spec)} \pm {format(self.ci, format_spec)}$'
        return f'${format(self.mean, format_spec)}$'


@func_to_cmd
def process_results(exp_name='exp02'):
    """
    Produces Table 1 from the paper (Appendix E). Will try to call pdflatex if installed.

    Args:
        exp_name (str): Name of the experiment used when the experiments were run. If no experiments by that name are found, entries are set to 'nan'.

    Prints in the console the tex string used to produce the tables, and will compile it if possible.
    """
    path = './experiments/results/' + exp_name
    doc = p2l.Document(exp_name + '_all_results', path)
    doc.packages['geometry'].options.append('landscape')

    dataset_list = [d for d in d_list if d.name not in ['cardiotocography10']]
    # dataset_list = d_list

    significance = 0.1

    with open(path + '/iris/no_pruning_exp_config.py') as file:
        namespace = {}
        exec(file.read().replace('"', ''), namespace)
        exp_config = namespace['exp_config']

    caption = f"""Mean test accuracy and standard deviation on {exp_config['n_draws']} random splits of {len(dataset_list)} datasets taken from the UCI Machine Learning Repository \\citep{{Dua:2019}}. In parenthesis is the total number of examples followed by the number of classes of the dataset. The best performances up to a ${significance}\\%$ accuracy gap are highlighted in bold. The maximum number of leaves is set to {exp_config['model_config']['max_n_leaves']}. The dataset is split into training and test sets with a ratio of {exp_config['test_split_ratio']}."""

    label = "results"

    alignement = r'@{\hspace{6pt}}'.join('l'+'c'*len(model_dict))
    table = doc.new(p2l.Table(
        (len(dataset_list)+6, len(model_dict)+1),
        float_format='.2f',
        alignment=alignement,
        caption=caption,
        label=label
    ))
    table.body.insert(0, '\\small')

    table[0:2,0].multicell('Dataset', v_shift='-3pt')
    table[0,1:] = 'Model'
    table[1,1:] = [name.replace('_', ' ')
                   .title()
                   .replace(' Pruning', '')
                   .replace('Shawe Taylor', 'ST')
                   .replace('Hyp Inv', 'HTI')
                   .replace('Kearns Mansour', 'KM')
                   for name in model_dict]
    table[0,1:].add_rule()
    is_nominal = lambda d: '*' if d.nominal_features else ''
    table[2:len(dataset_list)+2,0] = [d.name.replace('_', ' ').title() + is_nominal(d) + f' ({d.n_examples}, {d.n_classes})' for d in dataset_list]
    table[1].add_rule()
    table[-5].add_rule()

    ts_accs = np.zeros((len(model_dict), len(dataset_list)))

    for d, dataset in enumerate(dataset_list):
        for i, model_name in enumerate(model_dict):
            ts_acc = []
            path = './experiments/results/' + exp_name + '/' + dataset.name + '/'
            try:
                with open(path + model_name + '.csv', 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    pos = header.index('test_accuracy')
                    for row in reader:
                        ts_acc.append(row[pos])
                table[d+2, i+1] = MeanWithCI(100*np.array(ts_acc, dtype=float))

            except FileNotFoundError:
                pass

            if ts_acc:
                ts_accs[i,d] = np.mean(np.array(ts_acc, dtype=float))

        table[d+2,2:-1].highlight_best(best=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=significance, rtol=0)

    table[-4,0] = r'Number of best'
    bests = np.isclose(ts_accs, np.max(ts_accs[:-1,:], axis=0), rtol=0, atol=significance/100)
    table[-4,1:] = np.sum(bests, axis=1)
    table[-3,0] = r'Better than Ours ST'
    bests = np.zeros_like(ts_accs, dtype=int)
    for d in range(len(dataset_list)):
        b = ts_accs[1,d]
        for i in range(len(model_dict)):
            a = ts_accs[i,d]
            if a > b and not np.isclose(a, b, rtol=0, atol=significance/100):
                bests[i,d] = 1
    table[-3,1:] = np.sum(bests, axis=1)

    table[-2,0] = 'Mean'
    table[-2,1:] = [MeanWithCI(ts_accs[i]*100) for i in range(len(model_dict))]
    table[-1,0] = 'Fraction of oracle'
    table[-1,1:] = [MeanWithCI((1-ts_accs[-1]/ts_accs[i])*100) for i in range(len(model_dict))]

    d = [dataset_list[i] for i in [0, 2, 3, 4, 16]]

    # table[2,0] = f'BCWD\\textsuperscript{{a}} ({d[0].n_examples}, {d[0].n_classes})'
    # table[4,0] = f'CMSC\\textsuperscript{{b}} ({d[1].n_examples}, {d[1].n_classes})'
    # table[5,0] = f'CBS\\textsuperscript{{c}} ({d[2].n_examples}, {d[2].n_classes})'
    # table[6,0] = f'DRD\\textsuperscript{{d}} ({d[3].n_examples}, {d[3].n_classes})'
    # table[18,0] = f'WFR24\\textsuperscript{{e}} ({d[4].n_examples}, {d[4].n_classes})'

    # table += """\n\\footnotesize \\textsuperscript{a}Breast Cancer Wisconsin Diagnostic, \\textsuperscript{b}Climate Model Simulation Crashes, \\textsuperscript{c}Connectionist Bench Sonar,\n\n\\textsuperscript{d}Diabetic Retinopathy Debrecen, \\textsuperscript{e}Wall Following Robot 24"""

    doc.add_package('natbib')


    doc.build(delete_files='all', show_pdf=False)
    # try:
    #     doc.build(delete_files='all')
    # except:
    #     pass


if __name__ == "__main__":
    process_results(exp_name='exp06-less-test')
