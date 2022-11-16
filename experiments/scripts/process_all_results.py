import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np
from scipy.stats import t as students_t

from experiments.datasets import dataset_list as d_list
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
        # return f'${format(self.mean, format_spec)} ({format(self.ci, format_spec)})$'
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

    # dataset_list = [d for d in d_list if d.name not in ['cardiotocography10']]
    dataset_list = d_list

    significance = 0.1

    with open(path + '/iris/ours_shawe_taylor_pruning_exp_config.py') as file:
        namespace = {}
        exec(file.read().replace('"', ''), namespace)
        exp_config = namespace['exp_config']

    caption = f"""Mean test accuracy on {exp_config['n_draws']} random splits of {len(dataset_list)} data sets obtained from the UCI Machine Learning Repository \\citep{{Dua:2019}}. Starred data sets contain a mixture of feature types (the others are real-valued only). The total number of examples followed by the number of classes of the dataset is indicated in parenthesis. The best performances up to a ${significance}\\%$ accuracy gap are highlighted in bold. The maximum number of leaves is set to {exp_config['model_config']['max_n_leaves']}. The dataset is split into training and test sets with a ratio of {exp_config['test_split_ratio']}."""

    label = "results"

    models = {
        'no_pruning': 'OG',
        'cart_pruning': 'CC',
        'reduced_error_pruning': 'RE',
        'kearns_mansour_pruning': 'KM',
        'ours_shawe_taylor_pruning': 'Ours',
        'oracle_pruning': 'Oracle',
    }

    alignment = r'@{\hspace{6pt}}'.join('l'+'c'*len(models))
    table = doc.new(p2l.Table(
        (len(dataset_list)+7, len(models)+1),
        float_format='.2f',
        alignment=alignment,
        caption=caption,
        label=label
    ))
    table.body.insert(0, '\\small')

    table[0:2,0].multicell('Data set', v_shift='-3pt')
    table[0,1:] = 'Pruning model'
    table[1,1:] = list(models.values())
    table[0,1:].add_rule()
    is_nominal = lambda d: '*' if d.nominal_features else ''
    table[2:len(dataset_list)+2,0] = [d.name.replace('_', ' ').title() + is_nominal(d) + f' ({d.n_examples}, {d.n_classes})' for d in dataset_list]
    table[1].add_rule()

    ts_accs = np.zeros((len(models), len(dataset_list)))

    times = {name:[] for name in models}
    leaves = {name:[] for name in models}

    for d, dataset in enumerate(dataset_list):
        for i, model_name in enumerate(models):
            ts_acc = []
            path = './experiments/results/' + exp_name + '/' + dataset.name + '/'
            try:
                with open(path + model_name + '.csv', 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    pos_acc = header.index('test_accuracy')
                    pos_time = header.index('time')
                    pos_leaves = header.index('n_leaves')
                    for row in reader:
                        ts_acc.append(row[pos_acc])
                        times[model_name].append(float(row[pos_time]))
                        leaves[model_name].append(float(row[pos_leaves]))
                table[d+2, i+1] = MeanWithCI(100*np.array(ts_acc, dtype=float))

            except:
                pass

            if ts_acc:
                ts_accs[i,d] = np.mean(np.array(ts_acc, dtype=float))

        table[d+2,1:-1].highlight_best(best=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=significance, rtol=0)

    n_d = len(dataset_list)+1

    table[n_d].add_rule()

    table[n_d+1,0] = r'Number of best'
    bests = np.isclose(ts_accs, np.max(ts_accs[:-1,:], axis=0), rtol=0, atol=significance/100)
    table[n_d+1,1:] = np.sum(bests, axis=1)
    table[n_d+1,-1] = 'N/A'

    # table[-3,0] = r'Better than Ours ST'
    # bests = np.zeros_like(ts_accs, dtype=int)
    # for d in range(len(dataset_list)):
    #     b = ts_accs[1,d]
    #     for i in range(len(models)):
    #         a = ts_accs[i,d]
    #         if a > b and not np.isclose(a, b, rtol=0, atol=significance/100):
    #             bests[i,d] = 1
    # table[-3,1:] = np.sum(bests, axis=1)

    table[n_d+2,0] = 'Mean accuracy (\\%)'
    table[n_d+2,1:] = [MeanWithCI(ts_accs[i]*100).mean for i in range(len(models))]

    table[n_d+3,0] = 'Mean fraction of Oracle (\\%)'
    table[n_d+3,1:] = [MeanWithCI(100*(ts_accs[i]/ts_accs[-1])).mean if (ts_accs[i] > 0).all() else 0 for i in range(len(models))]


    d = [dataset_list[i] for i in [0, 2, 3, 4, 16]]

    table[n_d+4,0] = 'Mean pruning time (s)'
    for i, (name, time) in enumerate(times.items()):
        table[n_d+4,i+1] = np.mean(time)
    table[n_d+4,1] = 'N/A'

    table[n_d+5,0] = 'Mean number of leaves'
    for i, (name, leave) in enumerate(leaves.items()):
        table[n_d+5,i+1] = np.mean(leave)

    doc.add_package('natbib')


    print(doc.build(delete_files='all', show_pdf=False))


if __name__ == "__main__":
    process_results(exp_name='exp06-bst10.5')
    # process_results(exp_name='exp06-less-test')
