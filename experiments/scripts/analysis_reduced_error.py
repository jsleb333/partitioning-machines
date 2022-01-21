import python2latex as p2l
import numpy as np
import pandas as pd
import os, sys

from sympy import N
sys.path.append(os.getcwd())

from experiments.datasets import QSARBiodegradation
from experiments.scripts.comp_reduced_error import NoPruningVal, ReducedErrorPruning, STPruningVal, OraclePruningVal


def extract_data(filepath):
    tr, ts = [], []

    df = pd.read_csv(filepath+'.csv', sep=',', header=0)

    tr = df['train_accuracy'].to_numpy().mean()
    ts = df['test_accuracy'].to_numpy().mean()

    return tr, ts


if __name__ == '__main__':
    dataset = QSARBiodegradation()
    exp_path = f'./experiments/results/reder-exp01/{dataset.name}/'
    n_draws = 20
    test_split_ratio = .2
    seed = 42
    max_n_leaves = [10, 20, 30, 40, 50, 60]
    val_split_ratios = np.linspace(0,1,11)[:-1]*(1-test_split_ratio)
    models = [NoPruningVal, STPruningVal, ReducedErrorPruning, OraclePruningVal]

    figure_name = 'comp_reder_n_leaves'
    doc = p2l.Document(filename=figure_name, filepath=exp_path)

    for n_leaves in max_n_leaves:
        plot = p2l.Plot(plot_name=figure_name+f'={n_leaves}',
                        plot_path=exp_path,
                        # width='.45\\textwidth',
                        height='10cm',
                        # as_float_env=False,
                        )

        plot.legend_position = 'south west'
        plot.y_min = 0.45
        plot.y_max = 1
        plot.y_label = 'Accuracy'
        plot.x_min = 0
        plot.x_max = 1
        plot.x_label = 'Validation ratio'
        plot.axis.kwoptions['legend style'] = '{font=\\tiny}'
        plot.caption = f'maximum number of leaves = {n_leaves}'

        for model, color in zip(models, p2l.holi):
            tr = []
            ts = []
            x = []
            for val_split_ratio in val_split_ratios:
                filename = f'{model.model_name}-val={val_split_ratio:.2f}-n_leaves={n_leaves}'
                path = exp_path + f'{model.model_name}/n-leaves={n_leaves}/'
                try:
                    tr_acc, ts_acc = extract_data(path + filename)
                    tr.append(tr_acc)
                    ts.append(ts_acc)
                    x.append(val_split_ratio)
                except FileNotFoundError:
                    continue

            caption = model.model_name.replace('_', ' ').replace(' pruning', '').replace('reduced error', 'red err').replace(' val', '').title()
            plot.add_plot(np.array(x)/.8, tr, color=color, legend=caption + ' train')
            plot.add_plot(np.array(x)/.8, ts, 'dashed', color=color, legend=caption + ' test')
        doc += plot
        doc += '\n'

    # doc.build()
    doc.build(delete_files='all', show_pdf=False)
