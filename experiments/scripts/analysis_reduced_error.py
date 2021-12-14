import python2latex as p2l
import numpy as np
import pandas as pd
import os

exps = {
    'NoP_val_01': {
        'name': 'no_pruning',
        'caption': 'No pruning'
        },
    'RE_val_01':{
        'name': 'reduced_error_pruning',
        'caption': 'Reduced error pruning'
        },
    'ST_val_01':{
        'name': 'st_pruning',
        'caption': 'Shawe-Taylor pruning (ours)'
        },
}

def extract_data(prepend):
    ratios = np.linspace(0.05, 0.95, 19)
    tr, ts = [], []

    for ratio in ratios:
        str_ratio = f'{ratio:.2f}'
        filepath = f'{prepend}{str_ratio}.csv'
        if not os.path.exists(filepath):
            filepath = f'{prepend}{str_ratio[:-1]}.csv'

        df = pd.read_csv(filepath, sep=',', header=0)

        tr.append(df['train_accuracy'].to_numpy().mean())
        ts.append(df['test_accuracy'].to_numpy().mean())

    return ratios, tr, ts


def plot_single_exp(exp_name='RE_val_01'):
    plot = p2l.Plot(plot_name=exp_name,
                    plot_path='./experiments/results/re_results/'+exp_name,
                    height='.25\\textheight')

    name = exps[exp_name]['name']
    caption = exps[exp_name]['caption']

    prepend = f'./experiments/results/{exp_name}/qsar_biodegradation/{name}_val='

    ratios, tr, ts = extract_data(prepend)

    plot.add_plot(ratios, tr, legend='Train acc.')
    plot.add_plot(ratios, ts, legend='Test acc.')
    plot.legend_position = 'south west'

    plot.y_min = 0.6
    plot.y_max = 1
    plot.y_label = 'Accuracy'
    plot.x_min = 0
    plot.x_max = 1
    plot.x_label = 'Validation ratio'

    return plot


def plot_all_exps():
    plot = p2l.Plot(plot_name='all_exps',
                    plot_path='./experiments/results/re_results/',
                    height='.5\\textheight')

    for (exp_name, exp), color in zip(exps.items(), p2l.holi):
        name = exp['name']
        caption = exp['caption']
        prepend = f'./experiments/results/{exp_name}/qsar_biodegradation/{name}_val='

        ratios, tr, ts = extract_data(prepend)

        plot.add_plot(ratios, tr, legend=caption + ' train', color=color)
        plot.add_plot(ratios, ts, 'dashed', legend=caption + ' test', color=color)

    plot.legend_position = 'south west'

    plot.y_min = 0.6
    plot.y_max = 1
    plot.y_label = 'Accuracy'
    plot.x_min = 0
    plot.x_max = 1
    plot.x_label = 'Validation ratio'

    return plot


if __name__ == '__main__':
    doc = p2l.Document(filename='comp_re', filepath='./experiments/results/re_results/')
    doc += plot_all_exps()

    doc.build(delete_files='all', show_pdf=False)
