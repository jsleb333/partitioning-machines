import numpy as np
import csv
from time import time
from datetime import datetime

import sys, os
sys.path.append(os.getcwd())

from experiments.datasets.datasets import Dataset
from experiments.utils import Mock, get_default_kwargs
from experiments.models import Model, model_dict


class Logger:
    def __init__(self, exp_path) -> None:
        self.exp_path = exp_path

        os.makedirs(self.exp_path, exist_ok=True)
        self.is_closed = True

    def dump_exp_config(self, model_name: str, exp_config: dict) -> None:
        with open(self.exp_path + f'{model_name}_exp_config.py', 'w') as file:
            file.write(f"exp_config = { {k:str(v) for k, v in exp_config.items()} }")

    def prepare_csv_file(self, model_name: str) -> None:
        self.file = open(self.exp_path + model_name + '.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.is_closed = False

    def dump_row(self, row: dict, model_name: str) -> None:
        if self.is_closed:
            self.prepare_csv_file(model_name)
            self._dump_row(row.keys())
        self._dump_row(row.values())

    def _dump_row(self, row: list) -> None:
        self.csv_writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
        self.is_closed = True


class Tracker:
    def start(self, model_name):
        self.times = []
        print(f'Running model {model_name.replace("_", " ")}')

    def display_progress_before(self, draw: int) -> None:
        print(f'Running draw #{draw:02d}...', end='\r')
        self.draw_start = time()

    def display_progress_after(self, draw: int) -> None:
        self.times.append(time() - self.draw_start)
        print(f'Running draw #{draw:02d}...' + self._mean_time_per_draw(), end='\r')

    def _mean_time_per_draw(self):
        return f'\tMean time per draw: {sum(self.times)/len(self.times):.3f}s.' if self.times else ''

    def end(self, draw: int) -> None:
        print(f'\rCompleted all {draw+1} draws.' + self._mean_time_per_draw())


class Experiment:
    def __new__(cls, *args, **kwargs):
        new_exp = super().__new__(cls)
        new_exp.config = get_default_kwargs(cls) | kwargs | {'datetime': datetime.now()}
        model = new_exp.config['model']
        new_exp.config['model_config'] = model.config
        return new_exp

    def __init__(self, *,
                 dataset: Dataset,
                 model: Model,
                 val_split_ratio: float = 0,
                 test_split_ratio: float = .2,
                 n_draws: int = 25,
                 seed: int = 42,
                 exp_name: str = None
                 ):
        self.dataset = dataset
        self.model = model
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.n_draws = n_draws
        self.exp_name = exp_name
        self.rng = np.random.RandomState(seed)

    def run(self,
            *args,
            logger: Logger = Mock(),
            tracker: Tracker = Mock(),
            **kwargs) -> None:

        logger.dump_exp_config(self.model_name, self.config)
        tracker.start(self.model_name)

        for draw in range(self.n_draws):
            tracker.display_progress_before(draw)
            metrics = self._run(draw, *args, **kwargs)
            tracker.display_progress_after(draw)
            logger.dump_row(metrics, self.model_name)

        tracker.end(draw)
        logger.close()

    def _run(self, draw: int, *args, **kwargs) -> dict:
        draw_seed = self.rng.randint(2**31)

        self.dataset(self.val_split_ratio,
                     self.test_split_ratio,
                     shuffle=draw_seed)

        self.fit_tree(self.dataset)

        t_start = time()
        self.prune_tree(self.dataset)
        elapsed_time = time() - t_start

        acc_tr, acc_val, acc_ts = self.evaluate_tree(self.dataset)

        metrics = {'draw': draw,
                   'seed': draw_seed,
                   'train_accuracy': acc_tr,
                   'test_accuracy': acc_ts,
                   'n_leaves': self.model.tree.n_leaves,
                   'height': self.model.tree.height,
                   'bound': self.model.bound_value,
                   'time': elapsed_time}
        if acc_val is not None:
            metrics['val_accuracy'] = acc_val

        return metrics


if __name__ == '__main__':
    from datasets.datasets import Iris, Wine
    for model in model_dict.values():
        exp = Experiment(dataset=Iris,
                         model=model(),
                         val_split_ratio=.1)
    # # for exp in [OursShaweTaylorPruning]:
    # # for exp in [OursHypInvPruning]:
    # for exp in [KearnsMansourPruning]:
    #     e = exp(dataset=Iris, n_draws=2)
    #     e.run(tracker=Tracker(), logger=Logger(exp_path='./experiments/results/test/'))
