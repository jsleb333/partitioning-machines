import sys, os
sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split
from datetime import datetime
from graal_utils import Timer
import numpy as np

from experiments.experiment import NoPruning, ReducedErrorPruning, OursShaweTaylorPruning, OraclePruning
from experiments.experiment import Tracker, Logger
from experiments.utils import camel_to_snake, filter_signature
from experiments.datasets.datasets import load_datasets, QSARBiodegradation
from partitioning_machines import func_to_cmd


val_split_ratio = .25


class NoPruningTrVal(NoPruning):
    pass

class NoPruningTr(NoPruning):
    def __init__(self, *, val_split_ratio: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def _prepare_data(self, seed, *args, **kwargs) -> None:
        super()._prepare_data(seed, *args, **kwargs)
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            self.X_tr, self.y_tr,
            test_size=self.val_split_ratio,
            random_state=seed+6
        )

class STPruningTrVal(OursShaweTaylorPruning):
    pass

class STPruningTr(OursShaweTaylorPruning, NoPruningTr):
    pass

class REPruningTr(ReducedErrorPruning):
    pass

class OraclePruningTr(OraclePruning, NoPruningTr):
    pass

class OraclePruningTrVal(OraclePruning):
    pass

experiments_list = [NoPruningTr, NoPruningTrVal, STPruningTr, STPruningTrVal, REPruningTr, OraclePruningTr, OraclePruningTrVal]


@func_to_cmd
def launch_experiment(datasets=list(),
                      model_names=list(),
                      exp_name='',
                      test_split_ratio=.2,
                      val_split_ratio=.25,
                      n_draws=25,
                      n_folds=10,
                      max_n_leaves=50,
                      error_prior_exponent=13.1,
                      seed=42,
                      ):
    """
    Will launch the experiment with specified parameters. Automatically saves all results in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>.csv". Experiments parameters are saved in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>_params.py". See the README for more usage details.

    Args:
        datasets (list[str]):
            The list of datasets to be used in the experiment. By default (an empty list) will iterate over all available datasets. Otherwise, will launch experiment for the specified datasets. To see all available datasets, consult the file "./experiments/datasets/datasets.py".
        model_names (list[str]):
            Valid model names are 'original', 'cart', 'm-cart' and 'ours'. By default 'all' will run all 4 models one after the other.
        exp_name (str):
            Name of the experiment. Will be used to save the results on disk. If empty, the date and time of the beginning of the experiment will be used.
        test_split_ratio (float):
            Ratio of examples that will be kept for test.
        val_split_ratio (float):
            Ratio of examples that will be kept for validation. Will be computed after the split for the test.
        n_draws (int):
            Number of times the experiments will be run with a new random state.
        n_folds (int):
            Number of folds used by the pruning algorithms of CART and M-CART. (Ignored by 'ours' algorithm).
        max_n_leaves (int):
            Maximum number of leaves the original tree is allowed to have.
        error_prior_exponent (int):
            The distribution q_k will be of the form (1-r) * r**k, where r = 2**(-error_prior_exponent). (Ignored by 'cart' and 'm-cart' algorithms).
        seed (int):
            Seed for the random states.
    """
    models = {camel_to_snake(exp.__name__): exp for exp in experiments_list}
    if model_names:
        models = {name: model for name, model in models.items() if name in model_names}

    if not exp_name:
        exp_name = exp_name if exp_name else datetime.now().strftime("%Y-%m-%d_%Hh%Mm")

    for dataset in load_datasets(datasets):
        for model_name, model in models.items():
            with Timer(f'{model_name} model on dataset {dataset.name} with {dataset.n_examples} examples'):
                try:
                    exp_path = f'./experiments/results/{exp_name}/{dataset.name}/'
                    filter_signature(model)(
                        dataset=dataset,
                        exp_name=exp_name,
                        test_split_ratio=test_split_ratio,
                        val_split_ratio=val_split_ratio,
                        n_draws=n_draws,
                        n_folds=n_folds,
                        max_n_leaves=max_n_leaves,
                        error_prior_exponent=error_prior_exponent,
                        seed=seed,
                    ).run(logger=Logger(exp_path), tracker=Tracker())
                except Exception as err:
                    print(f'!!! Unable to complete experiment due to {err!r}!!!')


if __name__ == "__main__":

    with Timer():
        for i, ratio in enumerate(np.linspace(.05, .95, 19)):
            class NoPruningVal(NoPruningTr):
                def __init__(self, *,
                             model_name=f'no_pruning_val={ratio}',
                             **kwargs) -> None:
                    super().__init__(model_name=model_name, **kwargs)

            experiments_list = [NoPruningVal]

            Timer(launch_experiment)(
                model_names=[camel_to_snake(exp.__name__) for exp in experiments_list],
                datasets=['qsar_biodegradation'],
                exp_name='NoP_val_01',
                val_split_ratio=ratio,
                n_draws=10,
            )
