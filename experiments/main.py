"""
This file contains the code necessary to run all experiments of the paper 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020). See the README for usage details.
"""
import sys, os

sys.path.append(os.getcwd())
from datetime import datetime
from graal_utils import Timer

from experiments.models import model_dict
from experiments.experiment import Experiment, Tracker, Logger
from experiments.utils import filter_signature
from experiments.datasets import load_datasets, dataset_list
from partitioning_machines import func_to_cmd


@func_to_cmd
def launch_experiment(datasets=list(),
                      model_names=list(),
                      exp_name='',
                      test_split_ratio=.2,
                      val_split_ratio=.2,
                      n_draws=25,
                      n_folds=10,
                      max_n_leaves=50,
                      error_prior_exponent=13.1,
                      delta=.05,
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
        delta (float):
            TODO
        seed (int):
            Seed for the random states.
    """
    models = model_dict
    if model_names:
        models = {name: model for name, model in model_dict.items() if name in model_names}
        for name in model_names:
            if name not in model_dict:
                print('Unknown model: ', name)

    if not exp_name:
        exp_name = exp_name if exp_name else datetime.now().strftime("%Y-%m-%d_%Hh%Mm")

    for dataset in datasets:
        if dataset not in [d.name for d in dataset_list]:
            print('Unknown dataset: ', dataset)

    for dataset in load_datasets(datasets):
        for model_name, model in models.items():
            with Timer(f'{model_name} model on dataset {dataset.name} with {dataset.n_examples} examples'):
                try:
                    init_model = filter_signature(model)(
                        max_n_leaves=max_n_leaves,
                        error_prior_exponent=error_prior_exponent,
                        delta=delta,
                        n_folds=n_folds,
                        val_split_ratio=val_split_ratio,
                    )
                    exp_path = f'./experiments/results/{exp_name}/{dataset.name}/'
                    Experiment(
                        dataset=dataset,
                        model=init_model,
                        test_split_ratio=test_split_ratio,
                        n_draws=n_draws,
                        seed=seed,
                    ).run(logger=Logger(exp_path), tracker=Tracker())
                except Exception as err:
                    print(f'!!! Unable to complete experiment due to {err!r}!!!')
                    raise


if __name__ == "__main__":
    launch_experiment(
        model_names=['reduced_error_pruning', 'ours_shawe_taylor_pruning', 'oracle_pruning'],
        # datasets=['iris'],
        datasets=['cardiotocography10'],
        exp_name='card10-03',
        n_draws=25,
        seed=101,
        max_n_leaves=100,
        val_split_ratio=.16,
    )
