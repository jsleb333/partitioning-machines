import sys, os
sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split
from datetime import datetime
from graal_utils import Timer
import numpy as np

from experiments.experiment import NoPruning, ReducedErrorPruning, OursShaweTaylorPruning, OraclePruning
from experiments.main import launch_experiment
from experiments.utils import camel_to_snake


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


if __name__ == "__main__":

    with Timer():
        # for i, ratio in enumerate(np.linspace(.05, .95, 19)):
        for i, ratio in enumerate(np.linspace(.4, .95, 12)):
            class STPruningVal(OursShaweTaylorPruning, NoPruningTr):
                def __init__(self, *,
                             model_name=f'st_pruning_val={ratio:.2f}',
                             **kwargs) -> None:
                    super().__init__(model_name=model_name, **kwargs)

            experiments_list = [STPruningVal]

            Timer(launch_experiment)(
                model_names=[camel_to_snake(exp.__name__) for exp in experiments_list],
                datasets=['qsar_biodegradation'],
                exp_name='test',
                val_split_ratio=ratio,
                n_draws=1,
            )
