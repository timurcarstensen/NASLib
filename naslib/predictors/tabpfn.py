import numpy as np
from tabpfn_client import TabPFNRegressor, init

from naslib.predictors.utils.encodings import encode

from .predictor import Predictor


class TabPFN(Predictor):
    def __init__(
        self,
        encoding_type="adjacency_one_hot",
        ss_type=None,
        hpo_wrapper=False,
    ):
        super().__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        init()
        self.model = TabPFNRegressor()

    def fit(self, xtrain, ytrain, train_info=None):

        xtrain = np.array(
            [
                encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                for arch in xtrain
            ]
        )

        if isinstance(xtrain, list):
            xtrain = np.array(xtrain)
        if isinstance(ytrain, list):
            ytrain = np.array(ytrain)

        self.model.fit(xtrain, ytrain)

    def query(self, xtest, info=None):
        if type(xtest) is list:
            #  when used in itself, we use
            xtest = np.array(
                [
                    encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                    for arch in xtest
                ]
            )

        return np.squeeze(self.model.predict(xtest))

    def predict(self, xtest, info=None):
        if self.encoding_type != "adjacency_one_hot":
            raise NotImplementedError()

        xtest = np.array(
            [
                encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                for arch in xtest
            ]
        )
        return np.squeeze(self.model.predict(xtest))

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/predict.
        """
        reqs = {
            "requires_partial_lc": False,
            "metric": None,
            "requires_hyperparameters": False,
            "hyperparams": None,
            "unlabeled": False,
            "unlabeled_factor": 0,
        }
        return reqs
