from tabpfn_client import init, TabPFNRegressor
from .predictor import Predictor
from naslib.predictors.utils.encodings import encode
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        if self.encoding_type != "adjacency_one_hot":
            raise NotImplementedError()

        xtrain = np.array(
            [
                encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                for arch in xtrain
            ]
        )
        logger.info("Fitting TabPFN")
        self.model.fit(xtrain, ytrain)
        logger.info("Fitted TabPFN")

    def predict(self, xtest, info=None):
        if self.encoding_type != "adjacency_one_hot":
            raise NotImplementedError()

        xtest = np.array(
            [
                encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                for arch in xtest
            ]
        )
        logger.info("Predicting with TabPFN")
        output = self.model.predict(xtest)
        logger.info("Predicted with TabPFN")
        return output

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
