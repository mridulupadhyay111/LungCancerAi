"""
trainer/early_stopping.py
"""

import numpy as np


class EarlyStopping:

    def __init__(

        self,

        patience=20,

        delta=0.0,

    ):

        self.patience = patience

        self.delta = delta

        self.best = np.inf

        self.counter = 0

        self.stop = False

    def __call__(self, loss):

        if loss < self.best - self.delta:

            self.best = loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True

        return self.stop