import random
import string
import re

import numpy as np
import pandas as pd
from sklearn.utils import parallel_backend
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class LassoRegression():

    def __init__(self, num_folds=5, verbose=True):
        with(parallel_backend('threading')):
            self.model = LassoCV(cv=num_folds, verbose=verbose)

    def fit(self, train_x, train_y, visualize=True):
        """Fits model and plots results on training data.

        Args:
            train_x: Training features.
            train_y: Training labels.
        """
        self.model = self.model.fit(train_x, train_y)

        if visualize:

            plt.figure()
            ax = plt.subplot(111)

            plt.plot(self.model.alphas_, self.model.mse_path_, ':')
            plt.plot(self.model.alphas_, self.model.mse_path_.mean(axis=-1), 'k',
                     label='Average across the folds', linewidth=2)
            plt.axvline(self.model.alpha_, linestyle='--',
                        color='k', label='CV estimate')
            ax.set_xscale('log')

            plt.legend()
            plt.xlabel('$\lambda$')
            plt.ylabel('MSE')
            plt.axis('tight')
            plt.show()

    def test(self, test_x, test_y, visualize=True):
        prediction = self.model.predict(test_x)

        if visualize:
            plt.figure()
            plt.plot(test_y, prediction, '.', alpha=0.1)
            plt.title('RMSE: %f'%np.sqrt(np.mean( (prediction - test_y)**2 )))
            plt.show()
    