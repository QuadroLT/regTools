import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import levene


import abc

import os


class NewClass:
    __metaclass__ = abc.ABCMeta

    def __new__()
    pass


class RegModel(object):
    """
    TODO extend docstring here.
    Implementation of linear regression model applied
    for multilevel calibrations in chemistry
    the requirement for the data is at least two repetitions
    per level of experiment.
    """
    def __init__(self, data, exog, endog, model):

        """
        designed to work with pandas dataframes
        exog - sting; column name for exogeneous
        (net state variable, also called x)
        endog - string, column name for endogeneous
        (dependant variable, also called y)
        model - string to define linear fit 
        possible choices are:
        -- WLS: weighed least squares
        -- OLS: ordinary least squares
        """
        self.model = model
        self.data = data
        self.endog = endog
        self.exog = exog
        self.x = data[exog].values
        self.y = data[endog].values
        self.xs = data[exog].drop_duplicates().values
        self.reg_model = self.fit()

    def is_homoscedastic(self):
        """
        checks for homogeniety of variance in the given
        dataset. Levene test implemented

        returns p-value

        """
        temp = self.data
        xs = []
        ys = []
        for item in self.xs:
            y = temp[self.endog].loc[temp[self.exog]==item].values
            ys.append(y)
        ys = np.array(ys)
        stats, p_value = levene(*ys, center='mean')
        #print(stats, p_value)
        if p_value < 0.05:
            return False
        else:
            return True


    def get_weights(self, rtol=0.000000000000001):
        """
        Implementation of ISO 11843-2:2002 case II: variance is lineary
        dependant on the net state variable
        math::
        ($\sigma = \hat{c} + \hat{d}x$).
        the function returns regression parameters.
        rtol  - tolerance for convergence
        """
        temp = self.data
        xs = temp[self.exog].drop_duplicates().values
        c, d = (0, 0)
        i = 0
        converge = False
        while not converge:
            stdevs = temp.groupby(self.exog).std()
            if i == 0:
                stds =  (setenv "WORKON_HOME" "/Users/vtamosiunas/anaconda3/bin")
  (pyvenv-mode 1) stdevs[self.endog].values
                sigma = stds
            else:
                #stds = stdevs['x_weighed'].values
                sigma = c + d * xs
            wt = 1 / (sigma ** 2)
            t1 = np.sum(wt)
            t2 = np.sum(wt*xs)
            t3 = np.sum(wt*(xs**2))
            t4 = np.sum(wt*stds)
            t5 = np.sum(wt*stds*xs)
            # calculates coeficients
            c1 = (t3 * t4 - t2 * t5) / (t1 * t3 - t2 ** 2)
            d1 = (t1 * t5 - t2 * t4) / (t1 * t3 - t2 ** 2)
            # check for convergence
            converge = np.isclose(c-c1, 0,
                                  atol=rtol) and np.isclose(d-d1, 0,
                                                            atol=rtol)
            # writing new coeficients
            c, d = c1, d1
        i += 1
        # print(c, d)
        return c, d

    def fit(self, const=True):
        """
        Fits regression model dependant on choise
        - const: True, False; True assumes intercept
        is important. False pushes regression through
        zero.
        """
        y = self.y
        if const:
            x = sm.add_constant(self.x)
        else:
            x = self.x

        if self.model == 'OLS':
            reg_model = sm.OLS(y, x).fit()
            return reg_model
        elif self.model == 'WLS':
            c, d = self.get_weights()
            weights = (c + d * self.x)
            reg_model = sm.WLS(y, x, weights=1/(weights**2)).fit()
            print(reg_model.summary())
            print(reg_model.conf_int())
            return reg_model
        else:
            print('wrong argument')
            return 0
    def get_conf_int(self, x_new):
        """
        calculates confidence interval for new x values
        implemented usin

        Arguments:
        - x_new: array of numbers.
        """


        pass

    def lod(self):
        pass
import 
