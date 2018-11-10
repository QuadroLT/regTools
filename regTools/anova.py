import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


class Precision:
    """
    Class is designed to carry data and necessary calculations
    for anova calculations in the frame of method validation, homogeniety
    testing and other precission related experiments.
    """
    def __init__(self, data, responds, factor):
        """
        - data: pandas dataframe containing columns with dependant variable
        and categorical net state variable
        - responds: string - dependant variable (column name)
        - factor: string - categorical net state variable (column name)
        """
        self.data = data
        self.responds = responds
        self.factor = factor
        self.anova_table = self._calculate_anova_one_way()
        self.var_table = self._calculate_variances()

    def _calculate_anova_one_way(self):
        """
        Function is called upon instantiation of the class.
        data is used for further calculations
        """
        model = smf.ols(f'{self.responds}~C({self.factor})',
                        data=self.data).fit()
        table = sm.stats.anova_lm(model)
        return table

    def _calculate_variances(self):
        """
        Calculates summary table by factor. Following statistics is calculated:
        count, average, variance. Funtion is called upon instantiation of the
        class.
        """
        return self.data.groupby(self.factor).agg(['count', 'mean', 'var'])

    def get_precision(self):
        """
        Function calculates precision in line with concept given in
        D.L. Massart et. al. Handbook of Chemometrics and Qualimetrics
        Part A, pp. 388. And ISO 5725-2:1994
        """
        # calculation for the number of experiment per factor
        # in-line with ISO 5725-2:1994
        n_factor = len(self.var_table)
        sum_experiment = self.var_table[(self.responds, 'count')].sum()
        sum_experiment_sq = np.sum(self.var_table[(self.responds,
                                                   'count')] ** 2)
        num_exp = (sum_experiment - sum_experiment_sq / sum_experiment) / \
                  (n_factor - 1)
        # rest of procedure by ANOVA
        ms_between, ms_within = self.anova_table.mean_sq
        var_repeat = ms_within
        var_factor = (ms_between - ms_within)/num_exp
        if var_factor < 0:
            var_reprod = var_repeat
        else:
            var_reprod = var_factor + var_repeat
        std_repeat = np.sqrt(var_repeat)
        std_reprod = np.sqrt(var_reprod)

        average = self.data[self.responds].mean()
        cv_repeat = std_repeat / average * 100
        cv_reprod = std_reprod / average * 100

        result = pd.DataFrame({'Average': [average],
                               's_r': [std_repeat],
                               's_R': [std_reprod],
                               'CV_r': [cv_repeat],
                               'CV_R': [cv_reprod],
                               })
        return result
