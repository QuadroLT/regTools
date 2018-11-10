import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import gls, ols
from scipy.stats import t
from scipy.stats import levene


class Regression:
    """
    class intended for data storage and regression analysis in chemical
    measurements

    following methods implemented:
       - levene test for heterocedasticity
       - regression fit according to:
            - weighed least square regression model assuming variance is
              lineary dependant on the net state variable
            - ordinary least square regression model
       - detection capability according to ISO 11843-2:2004
       - inverse fitting of respondses of unknown samples to calculate result
         and the error with respect to confidence intervals given

    """

    def __init__(self, data, x, y, alpha=0.05):

        """
        - data: pandas dataframe with dependant and net state variables. At
        least two measurements per concentration are necessary.

        - x: string -  net state variable (column name)
        - y: string - dependant variable (column name)
        - alpha: float - error probability (default, 0.05)
        """

        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        self.model = self._fit()

    def _levene_test(self):
        """
        The method is class private.
        check for homogeniety of variance of residuals.
        returns p-value of test
        """

        model = ols(f'{self.y} ~ {self.x}', data=self.data).fit()
        xs = self.data[self.x]
        df = pd.DataFrame({'x': xs, 'y': model.resid})
        resids = []
        for x in df.x.drop_duplicates().values:
            resids.append(df.y.loc[df.x == x].values)

        w, p = levene(*resids)
        return p

    def _get_weights(self):
        """
        Internal method for weights estimation from linear regression. Assuming
        variance of responds lineary depends on the net state variable based on
        ISO 11843-2:2002.

        returns coeficients for weighs estimation for the calibration range.
        """

        df = self.data[[self.x, self.y]]
        df = df.groupby(self.x).std()
        x = df.index.values
        y = df[self.y].values
        xs = sm.add_constant(x)
        model = sm.WLS(y, xs, weights=1/y**2).fit()
        c, d = model.params
        converge = False
        while not converge:
            weights = (c + d * x)**2
            model = sm.WLS(y, xs, weights=1/weights).fit()
            c1, d1 = model.params
            converge = (np.isclose(c, c1) and np.isclose(d, d1))
            c, d = c1, d1
        return c, d

    def _fit(self):
        """
        makes model fit. the model is fitted according to the outcome of levene
        test. GLS is used dependent on the outcome of the test weight matrix is
        calculated.


        returns statsmodels gls object. For further manipulation with object
        see statsmodels documentation.

        """
        p = self._levene_test()
        if p > 0.05:
            self.weights = np.repeat(1, len(self.data[self.x]))
            print('Variance is homogenious, assuming OLS')
            self.c, self.d = 1, 1
            self.model_type = 'OLS'
        else:
            print('Variance is not homogenious, assuming WLS')
            self.model_type = 'WLS'
            self.c, self.d = self._get_weights()
            self.weights = (self.c + self.d * self.data[self.x]) ** 2
        model = gls(f'{self.y}~{self.x}', data=self.data, sigma=self.weights).fit()

        return model

    def criticalValues(self, replicates=1, new_x=None):
        """
        function to calculate confidence intervals for regression estimate.
        Function calculates both, confidence and prediction intervals based on
        new net state variable provided. returned data is arranged in dataframe
        format.
        Arguments:
            - new_x: numpy array of net state variable. If not provided,
            CI's and PI's are calculated for the initial calibration range.
            - replicates: int - number of preparations for new net state
            variable. (if the result is expected to be predicted from the
            mean of "n" replicates, then "n" shall be passed as the variable
            replicates)

        """
        try:
            if not new_x:

                new_x = np.linspace(
                    np.min(self.data[self.x]),
                    np.max(self.data[self.x]),
                    100)

        except ValueError:
            new_x = new_x

        x = self.data[self.x]

        if self.model_type == 'WLS':
            sigma0 = self.c ** 2
            new_weights = 1 / ((self.c + self.d * new_x) ** 2)

        else:
            sigma0 = 1
            new_weights = np.repeat(1, len(new_x))
            new_weights = 1 / new_weights

        weights = 1 / self.weights
        weighed_x = x * weights
        sum_weights = np.sum(weights)
        weighed_new_x = new_x * new_weights
        x_dist_sq = (weighed_new_x - weighed_x.mean()) ** 2
        x_var = np.sum((weights*(x - weighed_x.mean()) ** 2))
        new_y = self.model.params[0] + self.model.params[1] * new_x
        t_val = t.ppf(1-self.alpha, len(x)-2)

        sigma = np.sum((self.model.resid ** 2) * weights) / (len(x) - 2)

        pi_base = (sigma0 / replicates) + (1/sum_weights + x_dist_sq / x_var) * sigma
        ci_base = (1/sum_weights + x_dist_sq / x_var) * sigma

        pi_plus = new_y + t_val * np.sqrt(pi_base)
        pi_minus = new_y - t_val * np.sqrt(pi_base)
        ci_plus = new_y + t_val * np.sqrt(ci_base)
        ci_minus = new_y - t_val * np.sqrt(ci_base)
        df = pd.DataFrame({'x': new_x,
                           'y': new_y,
                           'ci_plus': ci_plus,
                           'ci_minus': ci_minus,
                           'pi_plus': pi_plus,
                           'pi_minus': pi_minus})

        return df

    def confidence_plot(self, replicates=1, new_x=None):
        """
        function to calculate confidence intervals for regression estimate.
        Function calculates both, confidence and prediction intervals based on
        new net state variable provided. returned data is arranged in dataframe
        format.
        Arguments:
            - new_x - numpy array of net state variable. If not provided,
            CI's and PI's are calculated for the initial calibration range.

        """
        try:
            if not new_x:

                new_x = np.linspace(
                    np.min(self.data[self.x]),
                    np.max(self.data[self.x]),
                    100,
                )
        except ValueError:
            new_x = new_x

        n = len(self.data[self.y])
        mse = np.sum(self.model.resid**2) / (n - 2)
        #print(mse)
        # sum of squares x
        ss_x = np.sum((self.data[self.x] - np.mean(self.data[self.x])) ** 2)
        # squared diffirence for new x
        s_xx = (new_x - np.mean(self.data[self.x])) ** 2
        y_hat = self.model.params[0] + self.model.params[1] * new_x
        ci_base = t.ppf(1-self.alpha, n-2) * np.sqrt((mse*(1/n + s_xx/ss_x)))
        ci_plus = y_hat + ci_base
        ci_minus = y_hat - ci_base

        pi_base = t.ppf(1-self.alpha, n-2)\
            * np.sqrt((mse*(1 + 1/n + s_xx/ss_x)))
        pi_plus = y_hat + pi_base
        pi_minus = y_hat - pi_base

        df = pd.DataFrame({'x': new_x, 'ci_plus': ci_plus,
                           'ci_minus': ci_minus, 'pi_plus': pi_plus,
                           'pi_minus': pi_minus})

        return df

    # def confidence(self, new_x=None, replicates=1):
    #     if self.model == 'WLS':
    #         df = self._get_confidence_intervals_wls(new_x=new_x, replicates=replicates)
    #     else:
    #         df = self._get_conidence_intervals_ols(new_x=new_x, replicates=replicates)
    # return df


    def inverse_fit(self, measurements, ys, labels):
        """
        function calculates measurement result value, based on the calibration
        given. The result is then expressed as result and uncertainty, derived
        from the prediction interval of the calibration function.
        Arguments:
            - measurements: pandas dataframe containing:
                - ys: string - column for measurement values
                - labels: string - columns containing measured samples IDs
        """
        raise NotImplementedError
