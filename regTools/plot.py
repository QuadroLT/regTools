# import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

def plot_model(reg, **kwargs):
    """
    function for plotting regression model: data, fit and confidence intervals
    takes object from 'Model' class.
    additional key word argumennts to be passed:
    -- xlabel: string for defining x axis label (may accept latex expressions)*
    -- ylabel: string for defining y axis label (may accept latex expressions)*

    * see matplotlib documentation for details
    """
    xmin = np.min(reg.x)
    xmax = np.max(reg.x)
    c, d = reg.get_weights()
    model = reg.reg_model
    xs = np.linspace(xmin, xmax, num=100)
    rsds = (c + d * xs) / xs * 100
    wxs = reg.x / (c + d * reg.x)
    ys = model.params[1] * xs + model.params[0]
    plt.figure(figsize=(15, 10))
    # plot regression line
    plt.subplot(212)
    plt.scatter(reg.x, reg.y, color='k')
    plt.plot(xs, ys, color = 'b')
    plt.xlabel('Concentration')
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    plt.ylabel('Responds')
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.title('Calibration curve')
    # plot precision curve
    plt.subplot(221)
    plt.plot(xs, rsds)
    plt.xlabel('Concentration')
    plt.ylabel('RSD, %')
    plt.title('Standard deviations')
    # plot residuals
    plt.subplot(222)
    plt.scatter(reg.x,  model.resid/(c+d*reg.x))
    plt.xlabel('net state variable')
    plt.ylabel('residuals')
    plt.show()
