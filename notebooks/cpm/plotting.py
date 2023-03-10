import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr,spearmanr

def plot_predictions(behav_obs_pred, tail="glm", figsize=(10,10), color='gray', font_scale=2, verbose=False, accuracy_metric='pearson', p_value=None, ax=None, marker_size=1, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if ax is None:
        create_new_fig = True
    else:
        create_new_fig = False
    """
    This function generates a scatter plot of observed behavior vs. predicted behavior. The plot also includes
    a linear fit with 9% confidence intervals, as computed by seaborn.regplot.
    
    INPUTS
    ======
    behav_obs_pred: pd.Dataframe structure with as many rows as scans and 4 columns corresponding to the predictions
                    using the three models (pos,neg and glm) and the observed behavior. (#scans X 4)
                    
    tail: model for which we want to plot the results. Possible values ('pos','neg','glm'). [default = 'glm']
    
    figsize: plot size. [default = (10,10)]
    
    color: color for scatter and fitted line. [default='gray']
    
    font_scale: size of labels and annotations. [default=2]
    
    verbose: if true, it will print the p and r values for the linear fit.
    
    OUTPUTS
    =======
    r: correlation value between predictive and observed values.
    
    p: p-value for the correlatin
    
    f: figure
    """
 
    # Extract observed (x) and predicted behavior for the model of interest (tail)
    x                         = behav_obs_pred.filter(regex=("obs")).astype(float)
    y                         = behav_obs_pred.filter(regex=(tail)).astype(float)
    x_values                  = x.values.flatten()
    y_values                  = y.values.flatten()
    # Find location of nan prediction (i.e., non-existent model) and non-nan predictions (i.e., model available)
    nan_predictions           = np.isnan(y_values)
    no_nan_predictions        = ~nan_predictions
    num_nan_predictions       = nan_predictions.sum()>0
    
    if num_nan_predictions>0:
        print("++ WARNING: Number of nan predictions = %d" % num_nan_predictions)
    
    # Select only scans with non-nan predictions
    # ==========================================
    x_values = x_values[no_nan_predictions]
    y_values = y_values[no_nan_predictions]
    # If no data is available set r = 0 and p = 1
    # ===========================================
    if p_value is None:
        if y_values.shape[0] == 0:
            r,p_value = 0,1
        else:
            if accuracy_metric == 'pearson':
                 r,p_value = pearsonr(x_values,y_values)
            if accuracy_metric == 'spearman':
                 r,p_value = spearmanr(x_values, y_values)
    else:
        if y_values.shape[0] == 0:
            r,p_value = 0,1
        else:
            if accuracy_metric == 'pearson':
                 r,_ = pearsonr(x_values,y_values)
            if accuracy_metric == 'spearman':
                 r,_ = spearmanr(x_values, y_values)
    # Create Plot
    # ===========
    if create_new_fig:
        f,ax = plt.subplots(1,1,figsize=figsize)
    sns.set(font='Helvetica')
    sns.set(font_scale=font_scale)
    sns.set(style='whitegrid')
    g = sns.regplot(x=x.T.squeeze(), y=y.T.squeeze(), color=color, ax=ax, scatter_kws={'s':marker_size})
    ax_min = min(min(g.get_xlim()), min(g.get_ylim()))
    ax_max = max(max(g.get_xlim()), max(g.get_ylim()))
    g.set_xlim(ax_min, ax_max)
    g.set_ylim(ax_min, ax_max)
    g.set_aspect('equal', adjustable='box')
    if verbose:
        print(r,p_value)
    g.annotate('r = {0:.2f} | p={1:.2e}'.format(r,p_value), xy = (0.3, 0.1), xycoords = 'axes fraction');
    # Apply requested configurations
    # ==============================
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if create_new_fig:
        plt.close(f)
        return r, p_value, f
    else:
        return r, p_value