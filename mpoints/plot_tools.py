import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import seaborn
except Exception as e:
    print('WARNING: Could not import seaborn')
try:
    import statsmodels.tsa.stattools as stattools
except Exception as e:
    print('WARNING: Could not import statsmodels')
from matplotlib.colors import ListedColormap
import copy
import bisect


def qq_plot(residuals, shape=None, path='', fig_name='qq_plot.pdf', log=False, q_min=0.01, q_max=0.99,
            number_of_quantiles=100, title=None, labels=None, model_labels=None, palette=None, figsize=(12, 6),
            size_labels=16, size_ticks=14, legend_size=16, bottom=0.12, top=0.93, left=0.08, right=0.92, savefig=False,
            leg_pos=0):
    """
    Qq-plot of residuals.

    :type residuals: list
    :param residuals: list of lists (one list of residuals per event type) or list of lists of lists when multiple models are compared (one list of lists per model).
    :type shape: (int, int)
    :param shape: 2D-tuple (number of rows, number of columns), shape of the array of figures.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type log: boolean
    :param log: set to True for qq-plots with log-scale.
    :type q_min: float
    :param q_min: smallest quantile to plot (e.g., 0.01 for 1%).
    :type q_max: float
    :param q_max: largest quantile to plot.
    :type number_of_quantiles: int
    :param number_of_quantiles: number of points used to plot.
    :type title: string
    :param title: suptitle.
    :type labels: list of strings
    :param labels: labels of the event types.
    :type model_labels: list of strings
    :param model_labels: names of the different considered models.
    :type palette: list of colours
    :param palette: color palette, one color per model.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_ticks: int
    :param size_ticks: fontsize of tick labels.
    :type legend_size: int
    :param legend_size: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type leg_pos: int
    :param leg_pos: position of the legend in the array of figures.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    quantile_levels = np.linspace(q_min, q_max, number_of_quantiles)
    quantiles_theoretical = np.zeros(number_of_quantiles)
    for i in range(number_of_quantiles):
        q = quantile_levels[i]
        x = - np.log(1 - q)  # standard exponential distribution
        quantiles_theoretical[i] = x
    # find number of models given and number of event types (dim)
    n_models = 1
    dim = len(residuals)
    # case when there is more than one model
    if type(residuals[0][0]) in [list, np.ndarray]:
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels == None:
        model_labels = [None]*n_models
    if shape is None:
        shape = (1, dim)
    v_size = shape[0]
    h_size = shape[1]
    if palette == None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(
        v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            n = j + h_size * i
            if n < dim:  # the shape of the subplots might be bigger than dim, i.e. 3 plots on a 2x2 grid.
                axes = None
                if v_size == 1 and h_size == 1:
                    axes = fig_array
                elif v_size == 1:
                    axes = fig_array[j]
                elif h_size == 1:
                    axes = fig_array[i]
                else:
                    axes = fig_array[i, j]
                # font size for tick labels
                axes.tick_params(axis='both', which='major',
                                 labelsize=size_ticks)
                if n_models == 1:
                    quantiles_empirical = np.zeros(number_of_quantiles)
                    for k in range(number_of_quantiles):
                        q = quantile_levels[k]
                        x = np.percentile(residuals[n], q * 100)
                        quantiles_empirical[k] = x
                    axes.plot(quantiles_theoretical,
                              quantiles_empirical, color=palette[0])
                    axes.plot(quantiles_theoretical, quantiles_theoretical,
                              color='k', linewidth=0.8, ls='--')
                else:
                    for m in range(n_models):
                        quantiles_empirical = np.zeros(number_of_quantiles)
                        for k in range(number_of_quantiles):
                            q = quantile_levels[k]
                            x = np.percentile(residuals[m][n], q * 100)
                            quantiles_empirical[k] = x
                        axes.plot(quantiles_theoretical, quantiles_empirical, color=palette[m],
                                  label=model_labels[m])
                        if m == 0:
                            axes.plot(quantiles_theoretical, quantiles_theoretical, color='k', linewidth=0.8,
                                      ls='--')
                    if n == leg_pos:  # add legend in the specified subplot
                        legend = axes.legend(frameon=1, fontsize=legend_size)
                        legend.get_frame().set_facecolor('white')
                if log:
                    axes.set_xscale('log')
                    axes.set_yscale('log')
                if labels is not None:
                    axes.set_title(labels[n], fontsize=size_labels)
    plt.tight_layout()
    if bottom != None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    f.text(0.5, 0.02, 'Quantile (standard exponential distribution)',
           ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Quantile (empirical)', va='center',
           rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array


def correlogram(residuals, path='', fig_name='correlogram.pdf', title=None, labels=None, model_labels=None,
                palette=None, n_lags=50, figsize=(8, 6), size_labels=16, size_ticks=14, size_legend=16, bottom=None,
                top=None, left=None, right=None, savefig=False):
    """
    Correlogram of residuals.

    :type residuals: list
    :param residuals: list of lists (one list of residuals per event type) or list of lists of lists when multiple models are compared (one list of lists per model).
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type title: string
    :param title: suptitle.
    :type labels: list of strings
    :param labels: labels of the event types.
    :type model_labels: list of strings
    :param model_labels: names of the different considered models.
    :type palette: list of colours
    :param palette: color palette, one color per model.
    :type n_lags: int
    :param n_lags: number of lags to plot.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_ticks: int
    :param size_ticks: fontsize of tick labels.
    :type legend_size: int
    :param legend_size: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    # find number of models given and number of event types (dim)
    n_models = 1
    dim = len(residuals)
    # case when there is more than one model
    if type(residuals[0][0]) in [list, np.ndarray]:
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels is None:
        model_labels = [None] * n_models
    v_size = dim
    h_size = dim
    if palette is None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(
        v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            axes = None
            if v_size == 1 and h_size == 1:
                axes = fig_array
            elif v_size == 1:
                axes = fig_array[j]
            elif h_size == 1:
                axes = fig_array[i]
            else:
                axes = fig_array[i, j]
            # font size for tick labels
            axes.tick_params(axis='both', which='major', labelsize=size_ticks)
            if n_models == 1:
                max_length = min(len(residuals[i]), len(residuals[j]))
                ccf = stattools.ccf(np.array(residuals[i][0:max_length]),
                                    np.array(residuals[j][0:max_length]),
                                    unbiased=True)
                axes.plot(ccf[0:n_lags+1], color=palette[0])
                axes.set_xlim(xmin=0, xmax=n_lags)
            else:
                for m in range(n_models):
                    max_length = min(
                        len(residuals[m][i]), len(residuals[m][j]))
                    ccf = stattools.ccf(np.array(residuals[m][i][0:max_length]),
                                        np.array(
                                            residuals[m][j][0:max_length]),
                                        unbiased=True)
                    axes.plot(ccf[0:n_lags + 1], color=palette[m],
                              label=model_labels[m])
                    axes.set_xlim(xmin=0, xmax=n_lags)
                if i+j == 0:  # only add legend in the first subplot
                    legend = axes.legend(frameon=1, fontsize=size_legend)
                    legend.get_frame().set_facecolor('white')
            if labels is not None:
                axes.set_title(labels[i] + r'$\rightarrow$' +
                               labels[j], fontsize=size_labels)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if bottom != None:
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    f.text(0.5, 0.025, 'Lag', ha='center', fontsize=size_labels)
    f.text(0.015, 0.5, 'Correlation', va='center',
           rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array


def transition_probabilities(probabilities, shape=None, path='', fig_name='transition_probabilities.pdf',
                             events_labels=None, states_labels=None, title=None, color_map=None, figsize=(12, 6),
                             size_labels=16, size_values=14, bottom=0.1, top=0.95, left=0.08, right=0.92,
                             wspace=0.2, hspace=0.2,
                             savefig=False, usetex=False):
    """
    Annotated heatmap of the transition probabilities of a state-dependent Hawkes process.

    :type probabilities: 3D array
    :param probabilities: the transition probabilities.
    :type shape: (int, int)
    :param shape: 2D-tuple (number of rows, number of columns), shape of the array of figures.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type events_labels: list of strings
    :param events_labels: labels of the event types.
    :type states_labels: list of strings
    :param states_labels: labels of the states.
    :type title: string
    :param title: suptitle.
    :param color_map: color map for the heatmap, see seaborn documentation.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of the annotations on top of the heatmap.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type wspace: float
    :param wspace: horizontal spacing between the subplots, see matplotlib subplots_adjust.
    :type hspace: float
    :param hspace: vertical spacing between the subplots, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type usetex: boolean
    :param usetex: set to True if matplolib figure is rendered with TeX.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    if color_map is None:
        color_map = seaborn.cubehelix_palette(
            as_cmap=True, reverse=False, start=0.5, rot=-.75)
    number_of_states = np.shape(probabilities)[0]
    number_of_event_types = np.shape(probabilities)[1]
    if shape is None:
        v_size = 1
        h_size = number_of_event_types
    else:
        v_size = shape[0]
        h_size = shape[1]
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize)
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            n = i*h_size + j
            if n < number_of_event_types:  # we could have more subplots than event types
                axes = None
                if v_size == 1 and h_size == 1:
                    axes = fig_array
                elif v_size == 1:
                    axes = fig_array[j]
                elif h_size == 1:
                    axes = fig_array[i]
                else:
                    axes = fig_array[i, j]
                # font size for tick labels
                axes.tick_params(axis='both', which='major',
                                 labelsize=size_labels)
                # Create annotation matrix
                annot = np.ndarray(
                    (number_of_states, number_of_states), dtype=object)
                for x1 in range(number_of_states):
                    for x2 in range(number_of_states):
                        p = probabilities[x1, n, x2]
                        if p == 0:
                            if usetex:
                                annot[x1, x2] = r'$0$\%'
                            else:
                                annot[x1, x2] = r'0%'
                        elif p < 0.01:
                            if usetex:
                                annot[x1, x2] = r'$<1$\%'
                            else:
                                annot[x1, x2] = r'<1%'
                        else:
                            a = str(int(np.floor(100 * p)))
                            if usetex:
                                annot[x1, x2] = r'$' + a + r'$\%'
                            else:
                                annot[x1, x2] = a + r'%'
                seaborn.heatmap(probabilities[:, n, :], ax=axes,
                                xticklabels=states_labels, yticklabels=states_labels, annot=annot, cbar=False,
                                cmap=color_map, fmt='s', square=True, annot_kws={'size': size_values})
                axes.set_yticklabels(states_labels, va='center')
                if not usetex:
                    axes.set_title(
                        r'$\phi_{' + events_labels[n] + '}$', fontsize=size_labels)
                else:
                    axes.set_title(
                        r'$\bm{\phi}_{' + events_labels[n] + '}$', fontsize=size_labels)
    if bottom != None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left,
                            right=right, wspace=wspace, hspace=hspace)
    f.text(0.5, 0.02, 'Next state', ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Previous state', va='center',
           rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array


def discrete_distribution(probabilities, path='', fig_name='distribution_events_states.pdf', v_labels=None,
                          h_labels=None, title=None, color_map=None, figsize=(12, 6), size_labels=16, size_values=14,
                          bottom=None, top=None, left=None, right=None, savefig=False, usetex=False):
    """
    Annotated heatmap of a given discrete distribution with 2 dimensions.

    :type probabilities: 2D array
    :param probabilities: the 2D discrete distribution.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type v_labels: list of strings
    :param v_labels: labels for the first dimension (vertical).
    :type h_labels: list of strings
    :param h_labels: labels for the second dimension (horizontal).
    :type title: string
    :param title: suptitle.
    :param color_map: color map for the heatmap, see seaborn documentation.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of the annotations on top of the heatmap.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type usetex: boolean
    :param usetex: set to True if matplolib figure is rendered with TeX.
    :rtype: Figure
    :return: the figure (see matplotlib).
    """
    if color_map is None:
        color_map = seaborn.cubehelix_palette(
            as_cmap=True, reverse=False, start=0.5, rot=-.75)
    v_size = np.shape(probabilities)[0]
    h_size = np.shape(probabilities)[1]
    # Create annotation matrix
    annot = np.ndarray((v_size, h_size), dtype=object)
    for x1 in range(v_size):
        for x2 in range(h_size):
            p = probabilities[x1, x2]
            if p == 0:
                if usetex:
                    annot[x1, x2] = r'$0$\%'
                else:
                    annot[x1, x2] = r'0%'
            elif p < 0.01:
                if usetex:
                    annot[x1, x2] = r'$<1$\%'
                else:
                    annot[x1, x2] = r'<1%'
            else:
                a = str(int(np.floor(100 * p)))
                if usetex:
                    annot[x1, x2] = r'$' + a + r'$\%'
                else:
                    annot[x1, x2] = a + r'%'
    f = plt.figure(figsize=figsize)
    ax = seaborn.heatmap(probabilities, xticklabels=h_labels, yticklabels=v_labels, annot=annot, cbar=False,
                         cmap=color_map, fmt='s', square=True, annot_kws={'size': size_values})
    # font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=size_labels)
    ax.set_yticklabels(v_labels, va='center')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if bottom is not None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f


def kernels_exp(impact_coefficients, decay_coefficients, events_labels=None, states_labels=None, path='',
                fig_name='kernels.pdf', title=None, palette=None, figsize=(9, 7), size_labels=16,
                size_values=14, size_legend=16, bottom=None, top=None, left=None, right=None, savefig=False,
                fig_array=None, fig=None,
                tmin=None, tmax=None, npoints=500, ymax=None, alpha=1, legend_pos=0, log_timescale=True,
                ls='-'):
    r"""
    Plots the kernels of a state-dependent Hawkes process.
    Here the kernels are assumed to be exponential, that is, :math:`k_{e'e}(t,x)=\alpha_{e'xe}\exp(-\beta_{e'xe}t)`.
    We plot the functions

    .. math::
        t\mapsto ||k_{e'e}(\cdot,x)||_{1,t} := \int _{0}^{t} k_{e'e}(s,x)ds.

    The quantity :math:`||k_{e'e}(\cdot,x)||_{1,t}` can be interpreted as the average number of events of type :math:`e`
    that are directly precipitated by an event of type :math:`e'` within :math:`t` units of time, under state :math:`x`.
    There is a subplot for each couple of event types :math:`(e',e)`.
    In each subplot, there is a curve for each possible state :math:`x`.

    :type impact_coefficients: 3D array
    :param impact_coefficients: the alphas :math:`\alpha_{e'xe}`.
    :type decay_coefficients: 3D array
    :param decay_coefficients: the betas :math:`\beta_{e'xe}`.
    :type events_labels: list of strings
    :param events_labels: labels of the event types.
    :type states_labels: list of strings
    :param states_labels: labels of the states.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type events_labels: list of strings
    :type title: string
    :param title: suptitle.
    :type palette: list of colours
    :param palette: color palette, one color per state :math:`x`.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of tick labels.
    :type size_legend: int
    :param size_legend: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type fig_array: array of Axes
    :param fig_array: fig_array, where to plot the kernels (see matplotlib).
    :type fig: Figure
    :param fig: figure, where to plot the figure (see matplotlib).
    :type tmin: float
    :param tmin: we plot over the time interval [`tmin`, `tmax`].
    :type tmax: float
    :param tmax: we plot over the time interval [`tmin`, `tmax`].
    :type npoints: int
    :param npoints: number of points used to plot.
    :type ymax: float
    :param ymax: upper limit of the y axis.
    :type alpha: float
    :param alpha: between 0 and 1, transparency of the curves.
    :type legend_pos: int
    :param legend_pos: position of the legend in the array of figures.
    :type log_timescale: boolean
    :param log_timescale: set to False to plot with a linear timescale.g
    :type ls: string
    :param ls: the linestyle (see matplotlib).
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    s = np.shape(impact_coefficients)
    number_of_event_types = s[0]
    number_of_states = s[1]
    beta_min = np.min(decay_coefficients)
    beta_max = np.max(decay_coefficients)
    t_max = tmax
    if tmax is None:
        t_max = -np.log(0.1) / beta_min
    t_min = tmin
    if tmin is None:
        t_min = -np.log(0.9) / beta_max
    tt = np.zeros(1)
    if log_timescale:
        order_min = np.floor(np.log10(t_min))
        order_max = np.ceil(np.log10(t_max))
        tt = np.logspace(order_min, order_max, num=npoints)
    else:
        tt = np.linspace(t_min, t_max, num=npoints)
    norm_max = ymax
    if ymax is None:
        norm_max = np.max(np.divide(impact_coefficients,
                          decay_coefficients)) * 1.05
    if palette is None:
        palette = seaborn.color_palette('husl', n_colors=number_of_states)
    if fig_array is None:
        fig, fig_array = plt.subplots(number_of_event_types, number_of_event_types, sharex='col', sharey='row',
                                      figsize=figsize)
    for e1 in range(number_of_event_types):
        for e2 in range(number_of_event_types):
            axes = None
            if number_of_event_types == 1:
                axes = fig_array
            else:
                axes = fig_array[e1, e2]
            for x in range(number_of_states):  # mean
                a = impact_coefficients[e1, x, e2]
                b = decay_coefficients[e1, x, e2]
                yy = a / b * (1 - np.exp(-b * tt))
                l = None
                if np.shape(states_labels) != ():
                    l = states_labels[x]
                axes.plot(tt, yy, color=palette[x],
                          label=l, alpha=alpha, ls=ls)
            # font size for tick labels
            axes.tick_params(axis='both', which='major', labelsize=size_values)
            if log_timescale:
                axes.set_xscale('log')
            axes.set_ylim(ymin=0, ymax=norm_max)
            axes.set_xlim(xmin=t_min, xmax=t_max)
            if np.shape(events_labels) != ():
                axes.set_title(
                    events_labels[e1] + r' $\rightarrow$ ' + events_labels[e2], fontsize=size_labels)
            pos = e2 + number_of_event_types*e1
            if pos == legend_pos and np.shape(states_labels) != ():
                legend = axes.legend(frameon=1, fontsize=size_legend)
                legend.get_frame().set_facecolor('white')
    if title is not None:
        fig.suptitle(title, fontsize=size_labels)
    plt.tight_layout()
    if bottom is not None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return fig, fig_array


def sample_path(times, events, states, model, time_start, time_end, color_palette=None, labelsize=16, ticksize=14,
                legendsize=16, num=1000, s=12, savefig=False, path='', fig_name='sample_path.pdf'):
    r"""
    Plots a sample path along with the intensities.

    :type times: array of floats
    :param times: times when the events occur.
    :type events: array of int
    :param events: type of the event at each event time.
    :type states: array of int
    :param states: state process after each event time.
    :type model: :py:class:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp`
    :param model: the model that is used to compute the intensities.
    :type time_start: float
    :param time_start: time at which the plot starts.
    :type time_end: float
    :param time_end: time at which the plot ends.
    :type color_palette: list of colours
    :param color_palette: one colour per event type.
    :type labelsize: int
    :param labelsize: fontsize of labels.
    :type ticksize: int
    :param ticksize: fontsize of tick labels.
    :type legendsize: int
    :param legendsize: fontsize of the legend.
    :type num: int
    :param num: number of points used to plot.
    :type s: int
    :param s: size of the dots in the scatter plot of the events.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type path: string
    :param path:  where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    if color_palette is None:
        color_palette = seaborn.color_palette(
            'husl', n_colors=model.number_of_event_types)
    'Compute the intensities - this may require all the event times prior to start_time'
    compute_times = np.linspace(time_start, time_end, num=num)
    aggregated_times, intensities = model.intensities_of_events_at_times(
        compute_times, times, events, states)
    'We can now discard the times outside the desired time period'
    index_start = bisect.bisect_left(times, time_start)
    index_end = bisect.bisect_right(times, time_end)
    initial_state = 0
    if index_start > 0:
        initial_state = states[index_start-1]
    times = list(copy.copy(times[index_start:index_end]))
    events = list(copy.copy(events[index_start:index_end]))
    states = list(copy.copy(states[index_start:index_end]))
    f, fig_array = plt.subplots(2, 1, sharex='col')
    'Plot the intensities'
    ax = fig_array[1]
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # intensity_max = intensities.max() * 1.01
    for n in range(model.number_of_event_types):
        ax.plot(aggregated_times, intensities[n], linewidth=1,
                color=color_palette[n], label=model.events_labels[n])
    ax.set_ylim(ymin=0)
    ax.set_ylabel('Intensity', fontsize=labelsize)
    ax.set_xlabel('Time', fontsize=labelsize)
    legend = ax.legend(frameon=1, fontsize=legendsize)
    legend.get_frame().set_facecolor('white')
    'Plot the state process and the events'
    ax = fig_array[0]
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # Plot the event times and types, one color per event type, y-coordinate corresponds to new state of the system
    color_map = ListedColormap(color_palette)
    ax.scatter(times, states, c=events, cmap=color_map, s=s, alpha=1, edgecolors='face',
               zorder=10)
    ax.set_xlim(xmin=time_start, xmax=time_end)
    ax.set_ylim(ymin=-0.1, ymax=model.number_of_states - 0.9)
    ax.set_yticks(range(model.number_of_states))
    ax.set_yticklabels(model.states_labels, fontsize=ticksize)
    ax.set_ylabel('State', fontsize=labelsize)
    # Plot the state process
    times.insert(0, time_start)
    states.insert(0, initial_state)
    times.append(time_end)
    # these two appends are required to plot until `time_end'
    states.append(states[-1])
    ax.step(times, states, where='post', linewidth=1, color='grey', zorder=1)
    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array
