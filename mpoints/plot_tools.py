import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn
import statsmodels.tsa.stattools as stattools

def qq_plot(residuals, shape, path=os.getcwd(), fig_name='qq_plot.pdf', log=False, q_min=0.01, q_max=0.99,
            number_of_quantiles=50, title=None, labels=None, model_labels=None, palette=None, figsize=(12, 6),
            size_labels=16, size_ticks=14, bottom=0.12, top=0.93, left=0.08, right=0.92, savefig=False):
    """

    :param residuals:
    :param path:
    :param log:
    :param q_min:
    :param q_max:
    :param number_of_quantiles:
    :return:
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
    if type(residuals[0][0]) in [list, np.ndarray]:  # case when there is more than one model
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels==None:
        model_labels = [None]*n_models
    v_size = shape[0]
    h_size = shape[1]
    seaborn.set()
    if palette==None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title != None:
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
                axes.tick_params(axis='both', which='major', labelsize=size_ticks)  # font size for tick labels
                if n_models == 1:
                    quantiles_empirical = np.zeros(number_of_quantiles)
                    for k in range(number_of_quantiles):
                        q = quantile_levels[k]
                        x = np.percentile(residuals[n], q * 100)
                        quantiles_empirical[k] = x
                    axes.plot(quantiles_theoretical, quantiles_empirical, color=palette[0])
                    axes.plot(quantiles_theoretical, quantiles_theoretical, color='k', linewidth=0.8, ls='--')
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
                    if i == 0 and j == h_size-1 :  # only add legend in the top-left subplot
                        legend = axes.legend(frameon=1, fontsize=11)
                        legend.get_frame().set_facecolor('white')
                if log:
                    axes.set_xscale('log')
                    axes.set_yscale('log')
                if labels != None:
                    axes.set_title( labels[n], fontsize=size_labels)
    plt.tight_layout()
    if bottom!=None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    f.text(0.5, 0.02, 'Quantile (standard exponential distribution)', ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Quantile (empirical)', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

def correlogram(residuals, path=os.getcwd(), fig_name='correlogram.pdf', title=None, labels=None, model_labels=None,
                palette=None, n_lags=50, figsize=(8, 6), size_labels=16, size_ticks=14, bottom=None, top=None,
                left=None, right=None,savefig=True):
    """

    :param residuals:
    :param path:
    :param fig_name:
    :param title:
    :param labels:
    :param model_labels:
    :param palette:
    :return:
    """
    # find number of models given and number of event types (dim)
    n_models = 1
    dim = len(residuals)
    if type(residuals[0][0]) in [list, np.ndarray]:  # case when there is more than one model
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels == None:
        model_labels = [None] * n_models
    v_size = dim
    h_size = dim
    seaborn.set()
    # seaborn.set_style('dark')  # no grid for good-looking small subplots
    if palette == None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title != None:
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
            axes.tick_params(axis='both', which='major', labelsize=size_ticks)  # font size for tick labels
            if n_models == 1:
                max_length = min(len(residuals[i]), len(residuals[j]))
                ccf = stattools.ccf(np.array(residuals[i][0:max_length]),
                                    np.array(residuals[j][0:max_length]),
                                    unbiased=True)
                axes.plot(ccf[0:n_lags+1], color=palette[0])
                axes.set_xlim(xmin=0, xmax=n_lags)
            else:
                for m in range(n_models):
                    max_length = min(len(residuals[m][i]), len(residuals[m][j]))
                    ccf = stattools.ccf(np.array(residuals[m][i][0:max_length]),
                                        np.array(residuals[m][j][0:max_length]),
                                        unbiased=True)
                    axes.plot(ccf[0:n_lags + 1], color=palette[m], label=model_labels[m])
                    axes.set_xlim(xmin=0, xmax=n_lags)
                if i+j==0:  # only add legend in the first subplot
                    legend = axes.legend(frameon=1, fontsize=size_labels)
                    legend.get_frame().set_facecolor('white')
            if labels != None:
                axes.set_title(labels[i] + r'$\rightarrow$' + labels[j], fontsize=size_labels)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if bottom!=None:
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    f.text(0.5, 0.025, 'Lag', ha='center', fontsize=size_labels)
    f.text(0.015, 0.5, 'Correlation', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

def plot_histogram_of_3d_matrix(path, fig_name, data, labels=None, title=None, superposed=False,
                    n_bins=10, kernel_smoothing=False, display_means=False, true_means=None, fig_size=(8, 12)):
    """

    :param path:
    :param fig_name:
    :param data:
    :param labels:
    :param title:
    :param superposed:
    :param n_bins:
    :param kernel_smoothing:
    :param display_means:
    :param true_means:
    :return:
    """
    data_shape = data.shape
    seaborn.set()
    palette = seaborn.color_palette('husl', data_shape[1])

    'Plot - no superposition case'
    if not superposed:
        h_size = data_shape[2]
        v_size = data_shape[0] * data_shape[1]
        f, fig_array = plt.subplots(v_size, h_size, figsize=fig_size)
        if title!=None:
            f.suptitle(title)
        for x in range(data_shape[1]):
            for e1 in range(data_shape[0]):
                for e2 in range(data_shape[2]):
                    if kernel_smoothing:
                        seaborn.distplot(data[e1, x, e2, :], hist=False, color=palette[x],
                                         kde_kws={"shade": True}, ax=fig_array[e1 + data_shape[0]*x, e2])
                    else:
                        fig_array[e1 + data_shape[0]*x, e2].hist(data[e1, x, e2, :], bins=n_bins, color=palette[x])
                    if display_means:
                        m = np.mean(data[e1, x, e2, :])
                        fig_array[e1 + data_shape[0] * x, e2].axvline(x=m, color='k', linewidth=1)
                        if np.shape(true_means)!=():
                            m = true_means[e1, x, e2]
                            fig_array[e1 + data_shape[0] * x, e2].axvline(x=m, color='k', ls='--', linewidth=1)
                    if labels!=None:
                        fig_array[e1 + data_shape[0]*x, e2].set_title(labels[e1][x][e2])
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, top=0.9)
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

    'Plot - superposition case'
    if superposed:
        h_size = data_shape[2]
        v_size = data_shape[0]
        f, fig_array = plt.subplots(v_size, h_size, figsize=fig_size)
        if title!=None:
            f.suptitle(title)
        for e1 in range(data_shape[0]):
            for e2 in range(data_shape[2]):
                ax = fig_array[e1, e2]
                ax.set_autoscaley_on(True)
                for x in range(data_shape[1]):
                    label = None
                    if labels != None:
                        label = labels[e1][x][e2]
                    if kernel_smoothing:
                        ax = seaborn.distplot(data[e1, x, e2, :], hist=False, color=palette[x],
                                         kde_kws={"shade": True}, ax=ax, label=label)
                    else:
                        ax.hist(data[e1, x, e2, :], bins=n_bins, color=palette[x], label=label)
                    if display_means:
                        m = np.mean(data[e1, x, e2, :])
                        ax.axvline(x=m, color=palette[x], linewidth=1)
                        if np.shape(true_means) != ():
                            m = true_means[e1, x, e2]
                            ax.axvline(x=m, color=palette[x], ls='--', linewidth=1)
                    ax.set_autoscaley_on(True)
                if labels != None:
                        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, top=0.9)
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

def plot_histogram_of_vector(path, fig_name, data, shape, labels=None, title=None, n_bins=10, kernel_smoothing=False,
                             display_means=False, true_means=None, fig_size=(8, 6)):
    """

    :param path:
    :param fig_name:
    :param data:
    :param shape:
    :param labels:
    :param title:
    :param n_bins:
    :param kernel_smoothing:
    :param display_means:
    :param true_means:
    :return:
    """
    dim = data.shape[0]
    v_size = shape[0]
    h_size = shape[1]
    seaborn.set()
    palette = seaborn.color_palette('husl', dim)
    f, fig_array = plt.subplots(v_size, h_size, figsize=fig_size)
    if title != None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            n = j + h_size*i
            if n < dim:  # the shape of the subpots might be bigger than dim, i.e. 3 plots on a 2x2 grid.
                axes = None
                if v_size==1 and h_size==1:
                    axes = fig_array
                elif v_size==1:
                    axes = fig_array[j]
                elif h_size==1:
                    axes = fig_array[i]
                else:
                    axes = fig_array[i, j]
                if kernel_smoothing:
                    seaborn.distplot(data[n, :], hist=False, color=palette[n],
                                     kde_kws={"shade": True}, ax=axes)
                else:
                    axes.hist(data[n, :], bins=n_bins, color=palette[n])
                if display_means:
                    m = np.mean(data[n, :])
                    axes.axvline(x=m, color='k', linewidth=1)
                    if np.shape(true_means)!=():
                        m = true_means[n]
                        axes.axvline(x=m, color='k', ls='--', linewidth=1)
                if labels!=None:
                    axes.set_title(labels[n])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.85)
    entire_path = os.path.join(path, fig_name)
    plt.savefig(entire_path)

def distribution_in_time(times, data, alpha_low=5, alpha_up=95, color='b', ax=None):
    means = np.zeros(len(times))
    lower_q = np.zeros(len(times))
    upper_q = np.zeros(len(times))
    for i in range(len(times)):
        means[i] = np.mean(data[i, :])
        lower_q[i] = np.percentile(data[i, :], alpha_low)
        upper_q[i] = np.percentile(data[i, :], alpha_up)
    'Fill area between lower and upper quantiles'
    polygon_x = list(times) + list(reversed(list(times)))
    polygon_y = list(upper_q) + list(reversed(list(lower_q)))
    if ax==None:
        plt.fill(polygon_x, polygon_y, alpha=0.2, color=color)
        plt.plot(times, means, color=color)
    else:
        ax.fill(polygon_x, polygon_y, alpha=0.2, color=color)
        ax.plot(times, means, color=color)

def distribution_in_time_of_3d_matrix(path, fig_name, times, data, labels=None, xlabel=None, title=None,
                                      true_means=None, alpha_low=5, alpha_up=95):
    data_shape = data.shape
    seaborn.set()
    palette = seaborn.color_palette('husl', data_shape[1])

    h_size = data_shape[2]
    v_size = data_shape[0] * data_shape[1]
    f, fig_array = plt.subplots(v_size, h_size, figsize=(8, 12), sharex='col')
    if title != None:
        f.suptitle(title)
    for x in range(data_shape[1]):
        for e1 in range(data_shape[0]):
            for e2 in range(data_shape[2]):
                ax = fig_array[e1 + data_shape[0] * x, e2]
                distribution_in_time(times, data[e1, x, e2, :, :], alpha_low, alpha_up, color=palette[x], ax=ax)
                if np.shape(true_means) != ():
                    m = true_means[e1, x, e2, :]
                    ax.plot(times, m, color='k', linewidth=0.75, ls='--')
                if labels != None:
                    ax.set_title(labels[e1][x][e2])
    if xlabel!=None:
        f.text(0.514, 0.01, xlabel, ha='center')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.92)
    entire_path = os.path.join(path, fig_name)
    plt.savefig(entire_path)

def distribution_in_time_of_vector(path, fig_name, times, data, shape, labels=None, xlabel=None, title=None,
                                   true_means=None, alpha_low=5, alpha_up=95):
    dim = data.shape[0]
    v_size = shape[0]
    h_size = shape[1]
    seaborn.set()
    palette = seaborn.color_palette('husl', dim)
    f, fig_array = plt.subplots(v_size, h_size, sharex='col')
    if title != None:
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
                distribution_in_time(times, data[n, :, :], alpha_low, alpha_up, color=palette[n], ax=axes)
                if np.shape(true_means) != ():
                    m = true_means[n, :]
                    axes.plot(times, m, color='k', linewidth=0.75, ls='--')
                if labels != None:
                    axes.set_title(labels[n])
    if xlabel!=None:
        f.text(0.514, 0.02, xlabel, ha='center')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.85)
    entire_path = os.path.join(path, fig_name)
    plt.savefig(entire_path)

def transition_probabilities(probabilities, shape=None, path=os.getcwd(), fig_name='transition_probabilities.pdf',
                             events_labels=None, states_labels=None, title=None, color_map=None, fig_size=(12, 6),
                             size_labels=16, size_values=14, bottom=0.1, top=0.95, left=0.08, right=0.92,
                             wspace=0.2, hspace=0.2,
                             savefig=False, usetex=False):
    if color_map == None:
        color_map = seaborn.cubehelix_palette(as_cmap=True, reverse=False, start=0.5, rot=-.75)
    number_of_states = np.shape(probabilities)[0]
    number_of_event_types = np.shape(probabilities)[1]
    if shape == None:
        v_size = 1
        h_size = number_of_event_types
    else:
        v_size = shape[0]
        h_size = shape[1]
    f, fig_array = plt.subplots(v_size, h_size, figsize=fig_size)
    if title != None:
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
                axes.tick_params(axis='both', which='major', labelsize=size_labels)  # font size for tick labels
                # Create annotation matrix
                annot = np.ndarray((number_of_states, number_of_states), dtype=object)
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
                    axes.set_title(r'$\phi_{' + events_labels[n] + '}$', fontsize=size_labels)
                else:
                    axes.set_title(r'$\bm{\phi}_{' + events_labels[n] + '}$', fontsize=size_labels)
    if bottom!=None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace)
    f.text(0.5, 0.02, 'Next state', ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Previous state', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

def discrete_distribution(probabilities, path=os.getcwd(), fig_name='distribution_events_states.pdf', v_labels=None,
                          h_labels=None, title=None, color_map=None, figsize=(12, 6), size_labels=16, size_values=14,
                          bottom=None, top=None, left=None, right=None, savefig=False, usetex=False):
    if color_map == None:
        color_map = seaborn.cubehelix_palette(as_cmap=True, reverse=False, start=0.5, rot=-.75)
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
    plt.figure(figsize=figsize)
    ax = seaborn.heatmap(probabilities, xticklabels=h_labels, yticklabels=v_labels, annot=annot, cbar=False,
                    cmap=color_map, fmt='s', square=True, annot_kws={'size': size_values})
    ax.tick_params(axis='both', which='major', labelsize=size_labels)  # font size for tick labels
    ax.set_yticklabels(v_labels, va='center')
    if title != None:
        plt.title(title)
    plt.tight_layout()
    if bottom != None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)

def kernels_exp(impact_coefficients, decay_coefficients, events_labels=None, states_labels=None, path=os.getcwd(),
                fig_name='kernels.pdf', title=None, palette=None, figsize=(9, 7), size_labels=16,
                size_values=14, bottom=None, top=None, left=None, right=None, savefig=False,
                fig_array=None, fig=None,
                tmin=None, tmax=None, npoints=500, ymax=None, alpha=1, legend_pos=0):
    s = np.shape(impact_coefficients)
    number_of_event_types = s[0]
    number_of_states = s[1]
    beta_min = np.min(decay_coefficients)
    beta_max = np.max(decay_coefficients)
    t_max = tmax
    if tmax == None:
        t_max = -np.log(0.1) / beta_min
    t_min = tmin
    if tmin == None:
        t_min = -np.log(0.9) / beta_max
    order_min = np.floor(np.log10(t_min))
    order_max = np.ceil(np.log10(t_max))
    tt = np.logspace(order_min, order_max, num=npoints)
    norm_max = ymax
    if ymax is None:
        norm_max = np.max(np.divide(impact_coefficients, decay_coefficients)) * 1.05
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
                axes.plot(tt, yy, color=palette[x], label=l, alpha=alpha)
            axes.tick_params(axis='both', which='major', labelsize=size_values)  # font size for tick labels
            axes.set_xscale('log')
            axes.set_ylim(ymin=0, ymax=norm_max)
            axes.set_xlim(xmin=t_min, xmax=t_max)
            if np.shape(events_labels) != ():
                axes.set_title(events_labels[e1] + r' $\rightarrow$ ' + events_labels[e2], fontsize=size_labels)
            pos = e2 + number_of_event_types*e1
            if pos == legend_pos and np.shape(states_labels) != () :
                legend = axes.legend(frameon=1, fontsize=size_labels)
                legend.get_frame().set_facecolor('white')
    if title != None:
        fig.suptitle(title, fontsize=size_labels)
    plt.tight_layout()
    if bottom != None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return fig, fig_array