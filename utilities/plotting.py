import mplhep
mplhep.style.use("LHCb2")
import matplotlib.pyplot as plt
import numpy as np

def pull_array(data, model, bins, ax_range, poisson=True):
    """Calculate pulls of data and model

    Args:
        data (array): an array of data
        model (zfit.pdf): the pdf of the model (zfit object)
        bins (int): the number of bins
        ax_range (tuple): the range of the x axis
        poisson (bool, optional): Use Poisson error for the counts. Defaults to True.

    Returns:
        tuple: the normalised pulls and the bin centers
    """    
    # get histogram from data
    data_counts, data_bins = np.histogram(data, bins=bins, range=ax_range)
    # get values of model in bin centers
    bins_centers = np.array([0.5*(data_bins[i]+data_bins[i+1]) for i in range(bins)])
    model_exp = model.pdf(bins_centers).numpy()
    # integral of model is 1, need to scale
    n_sample = data.size
    plot_scaling = n_sample / bins * (ax_range[1]-ax_range[0])
    model_exp_counts = model_exp * plot_scaling
    # calculate pulls
    pull = data_counts - model_exp_counts 
    data_err = np.sqrt(data_counts) if poisson else np.sqrt(data_counts) # to be fixed
    for i in range(len(data_err)):
        if data_err[i] == 0: data_err[i] = 1.
    norm_pull = pull / data_err
    return norm_pull, bins_centers
    

def plot_fit_results(data, model, xmin=None, xmax=None, nbins=100 ):
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)
    axis_range = (xmin,xmax)
    fig = plt.figure()
    gs = fig.add_gridspec(ncols=1,nrows=2,height_ratios=[1, 3],hspace=None)
    (ax1, ax2) = gs.subplots(sharex='col')
    mplhep.histplot(
        np.histogram(data_np, bins=nbins,range=axis_range),
        yerr=True,
        color="black",
        histtype="errorbar",
    )
    ax2.plot(x, y * plot_scaling, label="Sum - Model")
    ax2.set_ylabel('Counts')
    pullA, bin_centers = pull_array(data, model, nbins, axis_range)
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(-5,5)
    ax1.bar(bin_centers, pullA)#, 'k')
    ax1.plot(ax1.get_xlim(),[0,0],'k', lw=1)
    ax1.set_ylabel('Pull')
    return fig, (ax1, ax2)