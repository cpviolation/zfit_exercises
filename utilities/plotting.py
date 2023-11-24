import mplhep
mplhep.style.use("LHCb2")
import matplotlib.pyplot as plt
import numpy as np

class Plotter:

    def __init__(self, data, model, bins=100, range=None, 
                xlabel=None, xunit=None,
                model_label='Model', data_label='Data'):
        """A class to make fancy 1D plots of fits

        Args:
            data (array): an array of data
            model (zfit.pdf): the pdf of the model (zfit object)
            bins (int): the number of bins
            range (tuple): the range of the x axis
        """    
        self.data = data
        self.model = model
        self.bins = bins
        self.range = range
        if self.range is None:
            self.range = (np.min(self.data), np.max(self.data))
        self.scale_factor_value = None
        self.xlabel = xlabel
        self.xunit = xunit
        self.bin_width = (self.range[1]-self.range[0])/self.bins
        self.components = []
        self.model_label = model_label
        self.data_label = data_label
        return
    
    def add_component(self, pdf, fraction=0.5, color=None, label=None, linestyle=None):
        """Add a component to the model

        Args:
            pdf (zfit.pdf): the pdf of the model component (zfit object)
            fraction (float, optional): the fraction of the model taken by the component (should be in (0,1) and the sum of all components be <=1). Defaults to 'r'.
            color (str, optional): the color of the line showing the component. Defaults to None.
            label (str, optional): the label of the component. Defaults to None.
        """
        old_fraction = 0
        for component in self.components:
            old_fraction += component[1]
        if old_fraction+fraction > 1:
            raise ValueError('The sum of the fractions of the components should be <=1')
        self.components.append((pdf, fraction, color, label, linestyle))
        return
    
    def scale_factor(self, force=False):
        """Calculate the scale factor for the model

        Returns:
            float: the scale factor
        """
        if self.scale_factor_value is not None and not force:
            return self.scale_factor_value
        self.scale_factor_value = self.data.size / self.bins * (self.range[1]-self.range[0])
        return self.scale_factor_value
    
    def x_label(self):
        """Get the x label

        Returns:
            str: the x label
        """
        if self.xlabel is None:
            return 'x'
        if self.xunit is None:
            return self.xlabel
        return f'{self.xlabel} [{self.xunit}]'
    
    def y_label(self):
        """Get the y label

        Returns:
            str: the y label
        """
        if self.xunit is None:
            return f'Counts  / {self.bin_width}'
        return f'Counts / ({self.bin_width} {self.xunit})'
    
    def pull_array(self, poisson=True):
        """Calculate pulls of data and model

        Args:
            poisson (bool, optional): Use Poisson error for the counts. Defaults to True.

        Returns:
            tuple: the normalised pulls and the bin centers
        """    
        # get histogram from data
        data_counts, data_bins = np.histogram(self.data, bins=self.bins, range=self.range)
        # get values of model in bin centers
        bins_centers = np.array([0.5*(data_bins[i]+data_bins[i+1]) for i in range(self.bins)])
        model_exp = self.model.pdf(bins_centers).numpy()
        # integral of model is 1, need to scale to data
        model_exp_counts = model_exp * self.scale_factor()
        # calculate pulls
        pull = data_counts - model_exp_counts 
        data_err = np.sqrt(data_counts) if poisson else np.sqrt(data_counts) # to be fixed
        for i in range(len(data_err)):
            if data_err[i] == 0: data_err[i] = 1.
        norm_pull = pull / data_err
        return norm_pull, bins_centers

    def plot_fit_and_residuals(self):
        """Plot the fit and residuals

        Returns:
            tuple: Returns the matplotlib figure and its axes
        """
        # Create the figure and split in two with gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1,nrows=2,height_ratios=[1, 3],hspace=None)
        # Create the axes and impose the sharing of the x axis
        (ax1, ax2) = gs.subplots(sharex='col')
        # plot the histogram of the data
        mplhep.histplot(
            np.histogram(self.data, bins=self.bins,range=self.range),
            yerr=True,
            color="black",
            histtype="errorbar",
            label=self.data_label
        )
        # plot the model
        x = np.linspace(self.range[0], self.range[1], 1000)
        y = self.model.pdf(x).numpy()
        ax2.plot(x, y * self.scale_factor(), label=self.model_label)
        # if components are specified, plot them
        for (pdf, fraction, color, label, linestyle) in self.components:
            y = pdf.pdf(x).numpy()
            ax2.plot(x, y * fraction * self.scale_factor(), label=label, color=color, linestyle=linestyle)
        # set labels
        ax2.set_xlabel(self.x_label())
        ax2.set_ylabel(self.y_label())
        # create the array of the pulls
        pullA, bin_centers = self.pull_array()
        # set limits for the pull plot and label
        ax1.set_xlim(self.range[0],self.range[1])
        ax1.set_ylim(-5,5)
        ax1.set_ylabel('Pull')
        # plot the pulls
        ax1.bar(bin_centers, pullA, width=self.bin_width, edgecolor="white", linewidth=1)
        # plot the lines at 0, +/- 3
        ax1.plot(ax1.get_xlim(),[0,0],'k', lw=1)
        ax1.plot(ax1.get_xlim(),[3,3]  ,'r', lw=1)
        ax1.plot(ax1.get_xlim(),[-3,-3],'r', lw=1)
        # add legend
        ax2.legend()
        return fig, (ax1, ax2)