#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Fri Oct  2 17:31:34 2020
# Language : Python3

################################### IMPORTS ####################################

# External imports
from matplotlib import pyplot as plt

# Internal imports
from reader import Reader

##################################### CODE #####################################


# TODO
# Plotter is not really a class and more of a struct, in the sense that the data
# inside the class is only read and never written to. This makes a lot of sense
# because it is only used for plottings, but it means that the structure should
# probably be changed to reflect that. For instance, the Plotter.reader object
# should probably go away and only be used in the constructor.
#
# I am not completely decided yet.

# NOTE from Arne
# I don't really care if it's a class or struct, whatever you think is best.


class Plotter:
    """
    Class dedicated to plotting the output of ZigZag.
    """

    def __init__(self, path: str):
        """
        Constructor of the Plotter class.

        Arguments
        =========
         - path: The path of the directory where the output of ZigZag was built.
                 For now, please keep path of form '.../best_su_best_tm/'.
        """
        self.reader = Reader(path)

        # Shortcut to the Reader.layer_numbers value.
        self.layer_numbers = self.reader.layer_numbers

        # We get coarse energy readings from our Reader.
        self.total_energy, self.coarse_energy = self.reader.coarse_energy()

        # Same for fine_energy.
        self.fine_energy = self.reader.fine_energy()

    def bar_plot(
        self,
        ax,
        data,
        layer_numbers,
        colors=None,
        total_width=0.8,
        single_width=1,
        legend=True,
        ylabel="Energy",
        title="",
    ):
        """
        Draws a bar plot with multiple bars per data point.

        Arguments
        =========
         - ax : matplotlib.pyplot.axis
            The axis we want to draw our plot on.

         - data: dictionary
            A dictionary containing the data we want to plot. Keys are the names
            of the data, the items is a list of the values.

            Example:
            data = {
                "x":[1,2,3],
                "y":[1,2,3],
                "z":[1,2,3],
            }

         - colors : array-like, optional
            A list of colors which are used for the bars. If None, the colors
            will be the standard matplotlib color cyle. (default: None)

         - total_width : float, optional, default: 0.8
            The width of a bar group. 0.8 means that 80% of the x-axis is covered
            by bars and 20% will be spaces between the bars.

         - single_width: float, optional, default: 1
            The relative width of a single bar within a group. 1 means the bars
            will touch eachother within a group, values less than 1 will make
            these bars thinner.

         - legend: bool, optional, default: True
            If this is set to true, a legend will be added to the axis.

        Side-effects
        ============
        This call will prepare a plot for the given axis.
        """

        # Check if colors where provided, otherwhise use the default color cycle
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Number of bars per group
        number_bars = len(data)

        # The width of a single bar
        bar_width = total_width / number_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for index_bar, (name, values) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (index_bar - number_bars / 2) * bar_width + bar_width / 2

            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(
                    x + x_offset,
                    y,
                    width=bar_width * single_width,
                    color=colors[index_bar % len(colors)],
                )

            # Add a handle to the last drawn bar, which we will need for the
            # legend
            bars.append(bar[0])

        # Draw legend if we need
        if legend:
            ax.legend(bars, data.keys())

        # Change xticks so there is one for every layer
        x = range(len(values))
        x_labels = ["Layer {}".format(nb) for nb in layer_numbers]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Set plot y axis labels
        ax.set_ylabel(ylabel)

        # Set plot title
        ax.set_title(title)

    def stacked_bar_plot(
        self,
        ax,
        data,
        layer_numbers,
        stack_labels,
        colors=None,
        total_width=0.8,
        single_width=1,
        legend=True,
        ylabel="Energy",
        title="",
    ):
        """
        Draws a stacked bar plot with multiple bars per data point.

        Arguments
        =========
         - ax : matplotlib.pyplot.axis
            The axis we want to draw our plot on.

         - data: dictionary
            A dictionary containing the data we want to plot. Keys are the names
            of the data, the items is a nested list of the values. One list for
            each stack

            Example:
            data = {
                "x":[[1,2,3],[1,2,3]]
                "y":[[1,2,3],[1,2,3]]
                "z":[[1,2,3],[1,2,3]]
            }

         - colors : array-like, optional
            A list of colors which are used for the bars. If None, the colors
            will be the standard matplotlib color cyle. (default: None)

         - total_width : float, optional, default: 0.8
            The width of a bar group. 0.8 means that 80% of the x-axis is covered
            by bars and 20% will be spaces between the bars.

         - single_width: float, optional, default: 1
            The relative width of a single bar within a group. 1 means the bars
            will touch eachother within a group, values less than 1 will make
            these bars thinner.

         - legend: bool, optional, default: True
            If this is set to true, a legend will be added to the axis.

        Side-effects
        ============
        This call will prepare a plot for the given axis.
        """
        # Check if colors where provided, otherwhise use the default color cycle
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Number of bars per group
        number_bars = len(data)
        number_layers = len(layer_numbers)

        # The width of a single bar
        bar_width = total_width / number_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for index_bar, (name, nested_list) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (index_bar - number_bars / 2) * bar_width + bar_width / 2

            # Initialize the stack_below lists for correct stacking
            stack_below = [0 for _ in range(number_layers)]

            # Iterate over the lists in nested_list
            for j, values in enumerate(nested_list):
                # Draw a bar for every value of that type
                # for x, y in enumerate(values):
                x = [i + x_offset for i in range(number_layers)]
                bar = ax.bar(
                    x,
                    values,
                    width=bar_width * single_width,
                    bottom=stack_below,
                    color=colors[j % len(colors)],
                )

                # Add 'min_en' or 'max_ut' to bottom of correct bar
                if j == 0:
                    for x_pos in x:
                        ax.text(
                            x_pos,
                            150,
                            name.replace("_", " "),
                            ha="center",
                            fontsize=10,
                            color="k",
                        )

                # Add a handle to the last drawn bar, which we'll need for the
                # legend
                bars.append(bar[0])

                # Add the processed stack to the stack_below list
                stack_below = [sum(x) for x in zip(stack_below, values)]

        # Draw legend if we need
        if legend:
            ax.legend(bars, stack_labels)

        # Change xticks so there is one for every layer
        x = range(len(values))
        x_labels = ["Layer {}".format(nb) for nb in layer_numbers]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Set plot y axis labels
        ax.set_ylabel(ylabel)

        # Set plot title
        ax.set_title(title)

    def plot_energy(self, total_width=0.6, single_width=0.9, colors=None):
        """
        Draws and displays the plots for the energy with the data from the
        provided ZigZag output.

        Arguments
        =========
         - total_width : float, optional, default: 0.6
            The width of a bar group. 0.8 means that 80% of the x-axis is covered
            by bars and 20% will be spaces between the bars.

        - single_width: float, optional, default: 0.9
            The relative width of a single bar within a group. 1 means the bars
            will touch eachother within a group, values less than 1 will make
            these bars thinner.

         - colors: The color scheme to use for the plots.

        Side-effects
        ============
        This call will display some plots.
        """

        fig1, ax1 = plt.subplots()
        self.bar_plot(
            ax1,
            self.total_energy,
            self.layer_numbers,
            total_width=total_width,
            single_width=single_width,
            colors=colors,
            title="Total energy",
        )

        coarse_stack_labels = ["Memory access energy", "MAC energy"]
        fig2, ax2 = plt.subplots()
        self.stacked_bar_plot(
            ax2,
            self.coarse_energy,
            self.layer_numbers,
            coarse_stack_labels,
            total_width=total_width,
            single_width=single_width,
            colors=colors,
            title="Coarse grained energy",
        )

        # Construct the legend labels for the stacks in the bar plot
        fine_stack_labels = []
        memory_hierarchy_labels = self.reader.memory_labels

        for data_type, memory_elements in memory_hierarchy_labels.items():
            for memory_name in memory_elements:
                fine_stack_labels.append("{}_{}".format(data_type, memory_name))
        # One last label for the energy spent by the mac arrays.
        fine_stack_labels.append("mac_array")

        fig3, ax3 = plt.subplots()
        self.stacked_bar_plot(
            ax3,
            self.fine_energy,
            self.layer_numbers,
            fine_stack_labels,
            total_width=total_width,
            single_width=single_width,
            colors=colors,
            title="Fine grained energy",
        )

        plt.show()

    def plot_design_space(self, paths, legend=True, legend_labels=None):
        """
        NOTE from Arne
        Ideally, we would have another class or struct for this,
        as we're now using multiple paths for one plot.
        Or change this one to handle both single and multiple paths in init.
        For now, I'll just leave it here.

        NOTE from Arne
        In future: check that every path concerns the same layers

        Plots two design points for each path in paths.
        Design space is the latency, energy space.
        TODO Point size represents the area.

        Arguments
        =========
         - paths: A list of paths to ZigZag output folders.

        Side-effects
        ============
        This call will display some plots.

        """

        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Get all data for plots by iterating through paths
        for i, path in enumerate(paths):
            reader = Reader(path)

            # Shortcut to the Reader.layer_numbers value.
            layer_numbers = reader.layer_numbers

            # We get total energy readings from our Reader.
            # total_energy is a list of floats for every layer.
            total_energy, _ = reader.coarse_energy()
            total_energy_sum_min_en = sum(total_energy['min_en'])
            total_energy_sum_max_ut = sum(total_energy['max_ut'])

            # We get total latency readings from our Reader.
            # total_latency is a list of floats for every layer.
            total_latency = reader.total_latency()
            total_latency_sum_min_en = sum(total_latency['min_en'])
            total_latency_sum_max_ut = sum(total_latency['max_ut'])

            ax.scatter(total_energy_sum_min_en, total_latency_sum_min_en,
                marker='o', color=colors[i % len(colors)])

            ax.scatter(total_energy_sum_max_ut, total_latency_sum_max_ut,
                marker='^', color=colors[i % len(colors)])

        if legend is True:
            if legend_labels is None:
                # If no labels provided, use paths
                legend_labels = [opt_type + ' ' + path 
                    for path in paths for opt_type in ['min_en','max_ut']]

            ax.legend(legend_labels)

        ax.margins(y=0.2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('energy')
        ax.set_ylabel('latency')


        plt.show()

##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
