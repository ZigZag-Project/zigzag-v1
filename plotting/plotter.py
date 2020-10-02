from reader import Reader
from matplotlib import pyplot as plt


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
                 For now, please keep path of form '/best_su_best_tm/'.
        """

        self.reader = Reader(path)

        self.layer_numbers = self.reader.layer_numbers

        self.total_energy, self.coarse_energy = self.reader.coarse_energy()

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
        """

        # Check if colors where provided, otherwhise use the default color cycle
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Number of bars per group
        n_bars = len(data)

        # The width of a single bar
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for i, (name, values) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(
                    x + x_offset,
                    y,
                    width=bar_width * single_width,
                    color=colors[i % len(colors)],
                )

            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])

        # Draw legend if we need
        if legend:
            ax.legend(bars, data.keys())

        # Change xticks so there is one for every layer
        x = range(len(values))
        x_labels = [("Layer %d" % nb) for nb in layer_numbers]
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
        """
        # Check if colors where provided, otherwhise use the default color cycle
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Number of bars per group
        n_bars = len(data)
        n_layers = len(layer_numbers)

        # The width of a single bar
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for i, (name, nested_list) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

            # Initialize the stack_below lists for correct stacking
            stack_below = [0 for a in range(n_layers)]

            # Iterate over the lists in nested_list
            for j, values in enumerate(nested_list):
                # Draw a bar for every value of that type
                # for x, y in enumerate(values):
                x = [i + x_offset for i in range(n_layers)]
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
        mem_hierarchy_labels = self.reader.memory_labels
        for v in mem_hierarchy_labels["I"]:
            fine_stack_labels.append("I_" + v)
        for v in mem_hierarchy_labels["W"]:
            fine_stack_labels.append("W_" + v)
        for v in mem_hierarchy_labels["O"]:
            fine_stack_labels.append("O_" + v)
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
