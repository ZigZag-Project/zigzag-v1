from plotter import Plotter

if __name__ == "__main__":

    # Path where the output files are located.
    # For now, stick to a single memory hierarchy
    # and fixed spatial unrolling results
    path = 'test_outputs/best_su_best_tm/'

    # Instantiate a plotter object
    plotter = Plotter(path)

    # Call the three energy plots of varying detail
    plotter.plot_energy()
