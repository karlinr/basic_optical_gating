import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, Layout
from .logger import Logger, Colour

class iBasicOpticalGatingPlotter():
    def __init__(self, bogs, names = None) -> None:
        """
        Plots instances of the basic optical gating class

        Args:
            bogs (object or list): single or list of BOG objects
            names (string, optional): Names of the BOG objects. Defaults to None.
        """        
        # Initialise the logger
        self.logger = Logger("OGP")
        self.logger.set_normal()

        # Fill in our arrays of basic optical gating instances and names
        if type(bogs) is list:
            self.bogs = bogs
        else:
            self.bogs = [bogs]

        if names == None:
            self.names = [f"Optical gating {i}" for i in range(len(self.bogs))]
        elif type(names) is list:
            self.names = names
        else:
            self.names = [names]

        # Initialise vars
        self.n = len(self.bogs)

        self.settings = {
            "figsize" : (14,8)
        }

    def _begin_plot(self, plot_title):
        # Printo to log
        # Set theme
        #plt.style.use("seaborn-v0_8-poster")
        #plt.style.use("rose-pine-dawn")

        # Setup our figure
        plt.figure(figsize = self.settings["figsize"])
        plt.title(plot_title)


    def plot_sads(self):
        def plot_func(frame_number):
            self._begin_plot(f"SAD plot for frame {frame_number}")

            for i in range(self.n):
                xs = np.arange(self.bogs[i].sads[frame_number].shape[0]) - 2
                xs = xs / (self.bogs[i].reference_period)
                xs = xs * (2 * np.pi)
                plt.scatter(xs, self.bogs[i].sads[frame_number], s = 5)
                plt.ylim(np.min(self.bogs[i].sads), np.max(self.bogs[i].sads))
        interact(plot_func, 
                frame_number = widgets.IntSlider(min=0, max=self.bogs[0].sads.shape[0], value=0, layout=Layout(width='900px'))
        )

    def plot_delta_phases_phases(self):
        def plot_func(frame_number_begin, frame_number_end):
            self._begin_plot("Delta phases plot")
            for i in range(self.n):
                plt.scatter(self.bogs[i].phases[1::], self.bogs[i].delta_phases)
                plt.scatter(self.bogs[i].phases[frame_number_begin + 1:frame_number_end + 1], self.bogs[i].delta_phases[frame_number_begin + 1:frame_number_end + 1], c = np.arange(frame_number_end - frame_number_begin))
        interact(plot_func, 
                frame_number_begin = widgets.IntSlider(min=0, max=self.bogs[0].phases.shape[0], value=0, layout=Layout(width='900px')),
                frame_number_end = widgets.IntSlider(min=0, max=self.bogs[0].phases.shape[0], value=0, layout=Layout(width='900px'))

        )