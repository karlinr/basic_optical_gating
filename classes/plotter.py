import numpy as np
import matplotlib.pyplot as plt

from .logger import Logger, Colour

# SECTION: Plotter class
class BasicOpticalGatingPlotter():
    # TODO: Add plot settings
    # Clean up histogram plots - maybe move them into one?

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
        self.figsize = (14,8)
        
    def _check_if_run(self, check_name = None):
        # Checks if the relevant functions have been run within the basic optical gating instances

        not_run_count = 0
        for i in range(self.n):
            if self.bogs[i].has_run[check_name] == False:
                self.logger.print_message("ERROR", f"Run {check_name} in basic optical gating with name \"{self.names[i]}\" before plotting")
                not_run_count += 1
        if not_run_count >= 1:
            raise Exception(f"Not all basic optical gating instances have been run with {check_name}")

    def _begin_plot(self, plot_title):
        # Printo to log
        self.logger.print_message("INFO", f"Plotting {plot_title}")

        # Set theme
        #plt.style.use("seaborn-v0_8-poster")
        #plt.style.use("rose-pine-dawn")

        # Setup our figure
        plt.figure(figsize = self.figsize)
        plt.title(plot_title)

    def set_figsize(self, figsize):
        """
        Set the figure size for the plots

        Args:
            figsize (tuple): tuple of width and height. e.g (14,8)
        """        
        self.figsize = figsize

    def plot_sads(self, frame_number = None):
        """
        Plot the SADs for a given frame

        Args:
            frame_number (int, optional): The frame to plot SADs for. Defaults to None.
        """        
        self._check_if_run("get_sads")
        self._check_if_run("get_phases")

        if frame_number == None:
            frame_number = 0
            self.logger.print_message("WARNING", "No frame number specified, using 0")

        # Plot the SADs for a given frame
        self._begin_plot(f"SADs against frame number for frame: {frame_number}")
        for i in range(self.n):
            xs = np.arange(self.bogs[i].sads[frame_number].shape[0]) - 2
            xs = xs / (self.bogs[i].reference_period)
            xs = xs * (2 * np.pi)
            plt.scatter(xs, self.bogs[i].sads[frame_number], label = self.names[i], c = f"C{i}")
            argmin = np.argmin(self.bogs[i].sads[frame_number][2:-2]) + 2
            #plt.scatter([argmin], [self.bogs[i].sads[frame_number][argmin]], c = f"C{i}", alpha = 0.5, s = 100)
            #plt.axvline(self.bogs[i].phases[frame_number] + 2, c = f"C{i}", ls = "--")
        #plt.axvline(2, c = "black", ls = "--")
        #plt.axvline(self.bogs[0].sads[frame_number].shape[0] - 3, c = "black", ls = "--")
        #plt.xticks(range(self.bogs[i].sads[frame_number].shape[0]))
        plt.xlabel("Frame number")
        plt.ylabel("SAD")
        plt.legend()
        plt.show()

    def plot_phases(self):
        """ Plots the phases against frame number"""        
        self._check_if_run("get_phases")

        # Plot the phases
        self._begin_plot("Phases against frame number")
        for i in range(self.n):
            plt.scatter(self.bogs[i].unwrapped_phases, self.bogs[i].phases, label = self.names[i])
        plt.xlabel("Frame number")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    def plot_delta_phases(self):
        """ Plots the delta phases against frame number"""        
        self._check_if_run("get_delta_phases")

        self._begin_plot("Delta phases against frame number")
        for i in range(self.n):
            plt.scatter(self.bogs[i].unwrapped_phases[0:-1], self.bogs[i].delta_phases, label = self.names[i], s = 2)
        plt.xlabel("Unwrapped phase")
        plt.ylabel("Delta phase")
        plt.legend()
        plt.show()

    def plot_delta_phases_phases(self):
        """ Plots the delta phases against phases"""
        self._check_if_run("get_delta_phases")

        self._begin_plot("Delta phases against phases")
        for i in range(self.n):
            plt.scatter(self.bogs[i].phases[0:-1], self.bogs[i].delta_phases, label = self.names[i], s = 5)
        plt.axhline((2 * np.pi) / self.bogs[i].reference_period, color = "black", linestyle = "--", label = "Expected delta phase")
        plt.xlabel("Phase")
        plt.ylabel("Delta phase")
        #plt.ylim(0, 2)
        plt.legend()
        plt.show()

    def plot_unwrapped_phases(self, subtract = False):
        """
        Plots the unwrapped phases against frame number

        Args:
            subtract (bool, optional): Whether to subtract a linear unwrapped phase from the unwrapped phases. Defaults to False.
        """        
        # TODO - Add a proper linear fit to the unwrapped phases
        self._check_if_run("get_unwrapped_phases")

        self._begin_plot("Unwrapped phases against frame number")
        for i in range(self.n):
            xs = range(self.bogs[i].unwrapped_phases.shape[0])
            if subtract:
                plt.scatter(xs, self.bogs[i].unwrapped_phases - xs, label = self.names[i], s = 5)
            else:
                plt.scatter(xs, self.bogs[i].unwrapped_phases, label = self.names[i], s = 5)
        plt.xlabel("Frame number")
        plt.ylabel("Unwrapped phase")
        plt.legend()
        plt.show()

    def plot_subframe_histogram(self, frame_number = None, bins = 20, skip = 0):
        """
        _summary_

        Args:
            frame_number (_type_, optional): Frame to plot, if None then plots every frame on one plot. Defaults to None.
            bins (int, optional): Number of bins for the histogram. Defaults to 20.
            skip (int, optional): Number of frames to skip. Useful if we want to ignore reference frames. Defaults to 0.
        """        
        self._check_if_run("get_phases")

        for i in range(self.n):
            if frame_number != None:
                subframes = []
                #for j, frame_minima in enumerate(self.bogs[i].frame_minimas):
                for j in range(skip, self.bogs[i].frame_minimas.shape[0]):
                    if self.bogs[i].frame_minimas[j] == frame_number:
                        subframes.append(self.bogs[i].frame_minimas[j] - self.bogs[i].phases[j])
            else:
                subframes = self.bogs[i].phases - self.bogs[i].frame_minimas

            self._begin_plot(f"Subframe histogram for {self.names[i]}")
            plt.hist(subframes, bins = bins)
            plt.xlabel("Subframe")
            plt.ylabel("Count")
            plt.show()

    def plot_subframe_histogram_full(self, bins = 100):
        """
        Plot the subframe histogram across every phase

        Args:
            bins (int, optional): _description_. Defaults to 100.
        """        
        self._check_if_run("get_phases")
        for i in range(len(self.bogs)):
            self._begin_plot(f"Subframe histogram for {self.names[i]}")
            plt.hist(self.bogs[i].phases, bins = bins)
            plt.xlabel("Frame")
            plt.ylabel("Counts")
            plt.show()

        """self._begin_plot("Subframe histogram for all")
        hists = []
        for i in range(len(self.bogs)):
            hists.append(self.bogs[i].phases)
        plt.hist(hists, bins = bins, label = self.names, histtype = "barstacked")
        plt.axhline(self.bogs[0].phases.shape[0] / bins, c = "black", linestyle = "--", label = "Expected count")
        plt.xlabel("Frame")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()"""

    def plot_video(self):
        self._check_if_run("set_sequence")

        for i in range(self.n):
            x, y, width, height = self.bogs[i].roi
            plt.imshow(self.bogs[i].sequence[0,x:x + width,y:y + height])
            plt.show()

    def plot_all(self, subtract = False):
        """
        Plots all the plots

        Args:
            subtract (bool, optional): Whether to do linear fir on bias correction and unwrapped phases. Defaults to False.
        """        
        # TODO: Linear fit for both unwrapped and bias correction as seperate vars
        self.plot_sads(0)
        self.plot_phases()
        self.plot_delta_phases()
        self.plot_delta_phases_phases()
        self.plot_unwrapped_phases(subtract)
        self.plot_bias_correction(subtract)
        self.plot_subframe_histogram_full()


    def __repr__(self) -> str:
        return_string = "Basic optical gating plotter\n"
        return_string += f"\tfigsize: {self.figsize}\n"
        return_string += f"\tBOGs:\n"
        for i, name in enumerate(self.names):
            return_string += f"\t\t{i} : {name}\n"

        return return_string
# !SECTION