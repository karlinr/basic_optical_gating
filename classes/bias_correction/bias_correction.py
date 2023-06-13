# General imports
import matplotlib.pyplot as plt
import numpy as np
import j_py_sad_correlation as jps
import tifffile as tf
import time
import gc
import glob
import plistlib

# Useful imports
from scipy.interpolate import CubicSpline

# Logger
from .logger import Logger, Colour

# SECTION: Bias correction methods
class AdaptiveSubframeEstimation(BasicOpticalGating):
    def __init__(self) -> None:
        super().__init__()

    def get_reference_sequence_lf(self, reduction = 2):
        """
        Finds where the reference sequence is located within our full sequence
        then selects 1 in every (reduction) frames and pads appropriately
        Replaces our reference sequence with a lower framerate version
        NOTE: Need to create a method to estimate the reference period also would be better
        to use OOG method to generate reference sequence from scratch and then perform framerate
        reduction
        NOTE: Doesn't work with synthetic data generation as the reference sequence is not neccesarily
        included in the sequence

        Args:
            reduction (int, optional): Frames to skip. Defaults to 1.
            offset (int, optional): Offset from high framerate sequence. Defaults to 0.
        """

        # Get our high framerate indices
        sad = []
        for j in range(self.sequence.shape[0]):
            sad.append(np.sum(np.abs(self.sequence[j] - self.reference_sequence[0])))
        reference_min = np.argmin(sad)
        reference_max = reference_min + self.reference_sequence.shape[0]
        reference_frame_hf_indices = np.arange(reference_min, reference_max)

        # Get our low framerate indices
        reference_frame_lf_indices = reference_frame_hf_indices[2:-2][::reduction]
        reference_frame_lf_indices = np.insert(reference_frame_lf_indices, 0, reference_frame_lf_indices[0] - reduction)
        reference_frame_lf_indices = np.insert(reference_frame_lf_indices, 0, reference_frame_lf_indices[0] - reduction)
        reference_frame_lf_indices = np.append(reference_frame_lf_indices, reference_frame_lf_indices[-1] + reduction)
        reference_frame_lf_indices = np.append(reference_frame_lf_indices, reference_frame_lf_indices[-1] + reduction)

        # and our low framerate sequence
        self.reference_sequence_lf = self.sequence[reference_frame_lf_indices]
        self.reference_period_lf = self.reference_period / reduction

    def get_sads_lf(self):
        """ Get the SADs for every frame in our sequence"""       
        if self._check_if_run("set_sequence") == True and self._check_if_run("set_reference_sequence") == True:    
            self.logger.print_message("INFO", "Calculating SADs...")      
            self.sads = []
            self.drifts = []

            # Calculate SADs
            for i in range(self.sequence.shape[0]):
                frame = self._preprocess_frame(i)
                self.sads.append(self.get_sad(frame, self.reference_sequence_lf))

            self.sads_lf = np.array(self.sads)

            self.has_run["get_sads"] = True
            self.logger.print_message("SUCCESS", "SADs calculated")

            return self.sads_lf
        else:
            self.logger.print_message("ERROR", "Cannot calculate SADs without a sequence and reference sequence")

    def get_sads_region(self):
        return 0

    def get_phases(self):
        """ Get the phase estimates for our sequence"""        
        if self._check_if_run("get_sads") == True:
            self.logger.print_message("INFO", "Calculating phases...")
            self.phases = []
            self.frame_minimas = []

            # Track frames outside the reference period
            outside_errors = []
            
            # Get the frame estimates
            for i, sad in enumerate(self.sads):           
                # Get the frame estimate
                frame_minima = np.argmin(sad[2:-2]) + 2
                y_1 = sad[frame_minima - 1]
                y_2 = sad[frame_minima]
                y_3 = sad[frame_minima + 1]
                
                subframe_minima = v_fitting_standard(y_1, y_2, y_3)[0]

                if abs(subframe_minima) > 0.5:
                    self.logger.print_message("WARNING", f"Subframe minima outside range {subframe_minima}")

                #self.phases.append(((frame_minima - 2 + subframe_minima) / self.reference_period) * (2 * np.pi))
                self.phases.append(frame_minima - 2 + subframe_minima)

            self.phases = np.array(self.phases)
            self.frame_minimas = np.array(self.frame_minimas)
            self.drifts = np.array(self.drifts)

            self.has_run["get_phases"] = True
            self.logger.print_message("SUCCESS", "Phases calculated")

    def run(self, clear_memory = False):
        """ Run the full optical gating on our sequence.
            Outputs the phases, delta phases and unwrapped phases.
            Optionally run bias correct and clear memory after completion

            Args:
                bias_correct (bool, optional): Run with bias correction. Defaults to False.
                clear_memory (bool, optional): Delete our sequence from memory. Defaults to False.
        """
        self.get_reference_sequence_lf()
        self.get_sads_lf()
        self.get_sads()
        self.get_phases()
        self.get_delta_phases()
        self.get_unwrapped_phases()

        if self.settings["clear_memory"]:
            if self.clear_memory():
                self.logger.print_message("WARNING", "Cleared memory, rerun set_sequence if SADs need to be regenerated.")


        self.logger.print_message("SUCCESS", "Finished processing sequence.")

class SignalConvolution(BasicOpticalGating):
    def __init__(self) -> None:
        super().__init__()

    def get_reference_sads(self):
        sad = []
        for j in range(self.sequence.shape[0]):
            sad.append(np.sum(np.abs(self.sequence[j] - self.reference_sequence[0])))
        reference_min = np.argmin(sad)
        reference_max = reference_min + self.reference_sequence.shape[0]
        reference_frame_hf_indices = np.arange(reference_min, reference_max)

        reference_sequence = self.reference_sequence[reference_frame_hf_indices]

        sads = []
        for i in range(reference_sequence.shape[0]):
            sad = []
            for j in range(reference_sequence.shape[0]):
                sad.append(reference_sequence[i].astype(np.int32) - reference_sequence[j].astype(np.int32))
            sads.append(sad)

        plt.plot(sads[0])

class LinearExpansion(BasicOpticalGating):
    def __init__(self) -> None:
        super().__init__()

        self.settings["matching_method"] = "SSD"
        
        self.I_0p_I_0p = []
        self.I_0p_I_1 = []
        self.I_0p_I_1 = []
        self.I_1_I_1 = []
        self.I_1_I_1p = []
        self.I_1p_I_1p = []

    def get_bias_correction(self):
        reference_sequence = self._preprocess_reference_sequence_bias_correction()

        for i in range(1, reference_sequence.shape[0] - 1):
            y_1 = reference_sequence[i - 1]
            y_2 = reference_sequence[i]
            y_3 = reference_sequence[i + 1]

            y_mean = (y_1 + y_2 + y_3) / 3
            m = (y_1 - y_mean) + (y_3 - y_mean)
            print(m)

class QuadraticModel(BasicOpticalGating):
    def __init__(self) -> None:
        super().__init__()
        self.settings["matching_method"] == "SSD"

        self.bc_I_1_squared = []
        self.bc_I_2_squared = []
        self.bc_I_1_times_I_2 = []

    def get_bias_correction(self):
    # Get our ssd bias correction
        reference_sequence = self._preprocess_reference_sequence_bias_correction()

        for i in range(1, reference_sequence.shape[0] - 1):
            y_1 = reference_sequence[i - 1]
            y_2 = reference_sequence[i]
            y_3 = reference_sequence[i + 1]

            I_0 = y_2
            I_2 = (y_1 + y_3 - 2 * y_2) / 2
            I_1 = y_3 - I_2 - I_0

            I_1_squared = np.sum(I_1**2)
            I_2_squared = np.sum(I_2**2)
            I_1_times_I_2 = np.sum(I_1 * I_2)

            self.bc_I_1_squared.append(I_1_squared)
            self.bc_I_2_squared.append(I_2_squared)
            self.bc_I_1_times_I_2.append(I_1_times_I_2)

    def get_phases(self):
        if self._check_if_run("get_sads") == True:
            from scipy.optimize import curve_fit

            self.logger.print_message("INFO", "Calculating phases...")
            self.phases = []
            self.frame_minimas = []

            for i, sad in enumerate(self.sads):
                frame_minima = np.argmin(sad[2:-2]) + 2

                def fitting_function(t, t_n, c, smush):
                    I_1_squared = self.bc_I_1_squared[frame_minima - 1]
                    I_2_squared = self.bc_I_2_squared[frame_minima - 1]
                    I_1_times_I_2 = self.bc_I_1_times_I_2[frame_minima - 1]

                    return smush * (I_1_squared * (t - t_n)**2 + I_2_squared * (t**2 - t_n**2)**2 + 2 * I_1_times_I_2 * (t - t_n) * (t**2 - t_n**2) + c)

                # Get the frame estimate
                y_1 = sad[frame_minima - 1]
                y_2 = sad[frame_minima]
                y_3 = sad[frame_minima + 1]

                xs = np.linspace(-1, 1, 3)
                popt, popc = curve_fit(fitting_function, xs, sad[frame_minima - 1:frame_minima + 2], maxfev = 8000)
                self.phases.append(frame_minima - 2 + popt[0])

            self.phases = np.array(self.phases)

            self.has_run["get_phases"] = True


    def run(self):
        self.get_bias_correction()
        super().run()

class AdaptedV(BasicOpticalGating):
    def __init__(self) -> None:
        super().__init__()

    def _preprocess_reference_sequence_bias_correction(self, frame_number = None):
        # Pre-process the reference sequence for bias correction
        # NOTE: This is a separate function so that it can be overriden by subclasses
        if self.roi is not None:
            x, y, width, height = self.roi
            return self.reference_sequence[:, y:y+height, x:x+width].astype(np.int64)
        else:
            return self.reference_sequence.astype(np.int64)

    def get_bias_correction(self):
        """ Get the bias correction. Attempts to reduce biasing caused by varying rate of change in
            the sum of absolute differences between neighbouring frames which causes V-fitting to be biased.
            Currently in testing but shown to be effective in reducing bias in synthetic data.
        """       
        if self._check_if_run("set_reference_sequence") == True:   
            self.logger.print_message("INFO", "Calculating bias correction...")
            # Pre-process the reference sequence - used for subclassing
            reference_sequence = self._preprocess_reference_sequence_bias_correction()

            # Get the difference between neighbouring reference frames
            diffs = [0]
            for i in range(1, reference_sequence.shape[0]):
                diff = reference_sequence[i] - reference_sequence[i - 1]
                #diff = diff[diff < 0]
                diffs.append(np.sum(np.abs(diff)))
            diffs = np.array(diffs)

            if np.any(diffs < 0):
                self.logger.print_message("WARNING", "Negative diffs found")

            # Get running sum of differences
            diffs = np.cumsum(diffs)

            # Normalise - not strictly neccessary but makes it easier to compare plots
            diffs = diffs / diffs[-1]
            diffs *= (diffs.shape[0] - 1)

            # Create a cubic spline to interpolate the differences
            xs = np.arange(len(diffs))
                                
            self.cs = CubicSpline(xs, diffs)

            self.has_run["get_bias_correction"] = True
            self.logger.print_message("SUCCESS", "Bias correction complete")

    
    def get_phases(self):
        """ Get the phase estimates for our sequence"""        
        if self._check_if_run("get_sads") == True:
            self.logger.print_message("INFO", "Calculating phases...")
            self.phases = []
            self.frame_minimas = []

            # Track frames outside the reference period
            outside_errors = []
            
            # Get the frame estimates
            for i, sad in enumerate(self.sads):           
                # Get the frame estimate
                frame_minima = np.argmin(sad[2:-2]) + 2
                y_1 = sad[frame_minima - 1]
                y_2 = sad[frame_minima]
                y_3 = sad[frame_minima + 1]
                x_1 = frame_minima - 1
                x_2 = frame_minima
                x_3 = frame_minima + 1

                if y_1 > y_2 and y_3 > y_2:
                    # Do v-fitting and bias correction
                    # NOTE: This is an incredibly inneficient way of doing this - if implementing in production code we'll find an analytic 
                    # expression for the bias correction

                    # Get the v-fitting parameters
                    m_1_l, m_3_l, c_1_l, c_3_l, m_3_r, m_1_r, c_1_r, c_3_r = v_fitting(y_1, y_2, y_3, x_1, x_2, x_3)

                    # Generate a sequence used for finding the minima
                    cdiff_l = self.cs(x_2) - self.cs(x_1)
                    cs_normalised_l = lambda x: (self.cs(x) - self.cs(x_2)) / cdiff_l
                    cs_normalised_r = lambda x: (self.cs(x) - self.cs(x_3)) / cdiff_l
                    xs = np.linspace(x_1, x_3, 10000)

                    # Do fitting using corrected V
                    v_l_c_l = m_1_l * (x_2 + cs_normalised_l(xs)) + c_1_l
                    v_r_c_l = m_3_l * (x_3 + cs_normalised_r(xs)) + c_3_l
                    # Get minima location
                    x_min_l = (xs)[np.argmin(abs(v_l_c_l - v_r_c_l))]
                    y_min_l = m_3_l * (x_3 + cs_normalised_r(x_min_l)) + c_3_l

                    # Do fitting using corrected V
                    cdiff_r = self.cs(x_3) - self.cs(x_2)
                    cs_normalised_l = lambda x: (self.cs(x) - self.cs(x_1)) / cdiff_r
                    cs_normalised_r = lambda x: (self.cs(x) - self.cs(x_2)) / cdiff_r
                    
                    v_l_c_r = m_1_r * (x_1 + cs_normalised_l(xs)) + c_1_r
                    v_r_c_r = m_3_r * (x_2 + cs_normalised_r(xs)) + c_3_r
                    # Get minima location
                    x_min_r = (xs)[np.argmin(abs(v_l_c_r - v_r_c_r))]
                    y_min_r = m_3_r * (x_2 + cs_normalised_r(x_min_r)) + c_3_r

                    if y_min_l < y_min_r:
                        #self.phases.append(x_min_l - 2)
                        self.phases.append(((x_min_l - 2) / self.reference_period) * (2 * np.pi))
                    else:
                        #self.phases.append(x_min_r - 2)
                        self.phases.append(((x_min_r - 2) / self.reference_period) * (2 * np.pi))

                    self.frame_minimas.append(frame_minima - 2)
                else:
                    outside_errors.append(i)
                    self.phases.append(frame_minima - 2)
                    self.frame_minimas.append(frame_minima - 2)

            if len(outside_errors) > 0:
                self.logger.print_message("WARNING", f"Minima for frame(s) {outside_errors} outside of the valid range.")


            self.phases = np.array(self.phases)
            self.frame_minimas = np.array(self.frame_minimas)
            self.drifts = np.array(self.drifts)

            self.has_run["get_phases"] = True
            self.logger.print_message("SUCCESS", "Phases calculated")

    def run(self):
        self.get_bias_correction()
        super().run()

# !SECTION